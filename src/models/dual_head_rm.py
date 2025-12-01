# %%writefile src/models/dual_head_rm.py
# src/models/dual_head_rm.py
"""
Dual-head Reward Model (ORM + PRM)
=================================

File: src/models/dual_head_rm.py

This module implements a dual-head reward model intended for:
 - ORM: output a single solution-level correctness logit (binary).
 - PRM: output per-step quality scores (continuous in [0,1]) for up to `max_steps`.

Design notes / assumptions
--------------------------
- The base encoder is a pretrained transformer model (decoder-only or encoder-decoder).
  We load it with `AutoModel.from_pretrained` to obtain hidden states.
- Pooling strategy to map token-level hidden states -> example-level vector:
    - "eos": use the hidden state at the tokenizer's eos_token_id position (preferred for causal/decode-only models).
    - "last": use the last hidden state in the sequence (simple fallback).
    - "mean": mean-pool non-padded tokens.
- PRM head: a simple MLP that outputs `max_steps` real values per example.
  - After forward, the training pipeline should apply sigmoid() to get values in [0,1].
  - The model exposes both raw (logit) and sigmoid outputs for convenience.
- Loss helpers are included:
  - `orm_loss` uses `BCEWithLogitsLoss`.
  - `prm_masked_mse` uses masked MSE over valid steps (given `step_mask`).
  - `prm_solution_loss` supports aggregating PRM step scores by `mean` or `product`
    (paper-faithful uses `product` of normalized step scores) and computing MSE
    vs. a provided scalar target (if available).
- The module tries to be device-agnostic and dtype-agnostic, reusing the base model dtype.

Usage
-----
>>> model = DualHeadRM.from_pretrained("facebook/galactica-6.7b", max_steps=10)
>>> out = model(input_ids, attention_mask, step_mask=step_mask)
>>> out["orm_logits"]         # shape (B,)
>>> out["prm_logits"]         # shape (B, max_steps)
>>> out["prm_scores"]         # sigmoid(prm_logits) in [0,1]

Author: Generated for Stage 3.3 (Aklesh Mishra project)
"""

from typing import Optional, Dict, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModel, AutoConfig, PreTrainedModel
import os

# Hint PyTorch allocator to use expandable segments to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

# -------------------------
# Utility functions
# -------------------------
def _mask_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """
    Mean pool hidden_states (B, T, H) using attention_mask (B, T).
    Returns (B, H).
    """
    att = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (B, T, 1)
    summed = (hidden_states * att).sum(dim=1)  # (B, H)
    denom = att.sum(dim=1).clamp_min(1e-6)  # (B, 1)
    return summed / denom


# -------------------------
# Dual-head model
# -------------------------
class DualHeadRM(nn.Module):
    """
    Dual-head Reward Model.

    Args:
        base_model_name: huggingface model id to load the encoder (e.g., "facebook/galactica-6.7b")
        max_steps: maximum number of steps the PRM head predicts per chain
        pooling: pooling strategy to derive example-level representation from token hidden states;
                 one of {"eos", "last", "mean"}.
        hidden_dropout: dropout before heads
        prm_hidden_dim: optional hidden dim for PRM head MLP; if None, uses base hidden size.
        freeze_base: whether to freeze base model weights by default.
    """

    def __init__(
        self,
        base_model_name: str,
        max_steps: int = 10,
        pooling: str = "eos",
        hidden_dropout: float = 0.1,
        prm_hidden_dim: Optional[int] = None,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.max_steps = int(max_steps)
        self.pooling = pooling
        self.hidden_dropout = hidden_dropout
        self.device = device

        torch.cuda.empty_cache()

        # Load config + base model (we only need hidden states)
        config = AutoConfig.from_pretrained(
            base_model_name,
            output_hidden_states=False,  # saves memory
            return_dict=True,  # clean outputs
            trust_remote_code=False,
        )

        # Create empty shell (no weight allocation)
        with init_empty_weights():
            base = AutoModel.from_config(config)

        # Memory-safe weight load + dispatch across GPUs
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if "LOCAL_RANK" in os.environ:
            chosen_map = {"": f"cuda:{local_rank}"}
        else:
            # single-process evaluation: force single GPU (cuda:0)
            chosen_map = {"": "cuda:0"}

        self.base = load_checkpoint_and_dispatch(
            base,
            checkpoint=base_model_name,
            device_map=chosen_map,
            dtype=torch.float16,
        )
        print(f"[Init] Accelerate device_map used: {chosen_map}")

        hf_map = getattr(self.base, "hf_device_map", None) or getattr(self.base, "device_map", None)
        print("[Init] base.hf_device_map/device_map:", hf_map)
        # show a few param devices
        count = 0
        for n, p in self.base.named_parameters():
            if count < 3:
                print(f"[Init] base param sample: {n} -> {p.device}, dtype={p.dtype}")
            count += 1
            if count >= 20:
                break

        # ------------------------------------------------------
        # 🧠 Memory optimization for multi-GPU setups (OPT-2.7B safe)
        # ------------------------------------------------------

        # Enable gradient checkpointing — reduces activation memory by ~40%
        if hasattr(self.base, "gradient_checkpointing_enable"):
            try:
                self.base.gradient_checkpointing_enable()
                print("[Init] ✅ Gradient checkpointing enabled for base model.")
            except Exception as e:
                print(f"[Init] ⚠️ Could not enable gradient checkpointing: {e}")

        # Allow TF32 matmul kernels on Ampere+ GPUs (minor VRAM + speed gain)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[Init] ✅ TF32 compute enabled (safe on T4/Ampere).")
        except Exception:
            pass

        # detect hidden size
        hidden_size = getattr(self.base.config, "hidden_size", None)
        if hidden_size is None:
            # fallback (some models use d_model)
            hidden_size = getattr(self.base.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from base model config.")

        self.hidden_size = int(hidden_size)
        prm_hid = prm_hidden_dim or self.hidden_size

        # Optionally freeze base parameters (useful for small-data / LoRA workflows)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Dropout layer
        self.drop = nn.Dropout(hidden_dropout)

        # ORM head: single logit
        self.orm_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(self.hidden_size, 1),  # single logit
        )

        # PRM head: map pooled representation -> per-step logits (max_steps)
        # We'll use an MLP to produce a vector of size max_steps.
        self.prm_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, prm_hid),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(prm_hid, self.max_steps),
        )

        # --- Force ORM/PRM heads to match base model dtype ---
        base_dtype = next(self.base.parameters()).dtype
        self.orm_proj = self.orm_proj.to(dtype=base_dtype)
        self.prm_mlp = self.prm_mlp.to(dtype=base_dtype)
        print(f"[Init] ORM/PRM heads casted to {base_dtype}.")

        # Initialization helpers
        self._init_weights(self.orm_proj)
        self._init_weights(self.prm_mlp)

        # --- Device handling ---
        hf_device_map = getattr(self.base, "hf_device_map", None) or getattr(self.base, "device_map", None)
        used_device_map = hf_device_map is not None
        if device is not None and not used_device_map:
            # Only move if we didn’t already let HF dispatch layers
            self.to(device)

    # -------------------------
    # weight init
    # -------------------------
    @staticmethod
    def _init_weights(module: nn.Module):
        """
        Simple Xavier init for linear layers in MLP heads.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------
    # factory
    # -------------------------
    @classmethod
    def from_pretrained(cls, base_model_name: str, **kwargs):
        """
        Convenience constructor to match naming used elsewhere.
        """
        return cls(base_model_name=base_model_name, **kwargs)

    # -------------------------
    # pooling
    # -------------------------
    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, input_ids: Optional[torch.LongTensor] = None):
        """
        Pool token-level hidden states -> example vector (B, H).
        - last_hidden_state: (B, T, H)
        - attention_mask: (B, T)
        - input_ids: optional (B, T) to locate eos positions when pooling='eos'
        """
        if self.pooling == "mean":
            return _mask_mean(last_hidden_state, attention_mask)
        elif self.pooling == "last":
            # assume right-padded; pick last non-pad token index per sample
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            lengths = lengths.clamp_min(0).long()
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, last_hidden_state.size(-1))  # (B,1,H)
            return last_hidden_state.gather(dim=1, index=idx).squeeze(1)  # (B,H)
        elif self.pooling == "eos":
            # try to find eos token position if input_ids provided; otherwise fallback to last
            if input_ids is not None:
                # try to locate eos token id (could be none)
                # Make a best-effort: find last position where token == eos_token_id
                eos_id = getattr(self.base.config, "eos_token_id", None)
                if eos_id is not None:
                    # (B, T) boolean mask where input == eos
                    eos_mask = (input_ids == eos_id) & (attention_mask.bool())
                    # If any eos found, take the last eos index; else fallback to last non-pad
                    has_eos = eos_mask.any(dim=1)
                    pooled = []
                    for i in range(input_ids.size(0)):
                        if has_eos[i]:
                            pos = torch.where(eos_mask[i])[0].max()
                            pooled.append(last_hidden_state[i, pos])
                        else:
                            # fallback to last non-pad
                            last_pos = attention_mask[i].sum().clamp_min(1).long() - 1
                            pooled.append(last_hidden_state[i, last_pos])
                    return torch.stack(pooled, dim=0)

            # fallback to "last" pooling when no input_ids available (avoid recursion)
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            lengths = lengths.clamp_min(0).long()
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, last_hidden_state.size(-1))  # (B,1,H)
            return last_hidden_state.gather(dim=1, index=idx).squeeze(1)  # (B,H)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    # -------------------------
    # forward
    # -------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        step_mask: Optional[torch.FloatTensor] = None,
        step_targets: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: (B, T)
            attention_mask: (B, T) or None
            step_mask: (B, max_steps) float/boolean mask indicating valid step positions. If None, all steps are considered valid.
            step_targets: optional, (B, max_steps) float targets in [0,1] for losses calculation (not used by base forward).
            return_dict: whether to return a dict or tuple.

        Returns: dict with keys:
            - orm_logits: (B,) raw logits
            - prm_logits: (B, max_steps) raw logits (unbounded)
            - prm_scores: (B, max_steps) sigmoid(prm_logits) in [0,1]
            - pooled_repr: (B, H) pooled vector
            - other optional debug fields
        """
        bsz = input_ids.size(0)
        if attention_mask is None:
            # Best-effort: assume pad_token_id == 0; callers should pass mask when possible
            attention_mask = (input_ids != 0).long()

        # Base model forward
        base_outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # prefer last_hidden_state attribute; depending on model it might be `.last_hidden_state`
        last_hidden_state = getattr(base_outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            # some models return hidden_states in .hidden_states[-1] if configured; try that
            hidden_states = getattr(base_outputs, "hidden_states", None)
            if hidden_states:
                last_hidden_state = hidden_states[-1]
            else:
                raise RuntimeError("Base model did not return token hidden states. Check model config.")

        base_device = next(self.base.parameters()).device
        if last_hidden_state.device != base_device:
            print(f"[WARN] Base outputs were on {last_hidden_state.device}, moving to {base_device}")
            last_hidden_state = last_hidden_state.to(base_device, non_blocking=True)

        # --- Force last_hidden_state dtype to match ORM/PRM heads (fix Half vs Float) ---
        head_dtype = next(self.orm_proj.parameters()).dtype
        if last_hidden_state.dtype != head_dtype:
            last_hidden_state = last_hidden_state.to(dtype=head_dtype)

        # Pool to get example representation
        pooled = self._pool(last_hidden_state, attention_mask, input_ids)

        # Ensure pooled representation is on same device and dtype as ORM/PRM heads
        head_param = next(self.orm_proj.parameters())
        head_device = head_param.device
        head_dtype = head_param.dtype
        pooled = pooled.to(device=head_device, dtype=head_dtype, non_blocking=True)

        pooled = self.drop(pooled)  # (B, H)

        pooled_fp32 = pooled.to(dtype=torch.float32)
        orm_logits = self.orm_proj.float()(pooled_fp32).squeeze(-1)
        prm_logits = self.prm_mlp.float()(pooled_fp32)

        # Clamp to safe range
        orm_logits = torch.clamp(orm_logits, -15.0, 15.0)
        prm_logits = torch.clamp(prm_logits, -15.0, 15.0)

        orm_logits = orm_logits.to(dtype=next(self.base.parameters()).dtype)
        prm_logits = prm_logits.to(dtype=next(self.base.parameters()).dtype)

        prm_scores = torch.sigmoid(prm_logits)  # normalized in [0,1]

        # Replace NaNs if any sneak through
        orm_logits = torch.nan_to_num(orm_logits, nan=0.0, posinf=15.0, neginf=-15.0)
        prm_logits = torch.nan_to_num(prm_logits, nan=0.0, posinf=15.0, neginf=-15.0)
        out = {
            "orm_logits": orm_logits,
            "prm_logits": prm_logits,
            "prm_scores": prm_scores,
            "pooled_repr": pooled,
        }

        # --- Ensure all outputs are on GPU (Accelerate may offload some to CPU) ---
        out_device = next(self.orm_proj.parameters()).device
        for k, v in out.items():
            if torch.is_tensor(v):
                out[k] = v.to(out_device, non_blocking=True)

        if return_dict:
            return out
        else:
            return orm_logits, prm_logits, prm_scores

    # -------------------------
    # losses helpers
    # -------------------------
    def orm_loss(self, orm_logits: torch.Tensor, orm_labels: torch.Tensor, pos_weight: Optional[float] = None):
        """
        Compute ORM BCEWithLogitsLoss.
        - orm_logits: (B,)
        - orm_labels: (B,) {0,1} float
        - pos_weight: optional weight for positive class (to handle imbalance)
        """
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=orm_logits.dtype, device=orm_logits.device)
            loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            loss_f = nn.BCEWithLogitsLoss()
        return loss_f(orm_logits.view(-1), orm_labels.view(-1).type_as(orm_logits))

    def prm_masked_mse(self, prm_logits: torch.Tensor, step_targets: torch.Tensor, step_mask: torch.Tensor):
        """
        Masked MSE for PRM step-wise targets.
        - prm_logits: (B, max_steps) raw logits -> apply sigmoid inside
        - step_targets: (B, max_steps) floats in [0,1]
        - step_mask: (B, max_steps) 0/1 mask where 1 indicates valid step
        Returns scalar MSE over valid positions.
        """
        # convert to scores
        preds = torch.sigmoid(prm_logits)
        mask = step_mask.type_as(preds)
        assert preds.shape == step_targets.shape == mask.shape, "Shapes must match (B, max_steps)"
        diff = (preds - step_targets) * mask
        # compute mean over valid entries
        denom = mask.sum().clamp_min(1.0)
        loss = (diff * diff).sum() / denom
        return loss

    def prm_solution_loss(self, prm_logits: torch.Tensor, step_mask: torch.Tensor, gold_solution: Optional[torch.Tensor] = None, agg: str = "product"):
        """
        Aggregate per-step PRM scores into a single solution-level scalar using
        either 'product' or 'mean' aggregation, then compute MSE vs gold_solution
        if provided.

        - prm_logits: (B, max_steps)
        - step_mask: (B, max_steps)
        - gold_solution: (B,) target scalar in [0,1] (optional)
        - agg: 'product' | 'mean'
        Returns: (agg_values, loss_or_none)
        """
        scores = torch.sigmoid(prm_logits) * step_mask.type_as(prm_logits)
        # Avoid zeros in product by clamping tiny positive epsilon when using product
        if agg == "product":
            # product over valid steps: for each sample, take product of scores where mask==1
            # To make product numerically stable, we compute sum(log(scores + eps)) and exp
            eps = 1e-10
            masked = scores + eps
            # set masked positions (where mask==0) to 1 so they don't affect product
            masked = masked + (1.0 - step_mask.type_as(masked))
            log_sum = torch.log(masked).sum(dim=1)  # (B,)
            agg_vals = torch.exp(log_sum)  # (B,)
        elif agg == "mean":
            denom = step_mask.sum(dim=1).clamp_min(1.0)
            agg_vals = (scores.sum(dim=1) / denom)
        else:
            raise ValueError("agg must be 'product' or 'mean'")

        if gold_solution is None:
            return agg_vals, None
        # gold_solution expected in [0,1] floats
        loss = F.mse_loss(agg_vals, gold_solution.type_as(agg_vals))
        return agg_vals, loss

    # -------------------------
    # utilities
    # -------------------------
    def freeze_base(self):
        """Freeze base model parameters."""
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base parameters."""
        for p in self.base.parameters():
            p.requires_grad = True

    def resize_prm_head(self, new_max_steps: int):
        """
        Rebuild PRM head to support a different max_steps.
        Note: this will re-init the PRM head weights.
        """
        new_max = int(new_max_steps)
        prm_hid = self.prm_mlp[0].in_features if isinstance(self.prm_mlp[0], nn.Linear) else self.hidden_size
        self.max_steps = new_max
        self.prm_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, prm_hid),
            nn.ReLU(),
            nn.Dropout(self.hidden_dropout),
            nn.Linear(prm_hid, self.max_steps),
        )
        self._init_weights(self.prm_mlp)

    # -------------------------
    # debug string
    # -------------------------
    def extra_repr(self):
        return f"base={self.base_model_name} hidden={self.hidden_size} max_steps={self.max_steps} pooling={self.pooling}"

