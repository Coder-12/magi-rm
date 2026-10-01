# src/model.py
"""
RewardModel wrapper.

- Uses a causal LM backbone (AutoModelForCausalLM) as encoder.
- Pools last hidden states (by EOS token if present or mean pooling).
- Two heads:
    - orm_head: linear -> scalar logits (BCEWithLogitsLoss)
    - prm_head: linear -> scalar regression (MSELoss)
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, PreTrainedTokenizerBase


class RewardModel(nn.Module):
    def __init__(self, backbone_name: str, device: Optional[torch.device] = None):
        super().__init__()
        # Load backbone
        self.backbone_name = backbone_name
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_name, output_hidden_states=True)
        hidden_size = self.backbone.config.hidden_size

        # Two separate heads
        self.orm_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)  # returns logits
        )

        self.prm_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),  # regression scalar in [0..1] (we'll not enforce activation here)
            # Note: training uses MSE against normalized target
        )

        if device is not None:
            self.to(device)

    def _get_pooled_representation(self, input_ids, attention_mask, tokenizer):
        """
        Get a pooled hidden vector per sample.
        Strategy:
        - If tokenizer has eos_token_id, prefer last occurrence per sequence (common pattern).
        - Otherwise, do mean pooling over masked positions.
        """
        # forward through backbone to get hidden states
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # last_hidden_state shape: (batch, seq_len, hidden)
        last_hidden = outputs.hidden_states[-1]  # already the final layer
        # try to find eos positions if possible
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            # find last eos token index per sample; if none, fallback to mean pooling
            batch_eos_vectors = []
            input_ids_cpu = input_ids.detach().cpu().numpy()
            has_eos = False
            for b_idx, seq in enumerate(input_ids_cpu):
                # find last pos of eos_id in seq
                positions = (seq == eos_id).nonzero()[0]
                if positions.size > 0:
                    has_eos = True
                    pos = int(positions[-1])
                    vec = last_hidden[b_idx, pos, :]
                else:
                    vec = last_hidden[b_idx].masked_fill(~attention_mask[b_idx].unsqueeze(-1).bool(), 0.0).sum(dim=0) / (attention_mask[b_idx].sum().clamp(min=1).float())
                batch_eos_vectors.append(vec)
            if has_eos:
                return torch.stack(batch_eos_vectors, dim=0)
            # else fallback
        # mean pooling
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(last_hidden.dtype)
        mean_pooled = summed / lengths
        return mean_pooled

    def forward(self, input_ids, attention_mask, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        """
        Returns dict: {'orm_logits': (B,1), 'prm_pred': (B,1)}
        """
        # Get pooled hidden representation
        pooled = self._get_pooled_representation(input_ids, attention_mask, tokenizer)
        orm_logits = self.orm_head(pooled).squeeze(-1)
        prm_pred = self.prm_head(pooled).squeeze(-1)
        return {"orm_logits": orm_logits, "prm_pred": prm_pred}
