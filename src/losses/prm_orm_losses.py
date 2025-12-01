# %%writefile src/losses/prm_orm_losses.py
# src/losses/prm_orm_losses.py
"""
Loss utilities for Dual-Head Reward Model (PRM + ORM) — final polished version.

Features (complete):
 - Masked per-step PRM loss (SmoothL1 / Huber) with optional per-step confidence.
 - Mean- and product-aggregated PRM losses (product has numerics-safe mean-normalized log-sum).
 - Product loss supports log-space MSE for numerical stability.
 - ORM BCE loss with optional label smoothing and pos_weight.
 - Combined wrapper with:
     * optional chain-level confidence weighting (downweights PRM terms per example)
     * option to treat step_preds as logits (automatic sigmoid)
     * optional loss clipping (cap total loss)
 - Per-example computations when chain-level confidence is provided,
   otherwise returns batch-averaged scalars (backwards-compatible).
 - Returns total_loss (scalar tensor) and a components dict (detached tensors suitable for logging).

Backward compatible defaults chosen to follow the "hybrid / conservative" strategy:
 - λ_orm = 1.0 (strong)
 - λ_prm_step = 0.4 (moderate)
 - λ_prm_prod = 0.4 (moderate)
 - λ_consistency = 0.1 (small)
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helpers
# -------------------------
def _safe_masked_sum(x: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = x * mask
    denom = mask.sum(dim=dim).clamp_min(eps)
    return masked.sum(dim=dim), denom


def _safe_masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    s, denom = _safe_masked_sum(x, mask, dim=dim, eps=eps)
    return s / denom


# -------------------------
# PRM Aggregation
# -------------------------
def aggregate_prm_mean(step_preds: torch.Tensor, step_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean of step_preds over valid steps per example. Returns (B,)"""
    return _safe_masked_mean(step_preds, step_mask, dim=1, eps=eps)


def aggregate_prm_product(step_preds: torch.Tensor, step_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Naive product aggregation (B, S) -> (B,). Numerically safe via log.
    Result is in (0, 1], zero if no valid steps.
    """
    safe_preds = step_preds.clamp(min=eps, max=1.0)
    log_preds = torch.log(safe_preds) * step_mask
    log_prod = log_preds.sum(dim=1)
    n_valid = step_mask.sum(dim=1)
    prod = torch.exp(log_prod)
    prod = torch.where(n_valid > 0, prod, torch.zeros_like(prod))
    return prod


def aggregate_prm_product_safe(step_preds: torch.Tensor, step_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Numerically stable, length-normalized product aggregation.

    Computes mean(log(pred_i)) over valid steps, then exp(mean). This avoids
    extremely small values for long chains and produces a length-normalized
    product score in (0,1].

    Returns (B,)
    """
    safe_preds = step_preds.clamp(min=eps, max=1.0)
    log_preds = torch.log(safe_preds)
    masked_log = log_preds * step_mask
    n_valid = step_mask.sum(dim=1).clamp_min(1.0)
    log_prod_mean = masked_log.sum(dim=1) / n_valid
    prod_safe = torch.exp(log_prod_mean)
    prod_safe = torch.where(step_mask.sum(dim=1) > 0, prod_safe, torch.zeros_like(prod_safe))
    return prod_safe


# -------------------------
# PRM Losses
# -------------------------
def stepwise_prm_loss(
    step_preds: torch.Tensor,
    step_targets: torch.Tensor,
    step_mask: torch.Tensor,
    step_confidence: Optional[torch.Tensor] = None,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Masked SmoothL1 (Huber-like) loss for per-step PRM regression, with optional
    per-step confidence weighting.

    Args:
        step_preds: (B, S) predicted probabilities in [0,1] (NOT logits)
        step_targets: (B, S) targets in [0,1]
        step_mask: (B, S) 0/1 mask of valid steps
        step_confidence: optional (B, S) weights in (0,1], default=1s
        delta: Huber delta
        reduction: 'mean' (default) | 'sum' | 'none' (per-example)

    Returns:
        scalar loss (if reduction != 'none') or per-example (B,) tensor
    """
    if step_confidence is None:
        step_confidence = torch.ones_like(step_mask, dtype=step_preds.dtype, device=step_preds.device)

    diff = step_preds - step_targets
    abs_diff = diff.abs()
    # Huber-ish formulation
    loss_elem = torch.where(abs_diff < delta, 0.5 * diff**2 / delta, abs_diff - 0.5 * delta)

    # apply mask & confidence
    weighted = loss_elem * step_mask * step_confidence

    # per-example normalization by sum(step_mask * confidence)
    denom = (step_mask * step_confidence).sum(dim=1).clamp_min(1.0)
    per_example = weighted.sum(dim=1) / denom

    if reduction == "sum":
        return per_example.sum()
    elif reduction == "none":
        return per_example
    else:
        return per_example.mean()


def prm_solution_mean_loss(
    step_preds: torch.Tensor,
    step_targets: torch.Tensor,
    step_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    L1 loss between predicted and target step-wise mean aggregated PRM.
    Returns per-example (B,) when reduction == 'none'.
    """
    pred_mean = aggregate_prm_mean(step_preds, step_mask)
    tgt_mean = aggregate_prm_mean(step_targets, step_mask)
    loss = F.l1_loss(pred_mean, tgt_mean, reduction="none")
    if reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        return loss.mean()


def prm_solution_product_loss(
    step_preds: torch.Tensor,
    step_targets: torch.Tensor,
    step_mask: torch.Tensor,
    safe_mode: bool = True,
    use_log_mse: bool = True,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss on product-aggregated PRM predictions vs targets.

    Args:
        safe_mode: use aggregate_prm_product_safe (mean-normalized log-sum)
        use_log_mse: if True compute MSE on log-space: MSE(log(pred), log(tgt))
                     (recommended for numerical stability); else L1 on prod.
        eps: small constant for log stability
    """
    agg_fn = aggregate_prm_product_safe if safe_mode else aggregate_prm_product
    pred_prod = agg_fn(step_preds, step_mask, eps=eps)
    tgt_prod = agg_fn(step_targets, step_mask, eps=eps)

    if use_log_mse:
        lp = torch.log(pred_prod.clamp_min(eps))
        lt = torch.log(tgt_prod.clamp_min(eps))
        lp = lp.float()
        lt = lt.float()
        loss = F.mse_loss(lp, lt, reduction="none")
    else:
        loss = F.l1_loss(pred_prod, tgt_prod, reduction="none")

    if reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        return loss.mean()


# -------------------------
# ORM Loss
# -------------------------
def orm_bce_loss(
    orm_logits: torch.Tensor,
    orm_targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary cross-entropy with logits. Supports optional label smoothing.
    orm_logits: (B,) or (B,1), orm_targets: {0,1}
    label_smoothing: float in [0, 0.5). If >0, targets are clipped to [s, 1-s].
    """
    orm_logits = orm_logits.view(-1)
    orm_targets = orm_targets.view(-1).float()
    if label_smoothing > 0.0:
        s = float(label_smoothing)
        orm_targets = orm_targets.clamp(min=s, max=1.0 - s)

    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)(orm_logits, orm_targets)

    return F.binary_cross_entropy_with_logits(
        orm_logits,
        orm_targets.float(),
        reduction=reduction
    )

# -------------------------
# Combined Loss Wrapper (Hybrid / Conservative)
# -------------------------
def combined_prm_orm_loss(
    *,
    orm_logits: torch.Tensor,
    orm_targets: torch.Tensor,
    step_preds: torch.Tensor,
    step_targets: torch.Tensor,
    step_mask: torch.Tensor,
    step_confidence: Optional[torch.Tensor] = None,
    chain_confidence: Optional[torch.Tensor] = None,
    lambda_orm: float = 1.0,
    lambda_prm_step: float = 0.4,
    lambda_prm_mean: float = 0.0,
    lambda_prm_prod: float = 0.4,
    lambda_consistency: float = 0.1,
    step_preds_are_logits: bool = True,
    safe_prod: bool = True,
    pos_weight: Optional[torch.Tensor] = None,
    use_log_mse_for_prod: bool = True,
    label_smoothing: float = 0.0,
    loss_clip_max: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined total loss and return components dict.

    Chain-level confidence:
      - If `chain_confidence` provided (shape (B,)), PRM-related losses are
        weighted per-example by that confidence (higher confidence -> higher weight).
      - ORM loss is NOT scaled by chain_confidence (ORM is the main robust signal).

    Returns:
        total_loss (scalar), components dict (detached tensors and diagnostics)
    """
    MAX_LOGIT = 15.0  # sigmoid(±15) = [3e-7, 0.9999997], perfectly stable
    orm_logits = torch.clamp(orm_logits, -MAX_LOGIT, MAX_LOGIT)
    step_preds = torch.clamp(step_preds, -MAX_LOGIT, MAX_LOGIT)

    # convert preds if logits
    if step_preds_are_logits:
        step_probs = torch.sigmoid(step_preds)
    else:
        step_probs = step_preds

    step_probs = step_probs.float()
    step_targets = step_targets.float()
    step_mask = step_mask.float()

    # --- Sanity clamp / clean tensors to avoid NaN ---
    orm_logits = torch.nan_to_num(orm_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    step_preds = torch.nan_to_num(step_preds, nan=0.0, posinf=10.0, neginf=-10.0)

    # ensure targets are finite and within [0, 1]
    orm_targets = torch.clamp(torch.nan_to_num(orm_targets, nan=0.0), 0.0, 1.0)
    step_targets = torch.clamp(torch.nan_to_num(step_targets, nan=0.0), 0.0, 1.0)

    # ensure confidence scores are valid
    if step_confidence is not None:
        step_confidence = torch.clamp(torch.nan_to_num(step_confidence, nan=1.0), 0.0, 1.0)
    if chain_confidence is not None:
        chain_confidence = torch.clamp(torch.nan_to_num(chain_confidence, nan=1.0), 0.0, 1.0)

    # --- ORM loss (scalar)
    orm_loss = orm_bce_loss(orm_logits, orm_targets, pos_weight=pos_weight, label_smoothing=label_smoothing, reduction="mean")

    # --- PRM per-example losses (we compute per-example when chain_conf provided)
    # stepwise per-example
    prm_step_per_example = stepwise_prm_loss(
        step_preds=step_probs,
        step_targets=step_targets,
        step_mask=step_mask,
        step_confidence=step_confidence,
        delta=1.0,
        reduction="none",
    )  # (B,)

    prm_mean_per_example = prm_solution_mean_loss(step_probs, step_targets, step_mask, reduction="none")  # (B,)
    prm_prod_per_example = prm_solution_product_loss(
        step_probs, step_targets, step_mask, safe_mode=safe_prod, use_log_mse=use_log_mse_for_prod, reduction="none"
    )  # (B,)

    # --- Consistency per-example (compare ORM_prob vs PRM_agg)
    pred_prm_agg = (aggregate_prm_product_safe(step_probs, step_mask) if safe_prod else aggregate_prm_product(step_probs, step_mask))
    orm_prob = torch.sigmoid(orm_logits.view(-1))
    consistency_per_example = F.mse_loss(orm_prob, pred_prm_agg.detach(), reduction="none")  # (B,)

    # --- apply chain_confidence weighting if provided
    if chain_confidence is not None:
        chain_confidence = chain_confidence.view(-1).float()
        # ensure same batch size
        assert chain_confidence.shape[0] == prm_step_per_example.shape[0], "chain_confidence batch size mismatch"
        # weight per-example losses by chain_confidence for PRM-related losses
        prm_step_loss = (prm_step_per_example * chain_confidence).sum() / (chain_confidence.sum().clamp_min(1.0))
        prm_mean_loss = (prm_mean_per_example * chain_confidence).sum() / (chain_confidence.sum().clamp_min(1.0))
        prm_prod_loss = (prm_prod_per_example * chain_confidence).sum() / (chain_confidence.sum().clamp_min(1.0))
        consistency_loss = (consistency_per_example * chain_confidence).sum() / (chain_confidence.sum().clamp_min(1.0))
    else:
        # fallback to batch mean (backwards compatible)
        prm_step_loss = prm_step_per_example.mean()
        prm_mean_loss = prm_mean_per_example.mean()
        prm_prod_loss = prm_prod_per_example.mean()
        consistency_loss = consistency_per_example.mean()

    # --- Combine (ORM not scaled by chain_confidence; PRM terms scaled by lambdas)
    total = (
        lambda_orm * orm_loss
        + lambda_prm_step * prm_step_loss
        + lambda_prm_mean * prm_mean_loss
        + lambda_prm_prod * prm_prod_loss
        + lambda_consistency * consistency_loss
    )

    # --- Optional loss clipping (defensive)
    if loss_clip_max is not None:
        total = total.clamp(min=0.0, max=float(loss_clip_max))

    # diagnostics (detached)
    components = {
        "orm_loss": orm_loss.detach(),
        "prm_step_loss": prm_step_loss.detach(),
        "prm_mean_loss": prm_mean_loss.detach(),
        "prm_prod_loss": prm_prod_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
        "total_loss": total.detach(),
        "prm_mean_pred_mean": aggregate_prm_mean(step_probs, step_mask).detach().mean(),
        "prm_prod_pred_mean": pred_prm_agg.detach().mean(),
        "orm_prob_mean": orm_prob.detach().mean(),
    }

    if not torch.isfinite(total):
        print("[Warning] Non-finite total loss encountered. Check input batch or step_targets range.")

    return total, components


# -------------------------
# Target Normalization Helper
# -------------------------
def normalize_step_targets(step_targets: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Normalize PRM step targets to [0,1] if they are in 0..10.
    Robust to NaNs: finds max over non-nan elements.
    """
    if not torch.is_tensor(step_targets):
        raise TypeError("step_targets must be a torch.Tensor")

    # avoid nanmax (compatibility): use masked max over finite entries
    finite_mask = torch.isfinite(step_targets)
    if finite_mask.any():
        max_val = float(step_targets[finite_mask].max().detach().cpu().item())
    else:
        max_val = 0.0

    if max_val > 1.5:
        return step_targets.div_(10.0) if inplace else step_targets / 10.0
    return step_targets


# -------------------------
# Exports
# -------------------------
__all__ = [
    "stepwise_prm_loss",
    "prm_solution_mean_loss",
    "prm_solution_product_loss",
    "aggregate_prm_mean",
    "aggregate_prm_product",
    "aggregate_prm_product_safe",
    "orm_bce_loss",
    "combined_prm_orm_loss",
    "normalize_step_targets",
]
