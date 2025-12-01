"""
Automatic λ-weight calibration utility for multi-loss (Dual-Head RM).

Analyzes gradient magnitudes across loss components and suggests
λ-weight scaling factors for balanced training stability.

Usage:
    from utils.lambda_auto_calibrator import auto_calibrate_lambdas
    new_lambdas = auto_calibrate_lambdas(model, batch, current_lambdas, loss_fn)
"""

import torch
import numpy as np
from typing import Dict, Callable

def _get_grad_stats(tensor: torch.Tensor) -> Dict[str, float]:
    if tensor.grad is None:
        return {"mean": 0.0, "std": 0.0}
    g = tensor.grad.detach().abs()
    return {
        "mean": g.mean().item(),
        "std": g.std().item(),
        "max": g.max().item(),
        "min": g.min().item(),
    }

def auto_calibrate_lambdas(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    current_lambdas: Dict[str, float],
    loss_fn: Callable[..., torch.Tensor],
    target_grad: float = 0.01,
    safe_min: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run a single forward/backward pass to compute gradient magnitudes per loss head,
    then rescale λ weights so all heads produce roughly equal gradient norms.

    Args:
        model: DualHeadRewardModel (with requires_grad parameters)
        batch: dict containing input tensors (orm_logits, step_preds, etc.)
        current_lambdas: dict of {λ_orm, λ_prm_step, λ_prm_mean, λ_prm_prod, λ_consistency}
        loss_fn: combined_prm_orm_loss or equivalent callable returning (loss, components)
        target_grad: target average gradient magnitude per head
        safe_min: prevents division by zero
        verbose: print report

    Returns:
        new_lambdas: dict with recommended λ scalings
    """
    if model:
        model.zero_grad(set_to_none=True)

    # Forward pass and compute loss components
    loss, comps = loss_fn(**batch, **current_lambdas)
    loss.backward(retain_graph=True)

    stats = {}
    with torch.no_grad():
        if hasattr(model, "orm_head"):
            stats["ORM"] = _get_grad_stats(model.orm_head.weight)
        if hasattr(model, "prm_head"):
            stats["PRM"] = _get_grad_stats(model.prm_head.weight)

        grad_means = {k: v["mean"] for k, v in stats.items() if v["mean"] > 0}
        avg_grad = np.mean(list(grad_means.values())) if grad_means else 0.0

        new_lambdas = current_lambdas.copy()
        for k in grad_means:
            ratio = target_grad / max(grad_means[k], safe_min)
            new_lambdas_key = None
            if k == "ORM": new_lambdas_key = "lambda_orm"
            if k == "PRM": new_lambdas_key = "lambda_prm_step"  # primary PRM term

            if new_lambdas_key:
                new_lambdas[new_lambdas_key] = float(
                    np.clip(current_lambdas[new_lambdas_key] * ratio, 0.1, 3.0)
                )

        if verbose:
            print("\n=== λ Auto-Calibrator Report ===")
            for k, v in stats.items():
                print(f"{k} Grad: mean={v['mean']:.3e}, std={v['std']:.3e}, max={v['max']:.3e}")
            print(f"Avg Grad: {avg_grad:.3e}")
            print("\n--- Suggested λ Updates ---")
            for k, v in new_lambdas.items():
                diff = v / current_lambdas[k] if current_lambdas[k] != 0 else 0
                print(f"{k:20s}: {current_lambdas[k]:.3f} → {v:.3f}  (x{diff:.2f})")
    if model:
        model.zero_grad(set_to_none=True)
    return new_lambdas
