"""
Extended diagnostic smoke test for Dual-Head Reward Model loss functions.

Adds:
 - gradient magnitude balance diagnostics
 - per-loss gradient histograms
 - PRM aggregation sanity visualization (requires matplotlib)
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.losses.prm_orm_losses import combined_prm_orm_loss, normalize_step_targets, aggregate_prm_product_safe, aggregate_prm_mean
from src.utils.lambda_auto_calibrator import auto_calibrate_lambdas

torch.manual_seed(42)

# ==========================================================
# Synthetic batch setup
# ==========================================================
B, S = 6, 10  # slightly larger batch for histograms

step_preds_logits = torch.randn(B, S, requires_grad=True)
raw_targets = torch.randint(low=0, high=10, size=(B, S)).float()
step_targets = normalize_step_targets(raw_targets)
step_mask = torch.ones(B, S)
step_mask[0, 8:] = 0
step_mask[1, 9:] = 0
step_conf = torch.rand(B, S) * 0.5 + 0.5

orm_logits = torch.randn(B, requires_grad=True)
orm_targets = torch.randint(0, 2, (B,))

# ==========================================================
# Run full hybrid loss
# ==========================================================
loss, comps = combined_prm_orm_loss(
    orm_logits=orm_logits,
    orm_targets=orm_targets,
    step_preds=step_preds_logits,
    step_targets=step_targets,
    step_mask=step_mask,
    step_confidence=step_conf,
    lambda_orm=1.0,
    lambda_prm_step=0.5,
    lambda_prm_mean=0.4,
    lambda_prm_prod=0.5,
    lambda_consistency=0.1,
    step_preds_are_logits=True,
    safe_prod=True,
    use_log_mse_for_prod=True,
)

# ==========================================================
# Gradient diagnostics
# ==========================================================
loss.backward(retain_graph=True)

print("\n===== LOSS COMPONENTS =====")
for k, v in comps.items():
    if isinstance(v, torch.Tensor):
        v = v.item() if v.ndim == 0 else v.mean().item()
    print(f"{k:25s}: {v:.6f}")

print("\nTotal loss:", float(loss))
print("Has NaN?", torch.isnan(loss).any().item())
print("Has Inf?", torch.isinf(loss).any().item())

# Grad magnitudes
grad_stats = {
    "step_preds_grad_mean": step_preds_logits.grad.abs().mean().item(),
    "step_preds_grad_std": step_preds_logits.grad.abs().std().item(),
    "orm_grad_mean": orm_logits.grad.abs().mean().item(),
    "orm_grad_std": orm_logits.grad.abs().std().item(),
}

print("\n===== GRADIENT MAGNITUDES =====")
for k, v in grad_stats.items():
    print(f"{k:25s}: {v:.6e}")

assert torch.isfinite(step_preds_logits.grad).all(), "NaN/Inf in step_preds grad!"
assert torch.isfinite(orm_logits.grad).all(), "NaN/Inf in orm grad!"
print("✅ Gradients finite and healthy.\n")

# ==========================================================
# λ Auto-Calibration Integration
# ==========================================================

print("\n===== λ-AUTO CALIBRATION TEST =====")

current_lambdas = {
    "lambda_orm": 1.0,
    "lambda_prm_step": 0.4,
    "lambda_prm_mean": 0.4,
    "lambda_prm_prod": 0.4,
    "lambda_consistency": 0.1,
}

# Prepare batch dict compatible with combined_prm_orm_loss
batch_dict = dict(
    orm_logits=orm_logits.detach().clone().requires_grad_(True),
    orm_targets=orm_targets,
    step_preds=step_preds_logits.detach().clone().requires_grad_(True),
    step_targets=step_targets,
    step_mask=step_mask,
)

new_lambdas = auto_calibrate_lambdas(
    model=None,  # since we’re testing standalone losses (no full model here)
    batch=batch_dict,
    current_lambdas=current_lambdas,
    loss_fn=combined_prm_orm_loss,
    target_grad=0.02,   # desired average gradient magnitude
    verbose=True,
)

print("\n=== Suggested λ After Calibration ===")
for k, v in new_lambdas.items():
    print(f"{k:20s}: {v:.3f}")

# Visual diagnostic of grad vs λ ratios (optional)
import matplotlib.pyplot as plt

before = list(current_lambdas.values())
after = [new_lambdas[k] for k in current_lambdas.keys()]
labels = list(current_lambdas.keys())

plt.figure(figsize=(8, 4))
x = range(len(labels))
plt.bar(x, before, width=0.35, label="Before", alpha=0.6)
plt.bar([i + 0.35 for i in x], after, width=0.35, label="After", alpha=0.6)
plt.xticks([i + 0.2 for i in x], labels, rotation=30, ha="right")
plt.ylabel("λ Value")
plt.title("λ Calibration — Before vs After")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("tests/lambda_calibration_comparison.png", dpi=160)
print("📈 Saved λ calibration comparison plot → tests/lambda_calibration_comparison.png")

# ==========================================================
# Per-loss contribution visualization
# ==========================================================
with torch.no_grad():
    step_grad = step_preds_logits.grad.detach().cpu().flatten().abs()
    orm_grad = orm_logits.grad.detach().cpu().flatten().abs()

    plt.figure(figsize=(10, 4))
    plt.hist(step_grad, bins=40, alpha=0.7, label="PRM (step) grad")
    plt.hist(orm_grad, bins=40, alpha=0.7, label="ORM grad")
    plt.title("Gradient Magnitude Distribution per Head")
    plt.xlabel("|grad|")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("tests/grad_balance_hist.png", dpi=160)
    print("📊 Saved gradient histogram → tests/grad_balance_hist.png")

# ==========================================================
# PRM Aggregation sanity check
# ==========================================================
with torch.no_grad():
    step_probs = torch.sigmoid(step_preds_logits)
    prm_mean = aggregate_prm_mean(step_probs, step_mask)
    prm_prod = aggregate_prm_product_safe(step_probs, step_mask)
    orm_prob = torch.sigmoid(orm_logits)

    plt.figure(figsize=(6, 5))
    plt.scatter(prm_mean.cpu(), orm_prob.cpu(), c="blue", alpha=0.6, label="mean vs ORM")
    plt.scatter(prm_prod.cpu(), orm_prob.cpu(), c="red", alpha=0.5, label="prod vs ORM")
    plt.xlabel("PRM aggregated (mean / product)")
    plt.ylabel("ORM probability")
    plt.title("Aggregation sanity check")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("tests/prm_aggregation_sanity.png", dpi=160)
    print("📈 Saved PRM aggregation sanity plot → tests/prm_aggregation_sanity.png")

# ==========================================================
# Summary
# ==========================================================
print("\n===== SUMMARY =====")
print(f"Avg PRM mean agg:  {prm_mean.mean():.4f}")
print(f"Avg PRM prod agg:  {prm_prod.mean():.4f}")
print(f"Avg ORM prob:      {orm_prob.mean():.4f}")
print("\n✅ Extended smoke test passed — gradients balanced, plots saved.\n")
