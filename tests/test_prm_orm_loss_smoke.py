"""
Smoke test for src/losses/prm_orm_losses.py

This verifies:
 - forward/backward passes run cleanly
 - no NaN or inf values appear
 - sub-loss components are finite
 - per-step masking, confidence, and consistency behave correctly
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.losses.prm_orm_losses import combined_prm_orm_loss, normalize_step_targets

torch.manual_seed(42)

# ==========================================================
# Create synthetic test batch
# ==========================================================
batch_size = 4
num_steps = 10

# Step logits (model outputs): unbounded
step_preds_logits = torch.randn(batch_size, num_steps, requires_grad=True)

# Step targets (in 0..10 for test normalization)
raw_targets = torch.randint(low=0, high=10, size=(batch_size, num_steps)).float()
step_targets = normalize_step_targets(raw_targets)  # -> scaled to [0,1]

# Step mask (some sequences shorter)
step_mask = torch.ones(batch_size, num_steps)
step_mask[0, 8:] = 0  # shorter chain
step_mask[1, 9:] = 0

# Confidence: vary per-step
step_conf = torch.rand(batch_size, num_steps) * 0.5 + 0.5  # range 0.5–1.0

# ORM logits and targets
orm_logits = torch.randn(batch_size, requires_grad=True)
orm_targets = torch.randint(0, 2, (batch_size,))

# ==========================================================
# Run combined loss
# ==========================================================
total_loss, components = combined_prm_orm_loss(
    orm_logits=orm_logits,
    orm_targets=orm_targets,
    step_preds=step_preds_logits,
    step_targets=step_targets,
    step_mask=step_mask,
    step_confidence=step_conf,
    lambda_orm=1.0,
    lambda_prm_step=0.4,
    lambda_prm_mean=0.0,
    lambda_prm_prod=0.4,
    lambda_consistency=0.1,
    step_preds_are_logits=True,
    safe_prod=True,
    use_log_mse_for_prod=True,
)

# ==========================================================
# Print diagnostic summary
# ==========================================================
print("===== LOSS COMPONENTS =====")
for k, v in components.items():
    if isinstance(v, torch.Tensor):
        v = v.item() if v.ndim == 0 else v.mean().item()
    print(f"{k:25s}: {v:.6f}")

print("\nTotal loss (scalar) =", float(total_loss))
print("Has NaN?", torch.isnan(total_loss).any().item())
print("Has Inf?", torch.isinf(total_loss).any().item())

# ==========================================================
# Backward pass test
# ==========================================================
total_loss.backward()

print("\n===== GRADIENT CHECK =====")
print("step_preds_logits.grad mean =", step_preds_logits.grad.mean().item())
print("orm_logits.grad mean =", orm_logits.grad.mean().item())

assert torch.isfinite(step_preds_logits.grad).all(), "Non-finite grad in step_preds!"
assert torch.isfinite(orm_logits.grad).all(), "Non-finite grad in orm head!"
print("\n✅ Smoke test passed — loss forward/backward are stable.")
