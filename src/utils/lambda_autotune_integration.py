"""
Automatic λ-weight calibration integration for Dual-Head Reward Model training.

Performs a short dry-run (1–2 mini-batches) to balance gradient magnitudes
across ORM and PRM heads, then updates the config file with the tuned λs.

Usage:
    from utils.lambda_autotune_integration import calibrate_and_update_config
    calibrate_and_update_config(
        model, dataloader, config_path="configs/train_config.yaml",
        loss_fn=combined_prm_orm_loss, target_grad=0.02
    )
"""

import json
import yaml
import torch
from typing import Dict, Any
from src.utils.lambda_auto_calibrator import auto_calibrate_lambdas
from src.utils.lambda_calibration_report import generate_lambda_calibration_report
from datetime import datetime


def _load_config(config_path: str) -> Dict[str, Any]:
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")


def _save_config(config_path: str, cfg: Dict[str, Any]) -> None:
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    elif config_path.endswith(".json"):
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"[INFO] Updated config written to: {config_path}")


@torch.no_grad()
def _get_sample_batch(dataloader):
    """Retrieve one mini-batch safely from a dataloader (no shuffling)."""
    for batch in dataloader:
        return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}


def calibrate_and_update_config(
    model: torch.nn.Module,
    dataloader,
    config_path: str,
    loss_fn,
    target_grad: float = 0.02,
    dry_run_steps: int = 1,
) -> Dict[str, float]:
    """
    Runs λ auto-calibration and updates config file with tuned λs.

    Args:
        model: DualHeadRewardModel
        dataloader: PRM/ORM combined DataLoader
        config_path: path to YAML/JSON training config
        loss_fn: combined_prm_orm_loss function
        target_grad: target gradient magnitude per head
        dry_run_steps: how many batches to use for calibration

    Returns:
        new_lambdas: dict of updated λ values
    """
    cfg = _load_config(config_path)
    lambdas = cfg.get("lambda_weights", {
        "lambda_orm": 1.0,
        "lambda_prm_step": 0.4,
        "lambda_prm_mean": 0.4,
        "lambda_prm_prod": 0.4,
        "lambda_consistency": 0.1,
    })

    print("\n=== Running λ Auto-Calibration ===")
    model.train()
    model.zero_grad(set_to_none=True)

    # Collect gradient stats over N dry-run steps
    grad_suggestions = []
    for _ in range(dry_run_steps):
        batch = _get_sample_batch(dataloader)
        new_lambdas = auto_calibrate_lambdas(
            model=model,
            batch=batch,
            current_lambdas=lambdas,
            loss_fn=loss_fn,
            target_grad=target_grad,
            verbose=True,
        )
        grad_suggestions.append(new_lambdas)

    # Average tuned λs
    new_lambdas = {
        k: sum(d[k] for d in grad_suggestions) / len(grad_suggestions)
        for k in lambdas
    }

    cfg["lambda_weights"] = {k: float(v) for k, v in new_lambdas.items()}
    _save_config(config_path, cfg)

    print("\n✅ λ Auto-Calibration complete.")
    print("Updated λs:")
    for k, v in new_lambdas.items():
        print(f"  {k}: {v:.3f}")

    # --- auto-report section ---
    try:
        generate_lambda_calibration_report(
            pre_lambdas=lambdas,
            post_lambdas=new_lambdas,
            pre_grad_stats=None,  # optionally feed collected grad stats later
            output_path="reports/calibration_report.png",
            meta_info={
                "model": model.__class__.__name__,
                "dataset": getattr(dataloader.dataset, 'name', 'unknown'),
                "target_grad": target_grad,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    except Exception as e:
        print(f"[WARN] Failed to generate λ-calibration report: {e}")

    return new_lambdas



