# %%writefile src/training/evaluation.py
# src/training/evaluation.py
"""
Evaluation utilities for Dual-Head Reward Model (ORM + PRM)

Provides:
 - evaluate_rm(model, dataloader, lambda_dict, device, verbose=False)
    Runs a full validation pass computing:
      * total loss (using combined_prm_orm_loss)
      * ORM metrics: accuracy, balanced accuracy, Brier score, (optional ROC AUC)
      * PRM metrics: per-step MSE, mean absolute error, Pearson correlation of
        mean-aggregated PRM, Pearson correlation of product-aggregated PRM
      * Consistency MSE between ORM probability and PRM aggregated product
      * Diagnostic scalars (avg predicted means/prods, counts)
    Returns a dictionary of metrics suitable for logging and checkpointing.

Notes:
 - The function accepts dataloaders that yield batches as dict-like objects
   consistent with your dataloader_rm.py (keys: input_ids, attention_mask,
   step_targets, step_mask, orm_label, step_confidence (optional),
   chain_confidence (optional)).
 - Uses torch.no_grad() and supports AMP autocast for speed.
"""
import sys
from typing import Dict, Any, Optional
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# try optional metrics
try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

from src.losses.prm_orm_losses import (
    combined_prm_orm_loss,
    aggregate_prm_mean,
    aggregate_prm_product_safe,
    aggregate_prm_product,
)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Return Pearson correlation or None if constant / invalid."""
    if x.size == 0 or y.size == 0:
        return None
    if np.all(np.isclose(x, x.flat[0])) or np.all(np.isclose(y, y.flat[0])):
        return None
    try:
        r = np.corrcoef(x, y)[0, 1]
        if not np.isfinite(r):
            return None
        return float(r)
    except Exception:
        return None


def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier score for binary probabilistic predictions."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def evaluate_rm(
    model: torch.nn.Module,
    dataloader,
    lambda_dict: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run validation over dataloader and compute metrics.

    Args:
        model: DualHeadRM (eval mode expected to be set by caller or will be set here).
        dataloader: DataLoader that yields batches compatible with training.
        lambda_dict: dictionary of loss weights and flags accepted by combined_prm_orm_loss.
        device: torch.device
        verbose: if True prints progress.

    Returns:
        metrics dict with keys:
          - val_total_loss
          - val_orm_acc, val_orm_balanced_acc, val_orm_brier, val_orm_auc (optional)
          - val_prm_step_mse, val_prm_step_mae
          - val_prm_mean_corr, val_prm_prod_corr
          - val_consistency_mse
          - counts / diagnostics
    """
    was_training = model.training
    model.eval()

    losses = []
    orm_probs = []
    orm_labels = []
    prm_step_mse_vals = []
    prm_step_mae_vals = []
    prm_mean_preds = []
    prm_mean_targets = []
    prm_prod_preds = []
    prm_prod_targets = []
    consistency_losses = []

    total_examples = 0

    device = device or next(model.parameters()).device

    # inference loop
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Eval", leave=True, dynamic_ncols=True, position=0, file=sys.stdout) if verbose else dataloader
        for batch in iterator:
            # move all tensors in batch to model device and cast numeric tensors to model dtype
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            # move all tensors in batch to device (best-effort)
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                    # categorical / index tensors (input_ids, etc.) should be long
                if v.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    batch[k] = v.to(model_device, non_blocking=True).long()
                else:
                    # float tensors: cast to model dtype (fp16 likely) to match heads
                    batch[k] = v.to(model_device, non_blocking=True).to(dtype=model_dtype)

            # model forward (AMP off for deterministic eval)
            print("Batch devices/dtypes:", {k: (v.device, v.dtype) for k, v in batch.items() if torch.is_tensor(v)})
            print("Model first param:", next(model.parameters()).device, next(model.parameters()).dtype)

            try:
                print("[DEBUG] Dtypes:",
                      "base:", next(model.base.parameters()).dtype,
                      "orm:", next(model.orm_proj.parameters()).dtype,
                      "prm:", next(model.prm_mlp.parameters()).dtype)
            except Exception as e:
                print("[DEBUG] Could not print model dtypes:", e)

            out = model(batch["input_ids"], attention_mask=batch.get("attention_mask", None), return_dict=True)

            model_device = next(model.parameters()).device
            for k, v in batch.items():
                if torch.is_tensor(v) and v.device != model_device:
                    batch[k] = v.to(model_device, non_blocking=True)

            orm_logits = out["orm_logits"]  # (B,)
            prm_logits = out["prm_logits"]  # (B, S)

            # Use local target device = model's parameter device (guaranteed device)
            target_device = next(model.parameters()).device
            target_dtype = next(model.parameters()).dtype  # typically torch.float16 for your setup

            # step_preds: model output (logits) -> ensure device + dtype
            step_preds = prm_logits.to(device=target_device, non_blocking=True).to(dtype=target_dtype)

            # step_targets / masks / confidences: move & cast explicitly from batch
            step_targets = batch["step_targets"].to(device=target_device, non_blocking=True).to(dtype=target_dtype)
            step_mask = batch["step_mask"].to(device=target_device, non_blocking=True).to(dtype=target_dtype)

            # optional confidences used by loss: move & cast if present
            step_confidence = batch.get("step_confidence", None)
            if step_confidence is not None:
                step_confidence = step_confidence.to(device=target_device, non_blocking=True).to(dtype=target_dtype)

            chain_confidence = batch.get("chain_confidence", None)
            if chain_confidence is not None:
                chain_confidence = chain_confidence.to(device=target_device, non_blocking=True).to(dtype=target_dtype)

            # orm targets: ensure device + dtype match orm_logits
            orm_logits = orm_logits.to(device=target_device, non_blocking=True).to(dtype=target_dtype)
            orm_targets = batch["orm_label"].to(device=target_device, non_blocking=True).to(dtype=orm_logits.dtype)

            # debug print (temporary)
            print("orm_logits:", orm_logits.device, orm_logits.dtype)
            print("orm_targets:", orm_targets.device, orm_targets.dtype)
            print("step_preds:", step_preds.device, step_preds.dtype)
            print("step_targets:", step_targets.device, step_targets.dtype)
            if step_confidence is not None:
                print("step_confidence:", step_confidence.device, step_confidence.dtype)
            if chain_confidence is not None:
                print("chain_confidence:", chain_confidence.device, chain_confidence.dtype)

            # compute combined loss (we reuse training loss wrapper to ensure identical metric calc)
            loss_tensor, comps = combined_prm_orm_loss(
                orm_logits=orm_logits,
                orm_targets=orm_targets,
                step_preds=step_preds,
                step_targets=step_targets,
                step_mask=step_mask,
                step_confidence=batch.get("step_confidence", None),
                chain_confidence=batch.get("chain_confidence", None),
                **lambda_dict,
            )

            losses.append(float(loss_tensor.detach().cpu().item()))

            torch.cuda.empty_cache()

            # --- gather per-example predictions & targets for metrics
            orm_prob = torch.sigmoid(orm_logits.view(-1)).detach().cpu().numpy()
            orm_prob = np.clip(orm_prob, 0.0, 1.0)
            orm_label_np = batch["orm_label"].detach().cpu().numpy().astype(int)
            orm_probs.extend(orm_prob.tolist())
            orm_labels.extend(orm_label_np.tolist())

            # per-step metrics (use preds after sigmoid)
            step_probs = torch.sigmoid(step_preds).detach().cpu().numpy()  # (B, S)
            step_tg = step_targets.detach().cpu().numpy()
            step_m = step_mask.detach().cpu().numpy()

            # masked MSE/MAE per example
            # compute per-example sums ignoring padded values
            denom = (step_m).sum(axis=1)
            denom_safe = np.where(denom > 0, denom, 1.0)
            diff = (step_probs - step_tg) * step_m
            mse_per_example = np.sum(diff ** 2, axis=1) / denom_safe
            mae_per_example = np.sum(np.abs(diff), axis=1) / denom_safe
            prm_step_mse_vals.extend(mse_per_example.tolist())
            prm_step_mae_vals.extend(mae_per_example.tolist())

            # aggregated PRM (mean and product) predictions & targets
            # mean
            mean_pred = np.sum(step_probs * step_m, axis=1) / denom_safe
            mean_tgt = np.sum(step_tg * step_m, axis=1) / denom_safe
            prm_mean_preds.extend(mean_pred.tolist())
            prm_mean_targets.extend(mean_tgt.tolist())

            # product: use same safe product used in losses (mean-normalized log-sum)
            # implement numerically stable product in numpy
            eps = 1e-12
            masked_preds = np.clip(step_probs, eps, 1.0)
            log_preds = np.log(masked_preds) * step_m
            n_valid = denom_safe
            log_prod_mean = np.sum(log_preds, axis=1) / n_valid
            prod_pred = np.exp(log_prod_mean)
            # target product
            masked_tg = np.clip(step_tg, eps, 1.0)
            log_tg = np.log(masked_tg) * step_m
            log_tgt_mean = np.sum(log_tg, axis=1) / n_valid
            prod_tgt = np.exp(log_tgt_mean)

            prm_prod_preds.extend(prod_pred.tolist())
            prm_prod_targets.extend(prod_tgt.tolist())

            # consistency per-example (MSE between ORM prob and PRM agg)
            consistency = (orm_prob - prod_pred) ** 2
            consistency_losses.extend(consistency.tolist())

            total_examples += step_probs.shape[0]

    # compute aggregated metrics
    metrics: Dict[str, Any] = {}
    metrics["val_total_loss"] = float(np.mean(losses)) if len(losses) > 0 else float("nan")
    metrics["val_num_examples"] = int(total_examples)

    # ORM metrics
    orm_probs_arr = np.asarray(orm_probs, dtype=float)
    orm_labels_arr = np.asarray(orm_labels, dtype=int)
    # accuracy at 0.5
    if orm_labels_arr.size > 0:
        preds_bin = (orm_probs_arr >= 0.5).astype(int)
        acc = float((preds_bin == orm_labels_arr).mean())
        metrics["val_orm_acc"] = acc
        # balanced accuracy
        try:
            # per-class recall
            pos_idx = orm_labels_arr == 1
            neg_idx = orm_labels_arr == 0
            pos_recall = float(np.mean(preds_bin[pos_idx] == 1)) if pos_idx.sum() > 0 else 0.0
            neg_recall = float(np.mean(preds_bin[neg_idx] == 0)) if neg_idx.sum() > 0 else 0.0
            metrics["val_orm_balanced_acc"] = float(0.5 * (pos_recall + neg_recall))
        except Exception:
            metrics["val_orm_balanced_acc"] = None
        # Brier score
        metrics["val_orm_brier"] = float(_brier_score(orm_probs_arr, orm_labels_arr))
        # ROC AUC if sklearn present and both classes exist
        if _SKLEARN_AVAILABLE and len(np.unique(orm_labels_arr)) == 2:
            try:
                metrics["val_orm_auc"] = float(roc_auc_score(orm_labels_arr, orm_probs_arr))
            except Exception:
                metrics["val_orm_auc"] = None
        else:
            metrics["val_orm_auc"] = None
    else:
        metrics["val_orm_acc"] = None
        metrics["val_orm_balanced_acc"] = None
        metrics["val_orm_brier"] = None
        metrics["val_orm_auc"] = None

    # PRM per-step metrics
    prm_step_mse_arr = np.asarray(prm_step_mse_vals, dtype=float)
    prm_step_mae_arr = np.asarray(prm_step_mae_vals, dtype=float)
    metrics["val_prm_step_mse"] = float(np.mean(prm_step_mse_arr)) if prm_step_mse_arr.size > 0 else None
    metrics["val_prm_step_mae"] = float(np.mean(prm_step_mae_arr)) if prm_step_mae_arr.size > 0 else None

    # PRM aggregated correlations
    mean_pred_arr = np.asarray(prm_mean_preds, dtype=float)
    mean_tgt_arr = np.asarray(prm_mean_targets, dtype=float)
    prod_pred_arr = np.asarray(prm_prod_preds, dtype=float)
    prod_tgt_arr = np.asarray(prm_prod_targets, dtype=float)

    metrics["val_prm_mean_corr"] = _safe_pearson(mean_pred_arr, mean_tgt_arr)
    metrics["val_prm_prod_corr"] = _safe_pearson(prod_pred_arr, prod_tgt_arr)

    # consistency
    consistency_arr = np.asarray(consistency_losses, dtype=float)
    metrics["val_consistency_mse"] = float(np.mean(consistency_arr)) if consistency_arr.size > 0 else None

    # diagnostic scalars
    metrics["val_prm_mean_pred_mean"] = float(np.nanmean(mean_pred_arr)) if mean_pred_arr.size > 0 else None
    metrics["val_prm_prod_pred_mean"] = float(np.nanmean(prod_pred_arr)) if prod_pred_arr.size > 0 else None
    metrics["val_orm_prob_mean"] = float(np.nanmean(orm_probs_arr)) if orm_probs_arr.size > 0 else None

    # restore training mode if necessary
    if was_training:
        model.train()

    return metrics
