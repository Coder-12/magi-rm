# src/train_reward_models.py
"""
Train dual-head Reward Model (ORM + PRM) with Accelerate.

Usage:
    accelerate launch src/train_reward_models.py --config configs/train_config.yaml

Features:
 - Loads ORM & PRM datasets produced by scripts/prepare_rm_datasets.py
 - Alternates batches from ORM and PRM or interleaves them
 - Logs TensorBoard + JSON via setup_environment
 - Checkpointing and eval for both heads
"""

import time
import yaml
import argparse
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, AdamW

from src.utils.data_utils import ORMJsonlDataset, PRMJsonlDataset, RMDataCollator
from src.model import RewardModel
from src.setup_environment import setup_run, log_metrics

# ---------- helpers ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config (see configs/train_config.yaml)")
    return p.parse_args()


def make_dataloader(dataset, collate_fn, batch_size, shuffle=True, num_workers=2):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)


# ---------- main ----------
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # Setup run env & logging
    run_state = setup_run(run_name=cfg["run_name"], cfg=cfg, base_dir="runs", seed=cfg.get("seed", 12345))
    tb = run_state.tb_writer
    json_fh = run_state.json_log
    device = torch.device(run_state.device)

    # accelerator
    accelerator = Accelerator(fp16=(cfg.get("device", {}).get("prefer_cuda", True) and torch.cuda.is_available()))
    print("Accelerator initialized:", accelerator.state)

    # tokenizer and datasets
    tokenizer_name = cfg.get("base_model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=cfg.get("misc", {}).get("use_tokenizer_fast", True))
    # ensure eos token exists
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": ""})

    orm_ds = ORMJsonlDataset(cfg["train"]["orm_train_jsonl"])
    prm_ds = PRMJsonlDataset(cfg["train"]["prm_train_jsonl"])

    collator = RMDataCollator(tokenizer=tokenizer, max_length=cfg["misc"]["max_seq_length"], include_question=True)

    batch_size = int(cfg["train"]["per_device_batch_size"])
    orm_loader = make_dataloader(orm_ds, collate_fn=collator.collate_orm, batch_size=batch_size, shuffle=True, num_workers=2)
    prm_loader = make_dataloader(prm_ds, collate_fn=collator.collate_prm, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model
    model = RewardModel(cfg["base_model"], device=None)  # let Accelerator move it
    # If we modified tokenizer (added tokens), resize embedding
    try:
        model.backbone.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # Losses
    orm_loss_fn = nn.BCEWithLogitsLoss()
    prm_loss_fn = nn.MSELoss()

    # Optimizer & scheduler
    lr = cfg["optim"]["lr"]
    weight_decay = cfg["optim"].get("weight_decay", 0.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_train_steps = cfg["train"].get("max_train_steps", None)
    if not total_train_steps:
        # estimate from epochs
        total_train_steps = int(cfg["train"].get("epochs", 1) * (len(orm_loader) + len(prm_loader)))  # rough
    warmup_steps = cfg["optim"].get("warmup_steps", 500)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    # Prepare with accelerate
    model, optimizer, orm_loader, prm_loader, scheduler = accelerator.prepare(model, optimizer, orm_loader, prm_loader, scheduler)

    # training params
    lambda_orm = cfg["loss_weights"]["lambda_orm"]
    lambda_prm = cfg["loss_weights"]["lambda_prm"]
    epochs = cfg["train"].get("epochs", 3)
    save_every = cfg["train"].get("save_every_steps", 2000)
    val_every = cfg["logging"].get("val_every_steps", 500)
    tb_every = cfg["logging"].get("tb_every_steps", 50)

    global_step = 0
    best_val_metric = -1e9

    # Convert loaders to iterators for alternating batches
    orm_iter = iter(orm_loader)
    prm_iter = iter(prm_loader)
    steps_per_epoch = max(len(orm_loader), len(prm_loader))

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        start = time.time()
        # iterate steps_per_epoch times; at each step consume one batch from each loader when possible
        for step in range(steps_per_epoch):
            losses = []
            optimizer.zero_grad()

            # ORM batch (if available)
            try:
                orm_batch = next(orm_iter)
            except StopIteration:
                orm_iter = iter(orm_loader)
                orm_batch = next(orm_iter)

            orm_inputs = {k: v.to(accelerator.device) for k, v in orm_batch.items() if k in ("input_ids", "attention_mask")}
            orm_labels = orm_batch["labels"].to(accelerator.device)

            out = model(input_ids=orm_inputs["input_ids"], attention_mask=orm_inputs["attention_mask"], tokenizer=tokenizer)
            orm_logits = out["orm_logits"]
            loss_orm = orm_loss_fn(orm_logits, orm_labels)
            losses.append(lambda_orm * loss_orm)

            # PRM batch
            try:
                prm_batch = next(prm_iter)
            except StopIteration:
                prm_iter = iter(prm_loader)
                prm_batch = next(prm_iter)

            prm_inputs = {k: v.to(accelerator.device) for k, v in prm_batch.items() if k in ("input_ids", "attention_mask")}
            prm_labels = prm_batch["labels"].to(accelerator.device)

            out2 = model(input_ids=prm_inputs["input_ids"], attention_mask=prm_inputs["attention_mask"], tokenizer=tokenizer)
            prm_pred = out2["prm_pred"]
            # clamp predictions to [0,1]? MSE tolerates outside range; keep raw for gradient stability
            loss_prm = prm_loss_fn(prm_pred, prm_labels)
            losses.append(lambda_prm * loss_prm)

            # total losses
            total_loss = sum(losses)
            accelerator.backward(total_loss)

            # optimizer step
            optimizer.step()
            scheduler.step()
            epoch_losses.append(accelerator.gather_for_metrics(total_loss.detach()).mean().item() if hasattr(accelerator, "gather_for_metrics") else total_loss.item())
            global_step += 1

            # logging
            if global_step % tb_every == 0:
                # compute quick metrics for prints (gathering small scalars)
                scalar_loss = (sum([l.detach().clone() for l in losses]) / len(losses)).item() if isinstance(losses[0], torch.Tensor) else float(total_loss)
                if tb:
                    tb.add_scalar("train/total_loss", scalar_loss, global_step)
                log_metrics(json_fh, global_step, {"train_loss": float(scalar_loss), "epoch": epoch})

            # validation / checkpoint
            if global_step % val_every == 0:
                model.eval()
                # Run a lightweight validation: sample few batches from loaders
                val_stats = evaluate_model_simple(model, tokenizer, accelerator, orm_loader, prm_loader, device=accelerator.device, n_samples=128)
                # Combine metrics into a single score: higher ORM AUC and lower PRM MSE => score
                score = val_stats.get("orm_auc", 0.0) - val_stats.get("prm_mse", 0.0)
                if tb:
                    tb.add_scalar("val/orm_auc", val_stats.get("orm_auc", 0.0), global_step)
                    tb.add_scalar("val/prm_mse", val_stats.get("prm_mse", 0.0), global_step)
                log_metrics(json_fh, global_step, {"val": val_stats})
                # save best
                if score > best_val_metric:
                    best_val_metric = score
                    save_checkpoint(accelerator, model, optimizer, scheduler, run_dir=run_state.run_dir, step=global_step, name="best")
                model.train()

            # periodic checkpoint
            if global_step % save_every == 0:
                save_checkpoint(accelerator, model, optimizer, scheduler, run_dir=run_state.run_dir, step=global_step)

        # epoch end print
        elapsed = time.time() - start
        mean_loss = float(mean(epoch_losses)) if epoch_losses else 0.0
        print(f"[EPOCH {epoch+1}/{epochs}] mean_loss={mean_loss:.6f} time={elapsed:.1f}s")

    # final save
    save_checkpoint(accelerator, model, optimizer, scheduler, run_dir=run_state.run_dir, step=global_step, name="final")
    print("Training complete.")


# ---------- helper functions ----------
def evaluate_model_simple(model, tokenizer, accelerator, orm_loader, prm_loader, device, n_samples=256):
    """
    Lightweight evaluation: compute ORM classification metrics (logit->prob), PRM MSE on a small random subset.
    Returns dict {orm_auc, orm_acc, prm_mse, prm_corr}
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    model.eval()
    # ORM subset
    orm_preds = []
    orm_labels = []
    collected = 0
    with torch.no_grad():
        for batch in orm_loader:
            in_ids = batch["input_ids"].to(device)
            in_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=in_ids, attention_mask=in_mask, tokenizer=tokenizer)
            logits = out["orm_logits"].detach().cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            orm_preds.extend(probs.tolist())
            orm_labels.extend(labels.detach().cpu().numpy().tolist())
            collected += len(labels)
            if collected >= n_samples:
                break
    orm_auc = 0.0
    orm_acc = 0.0
    try:
        orm_auc = float(roc_auc_score(orm_labels, orm_preds))
        # acc by 0.5 threshold
        preds_bin = [1 if p >= 0.5 else 0 for p in orm_preds]
        orm_acc = float(accuracy_score(orm_labels, preds_bin))
    except Exception:
        pass

    # PRM subset
    prm_preds = []
    prm_labels = []
    collected = 0
    with torch.no_grad():
        for batch in prm_loader:
            in_ids = batch["input_ids"].to(device)
            in_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=in_ids, attention_mask=in_mask, tokenizer=tokenizer)
            preds = out["prm_pred"].detach().cpu().numpy().tolist()
            prm_preds.extend(preds)
            prm_labels.extend(labels.detach().cpu().numpy().tolist())
            collected += len(labels)
            if collected >= n_samples:
                break
    import numpy as np
    prm_mse = float(((np.array(prm_preds) - np.array(prm_labels)) ** 2).mean()) if prm_preds else 0.0
    prm_corr = 0.0
    try:
        prm_corr = float(np.corrcoef(prm_preds, prm_labels)[0, 1])
    except Exception:
        prm_corr = 0.0

    return {"orm_auc": orm_auc, "orm_acc": orm_acc, "prm_mse": prm_mse, "prm_corr": prm_corr}


def save_checkpoint(accelerator, model, optimizer, scheduler, run_dir: Path, step: int, name: str = None):
    """
    Save model + optimizer states via accelerator.
    Also export the two heads separately for serving later.
    """
    run_dir = Path(run_dir)
    ckpt_name = name or f"ckpt_{step}"
    ckpt_dir = run_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save using accelerator.save_state
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(str(ckpt_dir))
        # Additionally save head weights (CPU)
        try:
            # unwrap model if needed
            model_ = accelerator.unwrap_model(model)
            torch.save(model_.orm_head.state_dict(), ckpt_dir / "orm_head.pt")
            torch.save(model_.prm_head.state_dict(), ckpt_dir / "prm_head.pt")
            # save base model config
            model_.backbone.config.save_pretrained(str(ckpt_dir / "backbone_config"))
        except Exception as e:
            print(f"[WARN] Failed to save heads separately: {e}")
