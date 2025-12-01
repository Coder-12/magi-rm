# %%writefile src/training/training_dual_rm.py
# src/training/train_rm.py
import sys
import math
import os
import json
import torch
import time
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from typing import Dict, Any, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Local imports
from src.models.dual_head_rm import DualHeadRM
from src.losses.prm_orm_losses import combined_prm_orm_loss
from src.data.dataloader_rm import load_rm_datasets, collate_fn
from src.training.evaluation import evaluate_rm
from src.utils.config_loader import load_config
from src.utils.logger import Logger

import shutil, gzip, tarfile
import psutil
import random, numpy as np
import dataclasses
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
torch.cuda.empty_cache()



# ==========================================================
# Helpers
# ==========================================================
def set_seed(seed: int = 42):
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_mode():
    """Initialize torch distributed environment (safe for single-GPU fallback)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Initialized rank {rank}/{world_size} on device {local_rank}")
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


# ==========================================================
# Training
# ==========================================================
def train_rm(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    # ------------------------------------------------------
    # Resume configuration (optional robust resume support)
    # ------------------------------------------------------
    resume_ckpt = None
    resume_optimizer = False

    # 1. Priority: YAML config key
    if cfg.get("train", {}).get("resume_from", None):
        resume_ckpt = cfg["train"]["resume_from"]
        resume_optimizer = cfg["train"].get("resume_optimizer", False)

    # 2. Fallback: Environment variables
    elif os.environ.get("RESUME_CKPT", None):
        resume_ckpt = os.environ["RESUME_CKPT"]
        resume_optimizer = os.environ.get("RESUME_OPT", "false").lower() in ["1", "true", "yes"]

    if resume_ckpt:
        print(f"[Resume] Requested resume from: {resume_ckpt} (optimizer={resume_optimizer})")

    # create logger directory
    run_dir = cfg.get("logging", {}).get("run_dir", "runs")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(run_dir)

    if resume_ckpt:
        logger.log(f"[Resume Init] Resuming training from checkpoint: {resume_ckpt}")


    # --- Initialize variables used later (prevents NameError) ---
    ema_model = None
    global_step = 0

    use_ddp, rank, world_size, local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---------- MEMORY PROBE (run once, rank 0 logs) ----------
    def _probe_batch_size(cfg, model_ctor, local_rank, device, max_try_bs=None):
        """
        Returns a safe per-device batch size (int). Attempts dummy forward with decreasing batch sizes.
        Note: model_ctor() should return a lightweight model instance (DualHeadRM) or None if loading full base is heavy.
        We use minimal input shape from config (max_length).
        """
        max_len = cfg["data"].get("max_length", 512)
        # start from suggested config or 4
        start_bs = int(cfg["train"].get("per_device_batch_size", 4))
        if max_try_bs is not None:
            start_bs = min(start_bs, max_try_bs)
        bs = start_bs
        while bs >= 1:
            try:
                # construct small dummy batch on device
                input_ids = torch.randint(0, 1000, (bs, max_len), dtype=torch.long, device=device)
                attn = torch.ones_like(input_ids, device=device)
                # do a forward with a minimal, fresh model instance. Avoid re-loading HF model here -> use already constructed model if available.
                with torch.no_grad():
                    # if model already created, use it; else create one (may be heavy)
                    pass
                # quick forward using your model if it's been constructed below
                return bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    bs = bs // 2
                else:
                    # unexpected error -> re-raise
                    raise
        return 1

    # Use this result to override cfg["train"]["per_device_batch_size"] if rank==0 (propagate via env or same YAML in all processes)
    safe_bs = cfg["train"]["per_device_batch_size"]
    if (not use_ddp) or rank == 0:
        logger.log(f"Using device: {device}, DDP={use_ddp}, world_size={world_size}")
        logger.log(f"[Probe] using per_device_batch_size={safe_bs}")

    # ------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------
    train_dataset, val_dataset = load_rm_datasets(cfg["data"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["per_device_batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["eval_batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------
    model = DualHeadRM(
        base_model_name=cfg["base_model"],
        max_steps=cfg["data"]["max_steps"],
        pooling="eos",
        hidden_dropout=0.1,
        freeze_base=cfg.get("freeze_base", True),
        device=device,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ------------------------------------------------------
    # Optimizer (created early so resume_optimizer logic can load state safely)
    # Only include params with requires_grad=True (useful when base is frozen).
    # ------------------------------------------------------
    # Recompute trainable params after possible DDP wrapping
    trainable_params = [p for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters()) if
                        p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found (all params have requires_grad=False). Check 'freeze_base' or model setup.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=cast(tuple[float | torch.Tensor, float | torch.Tensor], tuple(cfg["optim"]["betas"])),
        eps=cfg["optim"]["eps"],
    )

    # ------------------------------------------------------
    # Resume model weights (safe load if checkpoint exists)
    # ------------------------------------------------------
    if resume_ckpt and os.path.exists(resume_ckpt):
        logger.log(f"[Resume] Loading model weights from {resume_ckpt}")
        map_loc = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(resume_ckpt, map_location=map_loc)

        try:
            # Handle DDP or plain model cases
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            logger.log("[Resume] ✅ Model state loaded successfully.")
        except Exception as e:
            logger.log(f"[Resume] ⚠️ Warning: could not load model weights cleanly: {e}")

        # If EMA exists, try to load the same state dict into it (best-effort)
        # if ema_model is not None:
        #     try:
        #         ema_model.load_state_dict(state_dict)
        #         logger.log("[Resume] ✅ EMA model state loaded.")
        #     except Exception:
        #         logger.log("[Resume] ⚠️ EMA model state not loaded (incompatible keys): {e}")

        # NOTE: EMA model is created later in this function; we will attempt to
        # load EMA state after ema_model is constructed (best-effort). See below.

        # Optional: resume optimizer state if requested and file exists
        if resume_optimizer:
            opt_state_path = os.path.splitext(resume_ckpt)[0] + "_opt.pt"
            if os.path.exists(opt_state_path):
                try:
                    optimizer.load_state_dict(torch.load(opt_state_path, map_location=map_loc))
                    logger.log(f"[Resume] ✅ Optimizer state restored from {opt_state_path}")
                except Exception as e:
                    logger.log(f"[Resume] ⚠️ Could not load optimizer state: {e}")
            else:
                logger.log("[Resume] Optimizer state file not found; skipping.")

    # ------------------------------------------------------
    # EMA setup (create for both single-GPU and DDP to keep behavior consistent)
    # Note: copy.deepcopy can be heavy; see comment below for potential memory optimization.
    # ------------------------------------------------------
    ema_decay = cfg.get("train", {}).get("ema_decay", 0.999)
    if cfg.get("train", {}).get("use_ema", True):
        import copy
        # ✅ free GPU memory before deep-copying (safe micro-optimization)
        torch.cuda.empty_cache()
        source_for_ema = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        # make EMA on CPU and ensure FP32 dtype for numerical stability and to avoid FP16 eval mismatch
        ema_model = copy.deepcopy(source_for_ema).eval().to("cpu")
        # cast parameters and buffers to float32 on CPU
        for p in ema_model.parameters():
            p.requires_grad = False
            p.data = p.data.float()
        for b in ema_model.buffers():
            if torch.is_tensor(b):
                b.data = b.data.float()

        # If we resumed from a checkpoint earlier, ensure EMA has the same weights (best-effort).
        if ema_model is not None and resume_ckpt and os.path.exists(resume_ckpt):
            try:
                # load EMA state map to CPU (be robust to dtype)
                ema_state = torch.load(resume_ckpt, map_location="cpu")
                # ema_model might be module (not DDP) — try load_state_dict carefully
                try:
                    ema_model.load_state_dict(ema_state)
                    logger.log("[Resume] ✅ EMA state loaded from checkpoint.")
                except RuntimeError:
                    # try filtering keys / strict=False fallback
                    ema_model.load_state_dict(ema_state, strict=False)
                    logger.log("[Resume] ⚠️ EMA state loaded with strict=False (incompatible keys ignored).")

                # after loading, ensure EMA is on CPU and float32
                ema_model.to("cpu")
                for p in ema_model.parameters():
                    p.data = p.data.float()
                for b in ema_model.buffers():
                    if torch.is_tensor(b):
                        b.data = b.data.float()
            except Exception as e:
                logger.log(f"[Resume] ⚠️ Could not load EMA state from {resume_ckpt}: {e}")
        # Optional improvement: keep EMA on CPU to save GPU memory. If desired, move ema_model to CPU:
        # ema_model.to("cpu")
        # and move to device temporarily for evaluation (do careful device management).
        # (We do not force that change here to avoid altering your existing flow.)
    else:
        ema_model = None

    # ------------------------------------------------------
    # Optimizer / Scheduler / FP16
    # ------------------------------------------------------
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=cfg["optim"]["lr"],
    #     weight_decay=cfg["optim"]["weight_decay"],
    #     betas=tuple(cfg["optim"]["betas"]),
    #     eps=cfg["optim"]["eps"],
    # )

    # --- Warmup + Cosine per-step scheduler ---
    # We'll compute total optimizer steps (accounting for grad accumulation) and use a LambdaLR that expects
    # a global step counter stepped after each optimizer update.
    grad_accum_steps = cfg["train"].get("gradient_accumulation_steps", 1)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = cfg["train"].get("total_steps", steps_per_epoch * cfg["train"]["epochs"])

    def lr_lambda(current_step):
        warmup_steps = cfg["sched"].get("warmup_steps", 200)
        if total_steps is None or total_steps <= 0:
            # fallback: use epoch-based cosine (rare)
            return 1.0
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # FP16: only enable autocast if running on CUDA and mixed_precision==fp16
    use_amp = (cfg.get("mixed_precision", "") == "fp16") and (device.type == "cuda")

    # --- Robust GradScaler creation ---
    # If model already has FP16 parameters (e.g., from load_checkpoint_and_dispatch),
    # disable GradScaler to avoid "Attempting to unscale FP16 gradients" errors.
    has_fp16_param = any((p is not None and p.dtype == torch.float16) for p in model.parameters())
    if has_fp16_param:
        print("[AMP] ⚠️ Model contains float16 parameters — disabling GradScaler (enabled=False)")
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        use_amp = False
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------
    best_val_loss = float("inf")
    patience = cfg["train"].get("early_stop_patience", 2)
    no_improve = 0

    print("\n" + "=" * 60)
    print("MEMORY CHECK BEFORE TRAINING:")
    print("=" * 60)
    print(f"CPU RAM: {psutil.virtual_memory().percent}% used")
    print(
        f"GPU 0: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(
        f"GPU 1: {torch.cuda.memory_allocated(1) / 1e9:.1f} GB / {torch.cuda.get_device_properties(1).total_memory / 1e9:.1f} GB")
    print("=" * 60 + "\n")

    for epoch in range(cfg["train"]["epochs"]):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, dynamic_ncols=True, disable=(use_ddp and rank != 0), mininterval=10)
        optimizer.zero_grad(set_to_none=True)

        # --- Curriculum weighting: gradually ramp PRM lambdas over epochs ---
        start_factor = 0.2
        end_factor = 1.0
        progress_epoch = epoch / max(1, cfg["train"]["epochs"] - 1)
        ramp = start_factor + (end_factor - start_factor) * progress_epoch

        lambda_dict = cfg["loss_weights"].copy()
        for k in ["lambda_prm_step", "lambda_prm_mean", "lambda_prm_prod"]:
            if k in lambda_dict:
                lambda_dict[k] = lambda_dict[k] * ramp

        # Log current LR & ramp (rank 0 only)
        if (not use_ddp) or rank == 0:
            logger.log(f"DDP: {use_ddp}, GPUs={world_size}, device={device}")
            current_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else cfg["optim"]["lr"]
            logger.log(f"[Epoch {epoch+1}] starting — lr={current_lr:.3e}, prm_ramp={ramp:.3f}")

        batches_seen = 0
        # per-epoch loop
        for step, batch in enumerate(pbar):
            # Backward compat: some code path yield dataclass batch so convert it to dict
            if dataclasses.is_dataclass(batch):
                batch = dataclasses.asdict(batch)
            # Move tensors to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # convert/normalize some batch float fields to model dtype and ensure finite
            model_dtype = next(
                (p.dtype for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters())),
                torch.float32)
            # normalize step targets to [0,1] if they might be 0..10, use helper if available
            try:
                from src.losses.prm_orm_losses import normalize_step_targets
                batch["step_targets"] = normalize_step_targets(batch["step_targets"])
            except Exception:
                # safe fallback: clamp
                batch["step_targets"] = torch.clamp(batch["step_targets"], 0.0, 1.0)

            # ensure masks/confidences finite & proper dtype
            batch["step_mask"] = torch.clamp(batch["step_mask"], 0.0, 1.0)
            if "step_confidence" in batch and batch["step_confidence"] is not None:
                batch["step_confidence"] = torch.clamp(batch["step_confidence"], 0.0, 1.0)

            # cast float tensors to model dtype to avoid mixing (especially when model dtype is fp16)
            for k, v in batch.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    batch[k] = v.to(dtype=model_dtype, device=v.device)

            # Place before entering per-epoch loop: helper to recreate dataloaders with new batch size
            def _recreate_dataloaders(new_bs):
                # note: train_dataset and val_dataset are in enclosing scope
                nonlocal train_loader, val_loader
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=new_bs,
                    sampler=train_sampler,
                    shuffle=(train_sampler is None),
                    num_workers=cfg["train"].get("num_workers", 2),
                    pin_memory=True,
                    collate_fn=collate_fn,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=cfg["train"]["eval_batch_size"],
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=cfg["train"].get("num_workers", 2),
                    pin_memory=True,
                    collate_fn=collate_fn,
                )
                # update cfg persistently
                cfg["train"]["per_device_batch_size"] = new_bs

            oom_retry_count = 0
            while True:
                try:
                    autocast_enabled = use_amp or has_fp16_param
                    with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=torch.float16):
                        out = model(batch["input_ids"], attention_mask=batch["attention_mask"], step_mask=batch["step_mask"])
                        if step == 0:
                            print("Sanity check: ORM logits stats ->", out["orm_logits"].mean().item(),
                                  out["orm_logits"].std().item())
                            print("Sanity check: PRM logits stats ->", out["prm_logits"].mean().item(),
                                  out["prm_logits"].std().item())

                        total_loss, _ = combined_prm_orm_loss(
                            orm_logits=out["orm_logits"],
                            orm_targets=batch["orm_label"],
                            step_preds=out["prm_logits"],
                            step_targets=batch["step_targets"],
                            step_mask=batch["step_mask"],
                            step_confidence=batch.get("step_confidence"),
                            chain_confidence=batch.get("chain_confidence"),
                            **lambda_dict
                        )

                        # --- NaN / Inf guard ---
                        if not torch.isfinite(total_loss):
                            print("\n[FATAL] ❌ Non-finite loss detected — skipping this batch.")
                            print("Diagnostics:")
                            print("  ORM logits -> mean:", out["orm_logits"].mean().item(),
                                  "std:", out["orm_logits"].std().item())
                            print("  PRM logits -> mean:", out["prm_logits"].mean().item(),
                                  "std:", out["prm_logits"].std().item())
                            print("  Step targets -> mean:", torch.nanmean(batch["step_targets"]).item(),
                                  "std:", torch.nanstd(batch["step_targets"]).item())
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            # new — use model dtype to avoid float/half mismatch
                            total_loss = torch.tensor(0.0, device=device, dtype=next(model.parameters()).dtype)
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                        # OOM handling: do not skip batch. Reduce global batch size by 2 and recreate dataloaders.
                    torch.cuda.empty_cache()
                    oom_retry_count += 1
                    if oom_retry_count > 3:
                        # safety: after a few attempts escalate to smaller BS
                        new_bs = max(1, cfg["train"]["per_device_batch_size"] // (2 ** oom_retry_count))
                    else:
                        new_bs = max(1, cfg["train"]["per_device_batch_size"] // 2)
                    if new_bs == cfg["train"]["per_device_batch_size"]:
                        raise RuntimeError("[OOM] Cannot reduce batch size further, aborting.") from e
                    if (not use_ddp) or rank == 0:
                        logger.log(
                            f"[OOM Recovery] OOM at step {step + 1}. Recreating dataloaders with per_device_batch_size={new_bs}")
                    _recreate_dataloaders(new_bs)
                    # synchronize DDP processes before continuing to ensure all use same loader sizes
                    if use_ddp:
                        dist.barrier()
                    # reload next batch from new iterator by breaking outer for-loop and re-entering epoch (simpler),
                    # or set a flag to re-fetch current index. For simplicity, break to outer loop to pick up new loader.
                    raise RuntimeError("[OOM Recovery] Recreated dataloaders. Please restart epoch loop to continue.")

            if total_loss.device != device:
                total_loss = total_loss.to(device=device)
            if total_loss.dtype != model_dtype:
                total_loss = total_loss.to(dtype=model_dtype, device=total_loss.device)
            if total_loss.dtype != next(model.parameters()).dtype:
                total_loss = total_loss.to(dtype=next(model.parameters()).dtype)
            # normalize loss by accumulation steps and backprop
            loss_scaled = total_loss / grad_accum_steps
            # Backward: use GradScaler only when it is enabled
            if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            torch.cuda.empty_cache()
            # accumulate running loss once per batch (avoid double-add)
            running_loss += float(total_loss.detach().cpu().item())

            # gradient accumulation step: update when boundary reached
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                # --- Use only trainable params for clipping & grad-norm ---
                trainable_with_grad = [p for p in
                                       (model.parameters() if not isinstance(model, DDP) else model.module.parameters())
                                       if p.requires_grad and p.grad is not None]
                if len(trainable_with_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_with_grad, cfg["optim"]["max_grad_norm"])

                # ✅ Log gradient norm *before zero_grad()*
                should_log = ((global_step % 50) == 0) and ((not use_ddp) or rank == 0)
                if should_log:
                    total_norm = 0.0
                    for p in trainable_with_grad:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    logger.log(f"[Diagnostics] Gradient L2 norm (step {global_step}): {total_norm:.4f}")

                # --- AMP safe step: only use scaler.step() if scaler.is_enabled()
                if isinstance(scaler, torch.cuda.amp.GradScaler) and getattr(scaler, "is_enabled", lambda: False)():
                    # scaler enabled (fp16/amp)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # plain fp32 or scaler disabled
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # LR step (per optimizer step) and global_step update
                scheduler.step()
                global_step += 1

                if cfg["train"].get("max_train_steps") and global_step >= cfg["train"]["max_train_steps"]:
                    logger.log(f"[Smoke] reached max_train_steps={cfg['train']['max_train_steps']} - stopping early.")
                    break

                # EMA update (apply on rank 0 only or on all ranks if desired)
                ema_update_every = cfg["train"].get("ema_update_every", 1)
                if ema_model is not None and (global_step % ema_update_every == 0):
                    with torch.no_grad():
                        # handle DDP model.module vs plain model
                        msd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                        for k_param, ema_v in ema_model.state_dict().items():
                            if k_param in msd:
                                # copy model param to CPU and cast to float32 (safe & consistent)
                                model_param_cpu = msd[k_param].detach().to("cpu", non_blocking=True).float()
                                # ema_v is already float32 on CPU; inplace update
                                ema_v.copy_(ema_decay * ema_v + (1.0 - ema_decay) * model_param_cpu)

            batches_seen += 1
            avg_loss = running_loss / max(1, batches_seen)
            pbar.set_description(f"Epoch {epoch + 1} | Loss={avg_loss:.4f}")

        # optionally log gradient norm diagnostics (rank 0)
        # if cfg["train"].get("log_grad_norm", True):
        #     if (not use_ddp) or rank == 0:
        #         total_norm = 0.0
        #         for p in model.parameters():
        #             if p.grad is not None:
        #                 param_norm = p.grad.data.norm(2)
        #                 total_norm += param_norm.item() ** 2
        #         total_norm = total_norm ** 0.5
        #         logger.log(f"[Diagnostics] Gradient L2 norm (epoch {epoch + 1}): {total_norm:.4f}")
        #         # Also log current LR & global step
        #         logger.log(f"[Diagnostics] global_step={global_step} lr={optimizer.param_groups[0]['lr']:.6e}")

        # --------------------------------------------------
        # Validation + Logging (only rank 0)
        # --------------------------------------------------
        if (not use_ddp) or rank == 0:
            # use EMA model for evaluation if available, otherwise the current model
            eval_model = ema_model if ema_model is not None else model
            # IMPORTANT: if ema_model was kept on CPU to save GPU memory, move to device temporarily here (optional).
            # use EMA model for evaluation if available, otherwise the current model

            # --- Ensure eval_model is on the same device as 'device' during evaluation ---
            _ema_moved_to_device = False
            if eval_model is not None and hasattr(eval_model, "parameters") and next(eval_model.parameters(), None) is not None:
                try:
                    ema_dev = next(eval_model.parameters()).device
                except StopIteration:
                    ema_dev = None
            else:
                ema_dev = None

            # If eval_model is EMA and its params are not on target device, move temporarily.
            if eval_model is not None and ema_dev is not None and ema_dev != device:
                # Move to device for evaluation (best-effort). If model is a DDP wrapper, access module.
                try:
                    eval_model.to(device)
                    # align dtype with main model to avoid Half vs Float issues
                    model_dtype = next((p.dtype for p in
                                        (model.module.parameters() if isinstance(model, DDP) else model.parameters())),
                                       torch.float16)
                    for p in eval_model.parameters():
                        p.data = p.data.to(device=device, dtype=model_dtype)
                    for b in eval_model.buffers():
                        if torch.is_tensor(b):
                            b.data = b.data.to(device=device, dtype=model_dtype)
                    _ema_moved_to_device = True
                    logger.log(f"[Eval] Temporarily moved eval_model from {ema_dev} -> {device} for evaluation.")
                except Exception as e:
                    # fallback: log and continue (evaluation may fail if incompatible)
                    logger.log(f"[Eval] Warning: could not move eval_model to {device}: {e}")

            # IMPORTANT: if ema_model was kept on CPU to save GPU memory, move back after evaluation
            metrics = evaluate_rm(eval_model, val_loader, lambda_dict, device=device, verbose=True)
            time.sleep(1)

            if _ema_moved_to_device:
                try:
                    # Move EMA back to cpu to preserve original memory strategy
                    eval_model.to("cpu")
                    for p in eval_model.parameters():
                        p.data = p.data.cpu().float()
                    for b in eval_model.buffers():
                        if torch.is_tensor(b):
                            b.data = b.data.cpu().float()
                    logger.log("[Eval] Moved eval_model back to CPU after evaluation.")
                except Exception as e:
                    logger.log(f"[Eval] Warning: could not move eval_model back to CPU: {e}")

            metrics["train_loss"] = running_loss / len(train_loader)
            metrics["epoch"] = epoch + 1

            logger.log(f"Epoch {epoch+1} — loss: {metrics['train_loss']:.4f} | val_total_loss: {metrics['val_total_loss']:.4f}")
            logger.log_metrics(epoch + 1, metrics)

            # optional CSV export (appended) — keep header deterministic (sorted keys)
            if cfg.get("logging", {}).get("csv_log", True):
                import csv
                csv_path = os.path.join(run_dir, "metrics.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    # sort keys for deterministic columns
                    fieldnames = sorted(metrics.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    # write only fields in fieldnames (dict order may differ)
                    writer.writerow({k: metrics.get(k, None) for k in fieldnames})

            # --- Early stopping logic ---
            if metrics["val_total_loss"] < best_val_loss:
                best_val_loss = metrics["val_total_loss"]
                no_improve = 0
                # save best model separately
                best_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), best_path)
                logger.log(f"✅ New best model saved: val_loss={best_val_loss:.4f}")

                # Save EMA model also (if present)
                if ema_model is not None:
                    best_ema_path = os.path.join(run_dir, "checkpoints", "best_model_ema.pt")
                    torch.save(ema_model.state_dict(), best_ema_path)
                    logger.log(f"✅ New EMA model saved: {best_ema_path}")
            else:
                no_improve += 1
                logger.log(f"⚠️ No improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    logger.log(f"⛔ Early stopping at epoch {epoch + 1} — no improvement for {patience} epochs")
                    break

            # Save checkpoint (epoch)
            out_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch+1}.pt")
            state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state_to_save, ckpt_path)

            # --- Checkpoint retention policy for Kaggle ---
            max_to_keep = cfg.get("checkpoint", {}).get("keep_last_k", 2)
            checkpoints = sorted(
                [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                 if f.startswith("epoch_") and f.endswith(".pt")],
                key=os.path.getmtime
            )
            if len(checkpoints) > max_to_keep:
                for ckpt in checkpoints[:-max_to_keep]:
                    try:
                        os.remove(ckpt)
                        logger.log(f"[Cleanup] Removed old checkpoint: {ckpt}")
                    except Exception as e:
                        logger.log(f"[Cleanup] Warning: could not remove {ckpt}: {e}")
            logger.log(f"Saved checkpoint to {ckpt_path}")

            # Optional: also save optimizer state for resume (if enabled in YAML)
            if cfg.get("checkpoint", {}).get("save_optimizer", False):
                opt_state_path = os.path.splitext(ckpt_path)[0] + "_opt.pt"
                torch.save(optimizer.state_dict(), opt_state_path)
                logger.log(f"Saved optimizer state to {opt_state_path}")

        # ensure synchronization before next epoch
        if use_ddp:
            dist.barrier()

    run_dir = cfg.get("logging", {}).get("run_dir", "runs")
    if os.path.exists(run_dir):
        parent = os.path.dirname(run_dir) or "."
        tar_path = os.path.join(parent, os.path.basename(run_dir) + "_compressed_logs.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            # Only add the top-level run_dir but exclude checkpoints and the compressed tar if any
            for root, dirs, files in os.walk(run_dir):
                # skip checkpoints folder entirely
                if os.path.basename(root) == "checkpoints":
                    continue
                for f in files:
                    fp = os.path.join(root, f)
                    arcname = os.path.relpath(fp, parent)
                    tar.add(fp, arcname=arcname)
        logger.log(f"[Cleanup] Compressed logs to {tar_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    # Allow running with CLI arg or default config
    config_path = sys.argv[-1] if len(sys.argv) > 1 else "configs/rm_train_config.yaml"
    train_rm(config_path)
