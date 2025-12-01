#!/usr/bin/env python3
"""
src/utils/training_utils.py — helpers for checkpointing & AMP.
"""

import os, torch
from datetime import datetime

def save_checkpoint(model, optimizer, scheduler, step, run_dir):
    path = os.path.join(run_dir, f"checkpoint_{step}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }, path)
    print(f"[CKPT] Saved checkpoint → {path}")

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[CKPT] Loaded checkpoint from {path}")
    return ckpt.get("step", 0)

def configure_amp(model, optimizer, dtype=torch.float16):
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    return scaler
