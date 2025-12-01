"""
setup_environment.py

Utilities for reproducible RM fine-tuning runs:
- deterministic seeding across torch/numpy/random
- TensorBoard + JSON logging initialization
- run directory creation and config saving
- helper to print environment and device info

Usage:
    from src.setup_environment import setup_run
    run_state = setup_run(run_name="rm_galactica_v1", cfg=cfg_dict)
    # run_state contains: run_dir, tb_writer, logger (json file handle), seed, device
"""

import os
import json
import time
import socket
import hashlib
import random
import getpass
import platform
import subprocess
from datetime import datetime, UTC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter


def _git_commit_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _hostname_short():
    try:
        return socket.gethostname().split('.')[0]
    except Exception:
        return "unknown_host"


def fix_random_seed(seed: int):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDNN deterministic flags (may slow training but increases reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_run_dir(base_dir: str, run_name: str):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"{timestamp}__{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _init_json_logger(path: Path):
    fh = open(path, "a", encoding="utf-8")
    # Write a header line
    header = {"created_at": datetime.now(UTC).isoformat()}
    fh.write(json.dumps({"meta": header}) + "\n")
    fh.flush()
    return fh


def _save_config(path: Path, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def setup_run(
    run_name: str,
    cfg: dict,
    base_dir: str = "runs",
    seed: int = 12345,
    use_tensorboard: bool = True,
    device: str | None = None,
):
    """
    Create run directories, set seeds, and initialize logging.
    Returns a SimpleNamespace with run_dir, tb_writer (or None), json_logger (file handle), device, seed.
    """
    # Ensure types
    cfg = dict(cfg) if cfg is not None else {}

    # Create run dir
    run_dir = _make_run_dir(base_dir, run_name)

    # Deterministic seed
    fix_random_seed(seed)

    # Save config for reproducibility
    cfg_path = run_dir / "config.json"
    _save_config(cfg_path, cfg)

    # Save environment snapshot
    env_snapshot = {
        "created_at": datetime.now(UTC).isoformat(),
        "user": getpass.getuser(),
        "hostname": _hostname_short(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "git_commit": _git_commit_hash(),
        "seed": int(seed),
    }
    with open(run_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env_snapshot, f, indent=2)

    # TensorBoard writer
    tb_writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except Exception:
            tb_writer = None

    # JSON logger for compact scalar writes
    json_log = _init_json_logger(run_dir / "metrics.jsonl")

    # Device resolution
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = SimpleNamespace(
        run_dir=run_dir,
        tb_writer=tb_writer,
        json_log=json_log,
        device=device,
        seed=int(seed),
        cfg=cfg,
    )

    # Print a short reproducibility banner
    banner = {
        "run_dir": str(run_dir),
        "device": device,
        "seed": seed,
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    print("=" * 80)
    print("RUN START".center(80))
    print(json.dumps(banner, indent=2))
    print("=" * 80)

    return state


def log_metrics(json_log_fh, step: int, metrics: dict):
    """
    Append a metrics JSON object to the metrics JSONL logger.
    """
    if json_log_fh is None:
        return
    record = {"step": int(step), "ts": datetime.now(UTC).isoformat(), "metrics": metrics}
    json_log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    json_log_fh.flush()


def close_run(state: SimpleNamespace):
    """Close resources like tensorboard writer and json file handles."""
    try:
        if state.tb_writer:
            state.tb_writer.close()
    except Exception:
        pass
    try:
        if state.json_log:
            state.json_log.close()
    except Exception:
        pass
