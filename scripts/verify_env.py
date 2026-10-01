#!/usr/bin/env python3
"""
scripts/verify_env.py
Quick check that Stage 3.1 environment setup works.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from src.setup_environment import setup_run, close_run

def main():
    cfg = yaml.safe_load(open("configs/train_config.yaml"))
    state = setup_run(run_name="verify_env", cfg=cfg, seed=cfg.get("seed", 42),
                      base_dir="runs", use_tensorboard=False)
    print(f"\n✅ Environment verified: device={state.device}, seed={state.seed}, run_dir={state.run_dir}")
    close_run(state)

if __name__ == "__main__":
    main()
