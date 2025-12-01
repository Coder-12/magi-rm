#!/usr/bin/env python3
"""
src/env_setup.py — reproducible environment bootstrap.
Sets deterministic seeds, mixed precision mode, device, and DDP initialization.
"""

import os, random, torch, numpy as np

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[ENV] Global seed set to {seed}")

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[ENV] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[ENV] Using CPU")
    return dev

def setup_mixed_precision(dtype="fp16"):
    if dtype.lower() in ["fp16", "float16"]:
        return torch.float16
    if dtype.lower() in ["bf16", "bfloat16"]:
        return torch.bfloat16
    return torch.float32
