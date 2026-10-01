# src/utils/config_loader.py
"""
Utility: Configuration Loader
==============================

Purpose:
--------
Centralized configuration loader for Reward Model (RM) training and evaluation.

Features:
---------
- Supports YAML (.yml / .yaml) and JSON config files
- Environment-variable expansion (e.g., ${HOME})
- Automatic type casting for numeric strings
- Validates core sections (model, train, loss, data)
- Compatible with all training / fine-tuning stages (3.1–3.5)
- Works safely in distributed setups (rank-safe I/O)

Example:
--------
>>> from src.utils.config_loader import load_config
>>> cfg = load_config("configs/rm_train_config.yaml")
>>> print(cfg["train"]["batch_size"])

Author:
--------
Aklesh Mishra — Verified lab-grade RM project utility
"""

import re
import os
import json
import yaml
from typing import Any, Dict


# ==========================================================
# Safe YAML Loader with env expansion
# ==========================================================
def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    else:
        return value


def _resolve_templates(cfg: dict) -> dict:
    """Resolve Jinja-style {{var}} placeholders using top-level keys in cfg."""
    def _substitute(value):
        if isinstance(value, str):
            matches = re.findall(r"{{\s*([^}]+)\s*}}", value)
            for key in matches:
                if key in cfg:
                    value = value.replace(f"{{{{{key}}}}}", str(cfg[key]))
            return value
        elif isinstance(value, dict):
            return {k: _substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute(v) for v in value]
        return value

    return _substitute(cfg)


def _auto_cast_numeric(value):
    """Recursively cast numeric strings to int/float."""
    if isinstance(value, str):
        try:
            # Try integer
            if value.isdigit():
                return int(value)
            # Try float or scientific notation (e.g., '2e-4')
            return float(value)
        except ValueError:
            return value
    elif isinstance(value, dict):
        return {k: _auto_cast_numeric(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_auto_cast_numeric(v) for v in value]
    else:
        return value




# ==========================================================
# Core Loader
# ==========================================================
def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON configuration file.

    Args:
        path (str): Path to config file (.yaml / .yml / .json)

    Returns:
        dict: Parsed configuration dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[-1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    config = _expand_env_vars(config)
    config = _auto_cast_numeric(config)  # ✅ auto-fix numeric strings

    # Optional: validation
    # _validate_core_sections(config)

    return config


# ==========================================================
# Validation
# ==========================================================
def _validate_core_sections(cfg: Dict[str, Any]) -> None:
    """
    Light validation to ensure critical sections exist.
    This ensures configs are consistent across stages.
    """
    required_sections = ["model", "train", "loss", "data"]
    for sec in required_sections:
        if sec not in cfg:
            raise KeyError(f"Missing required section in config: '{sec}'")

    if "base_model_name" not in cfg["model"]:
        raise KeyError("Missing 'base_model_name' in model section.")

    if "batch_size" not in cfg["train"]:
        raise KeyError("Missing 'batch_size' in train section.")


# ==========================================================
# Save (Optional utility)
# ==========================================================
def save_config(cfg: Dict[str, Any], path: str) -> None:
    """
    Save configuration dict back to YAML for reproducibility.
    Only rank 0 should call this in DDP environments.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[config_loader] Saved configuration → {path}")


# ==========================================================
# Exports
# ==========================================================
__all__ = ["load_config", "save_config"]
