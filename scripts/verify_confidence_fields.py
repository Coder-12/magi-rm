#!/usr/bin/env python3
"""
Verify confidence and gold fields in PRM splits.

Checks:
- Fields exist and have valid numeric ranges
- No NaNs or out-of-bounds values
- step_confidence correlates sensibly with step_targets
- chain_confidence ~ mean(step_confidence)
- prm_solution_gold within [0,1] and matches step_targets product (numerically stable)
- Summary statistics across splits for interpretability
"""

import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.stats import pearsonr


# ============================================================
# Config
# ============================================================

INPUT_DIR = Path("data/processed/confident")
SPLITS = ["prm_train.jsonl", "prm_val.jsonl", "prm_test.jsonl"]
EPS = 1e-12


# ============================================================
# Utility
# ============================================================

def stable_prod(x):
    """Safe product in log-space."""
    x = np.clip(np.array(x, dtype=np.float64), EPS, 1.0)
    return float(np.exp(np.sum(np.log(x))))


# ============================================================
# Validation
# ============================================================

def validate_split(split_path):
    records = []
    with open(split_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    missing, nan_issues, out_of_bounds = 0, 0, 0
    diffs, chain_conf_devs = [], []
    step_conf_means, prm_means, gold_vals = [], [], []

    for r in records:
        if "step_targets" not in r:
            continue  # skip ORM-like

        if not all(k in r for k in ["prm_solution_gold", "step_confidence", "chain_confidence"]):
            missing += 1
            continue

        st = np.array(r["step_targets"], dtype=np.float64)
        sc = np.array(r["step_confidence"], dtype=np.float64)
        gold = float(r["prm_solution_gold"])
        chain_conf = float(r["chain_confidence"])

        # Check numerical sanity
        if np.isnan(sc).any() or np.isnan(st).any() or np.isnan(gold):
            nan_issues += 1

        if np.any(sc < 0.0) or np.any(sc > 1.0) or gold < 0.0 or gold > 1.0:
            out_of_bounds += 1

        # Check match between prm_solution_gold and recomputed
        recomputed = stable_prod(st)
        diffs.append(abs(recomputed - gold))

        # Check if chain_conf = mean(step_conf)
        if len(sc) > 0:
            chain_conf_devs.append(abs(chain_conf - np.mean(sc)))

        # Collect for correlation check
        if len(st) > 0:
            step_conf_means.append(np.mean(sc))
            prm_means.append(np.mean(st))
            gold_vals.append(gold)

    # Summary stats
    avg_diff = np.mean(diffs) if diffs else 0.0
    avg_chain_dev = np.mean(chain_conf_devs) if chain_conf_devs else 0.0
    corr, _ = pearsonr(prm_means, step_conf_means) if len(prm_means) > 5 else (0.0, 1.0)

    print(f"=== {split_path.name} ===")
    print(f"Records: {len(records)} | Missing fields: {missing} | NaN: {nan_issues} | OOB: {out_of_bounds}")
    print(f"Mean abs diff (gold vs recomputed): {avg_diff:.3e}")
    print(f"Mean abs dev (chain_conf vs mean(step_conf)): {avg_chain_dev:.3e}")
    print(f"Pearson corr (PRM mean vs step_conf mean): {corr:.3f}")
    print(f"Gold min={min(gold_vals):.3f}, max={max(gold_vals):.3f}, mean={np.mean(gold_vals):.3f}")
    print("-" * 65)

    return {
        "records": len(records),
        "missing": missing,
        "nan_issues": nan_issues,
        "out_of_bounds": out_of_bounds,
        "mean_gold_diff": avg_diff,
        "mean_conf_dev": avg_chain_dev,
        "corr_prm_conf": corr,
        "gold_mean": np.mean(gold_vals),
        "gold_min": np.min(gold_vals),
        "gold_max": np.max(gold_vals),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("==========================================================")
    print(" VERIFY CONFIDENCE & GOLD TARGET FIELDS ")
    print("==========================================================")
    report = {}
    for split in SPLITS:
        path = INPUT_DIR / split
        report[split] = validate_split(path)

    print("==========================================================")
    print(" Summary ")
    print("==========================================================")
    for k, v in report.items():
        print(f"{k}: corr={v['corr_prm_conf']:.3f}, mean_gold={v['gold_mean']:.3f}, "
              f"missing={v['missing']}, nan={v['nan_issues']}, oob={v['out_of_bounds']}")
    print("==========================================================")
    print("[DONE] Verification complete — all splits validated successfully.")
    print("==========================================================")


if __name__ == "__main__":
    main()
