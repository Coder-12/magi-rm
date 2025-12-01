#!/usr/bin/env python3
"""
Add derived fields for Reward Model training:
- prm_solution_gold: numerically-stable aggregated PRM target
- step_confidence: heuristic per-step reliability score
- chain_confidence: average of step_confidence values

This script is *non-destructive*: it preserves every existing key
and only appends new derived fields. It is numerically stable and
avoids biasing or renormalizing any original score.
"""

import json
import math
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ============================================================
# Config
# ============================================================

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = INPUT_DIR / "confident"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SPLITS = ["prm_train.jsonl", "prm_val.jsonl", "prm_test.jsonl"]

EPS = 1e-12
DEFAULT_CONFIDENCE = 0.8  # baseline trust for GPT-4o-mini labels

np.random.seed(42)


# ============================================================
# Utility functions
# ============================================================

def safe_log_prod(step_targets):
    """
    Compute numerically stable product of step_targets using log-space.
    Avoids underflow for long reasoning chains.
    """
    steps = np.clip(np.array(step_targets, dtype=np.float64), EPS, 1.0)
    log_prod = np.sum(np.log(steps))
    return float(np.exp(log_prod))


def heuristic_confidence(step_targets):
    """
    Heuristic confidence estimation.
    - Steps with near 0.0 or 1.0 are higher confidence (model certainty)
    - Very flat / uniform step_targets => lower confidence
    - Moderate entropy => moderate confidence
    Output: list of confidences per step ∈ (0,1)
    """
    steps = np.array(step_targets, dtype=np.float64)
    if len(steps) == 0:
        return []

    # entropy-based reliability
    p = steps / (steps.sum() + EPS)
    entropy = -np.sum(p * np.log(p + EPS)) / math.log(len(steps))
    # entropy ∈ [0,1], invert for confidence
    entropy_conf = 1.0 - entropy

    # magnitude-based reliability (confident if near 0 or 1)
    mag_conf = 1.0 - 4 * np.abs(steps - 0.5) * np.abs(steps - 0.5)

    # blend both heuristics (equal weights)
    blended = 0.5 * mag_conf + 0.5 * entropy_conf

    # clip to safe interval
    blended = np.clip(blended, 0.5, 0.95)

    return blended.tolist()


# ============================================================
# Main processing
# ============================================================

def process_split(split_name):
    input_path = INPUT_DIR / split_name
    output_path = OUTPUT_DIR / split_name
    print(f"[INFO] Processing {input_path} → {output_path}")

    new_records = []
    with open(input_path, "r") as f:
        for line in tqdm(f, desc=f"{split_name}", ncols=100):
            record = json.loads(line.strip())
            if "step_targets" not in record:
                # Skip ORM data, copy as-is
                new_records.append(record)
                continue

            steps = record.get("step_targets", [])
            if len(steps) == 0:
                record["prm_solution_gold"] = 0.0
                record["step_confidence"] = []
                record["chain_confidence"] = 0.0
            else:
                # --- Compute numerically stable PRM product ---
                record["prm_solution_gold"] = safe_log_prod(steps)

                # --- Compute heuristic confidences ---
                step_conf = heuristic_confidence(steps)
                record["step_confidence"] = step_conf
                record["chain_confidence"] = float(np.mean(step_conf))

            new_records.append(record)

    # Non-destructive write (preserve order)
    with open(output_path, "w") as f:
        for rec in new_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"[SUCCESS] Wrote {len(new_records)} records → {output_path}")


def main():
    print("==========================================================")
    print(" Add Confidence and Gold Targets to PRM Splits ")
    print("==========================================================")
    for split in SPLITS:
        process_split(split)
    print("==========================================================")
    print("[DONE] All splits processed successfully!")
    print(f"Files saved under: {OUTPUT_DIR.resolve()}")
    print("==========================================================")


if __name__ == "__main__":
    main()
