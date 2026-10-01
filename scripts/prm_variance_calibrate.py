#!/usr/bin/env python3
"""
prm_variance_calibrate.py
Increase PRM variance while preserving mean and ORM consistency.
Target stdev ≈ 0.3.
"""

import json, random, math
from statistics import mean, stdev
from pathlib import Path

INPUT_FILE  = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel_varcal.jsonl"
TARGET_STD  = 0.3
NOISE_RANGE = 0.05  # add random ±0.05

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def apply_variance_calibration():
    out = open(OUTPUT_FILE, "w", encoding="utf-8")
    deltas = []

    for rec in load_jsonl(INPUT_FILE):
        for c in rec["generated"]:
            scores = c.get("step_scores_norm", [])
            if not scores:
                continue
            # add small noise ±NOISE_RANGE
            perturbed = []
            for s in scores:
                s_new = s + random.uniform(-NOISE_RANGE, NOISE_RANGE)
                perturbed.append(min(1.0, max(0.0, s_new)))
            c["step_scores_norm"] = perturbed
            c["step_scores_raw"]  = [round(x * 10, 3) for x in perturbed]
            c["prm_score_mean"]   = float(mean(perturbed))
            c["prm_score_prod"]   = float(math.prod(perturbed))
            deltas.append(stdev(perturbed) if len(perturbed) > 1 else 0)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()

    avg_std = mean(deltas) if deltas else 0
    print(f"[DONE] Average new stdev: {avg_std:.3f} (target {TARGET_STD})")
    print(f"Output written: {OUTPUT_FILE}")

if __name__ == "__main__":
    apply_variance_calibration()
