#!/usr/bin/env python3
"""
prm_mean_shift.py
Additive shift to step_scores_norm to reach a target mean (clipped).
"""

import json, math
from statistics import mean
from pathlib import Path

INPUT = "data/collected/gsm8k_training_chains_1000_labels_varcal_iter.jsonl"
OUTPUT = "data/collected/gsm8k_training_chains_1000_labels_final_for_training.jsonl"

TARGET_MEAN = 0.80
MAX_SHIFT = 0.05  # safety cap

def load_all(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs

def current_mean(recs):
    vals = []
    for r in recs:
        for c in r["generated"]:
            s = c.get("step_scores_norm", [])
            if s:
                vals.append(mean(s))
    return mean(vals) if vals else 0.0

recs = load_all(INPUT)
curr = current_mean(recs)
print("Current PRM mean:", curr)
shift = min(MAX_SHIFT, max(0.0, TARGET_MEAN - curr))
print("Applying shift:", shift)

if shift <= 0:
    print("No shift required. Exiting.")
else:
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for r in recs:
            for c in r["generated"]:
                s = c.get("step_scores_norm", [])
                if s:
                    new = [min(1.0, max(0.0, x + shift)) for x in s]
                    c["step_scores_norm"] = [float(x) for x in new]
                    c["step_scores_raw"] = [round(x * 10.0, 4) for x in new]
                    c["prm_score_mean"] = float(mean(new))
                    c["prm_score_prod"] = float(math.prod(new)) if new else 0.0
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote shifted dataset to:", OUTPUT)
