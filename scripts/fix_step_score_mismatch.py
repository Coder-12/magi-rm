#!/usr/bin/env python3
"""
fix_step_score_mismatch.py
Auto-corrects step/score length mismatches by padding or trimming
to ensure consistent schema for RM (PRM + ORM) training.
"""

import json
from pathlib import Path
from statistics import mean

INPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_fixed_final.jsonl"

fixed, total, padded, trimmed = 0, 0, 0, 0
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        rec = json.loads(line)
        for chain in rec.get("generated", []):
            steps = chain.get("steps", [])
            scores = chain.get("step_scores_raw", [])
            n_s, n_r = len(steps), len(scores)
            if abs(n_s - n_r) > 0:
                fixed += 1
                if n_r < n_s:
                    # pad with mean or last score
                    pad_val = mean(scores) if scores else 5.0
                    scores += [pad_val] * (n_s - n_r)
                    padded += 1
                elif n_r > n_s:
                    # trim extra
                    scores = scores[:n_s]
                    trimmed += 1
                # recompute normalized + aggregates
                chain["step_scores_raw"] = scores
                chain["step_scores_norm"] = [s / 10.0 for s in scores]
                chain["prm_score_mean"] = sum(chain["step_scores_norm"]) / len(chain["step_scores_norm"])
                prod = 1.0
                for s in chain["step_scores_norm"]:
                    prod *= s
                chain["prm_score_prod"] = prod
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += 1

print(f"Total records processed: {total}")
print(f"Chains fixed: {fixed} (padded={padded}, trimmed={trimmed})")
print(f"Output written to: {OUTPUT_FILE}")
