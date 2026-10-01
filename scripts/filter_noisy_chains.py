#!/usr/bin/env python3
"""
filter_noisy_chains.py

Filter reasoning chains by step-length and noise heuristics before RM training.
Keeps all structural metadata, removes noisy or malformed chains.
"""

import json
from pathlib import Path
from statistics import mean, stdev

INPUT_FILE  = "data/collected/gsm8k_training_chains_1000_norm_analyzed.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_clean.jsonl"
MIN_STEPS, MAX_STEPS = 3, 12

def get_final_answer(chain):
    """Return normalized answer if available, else raw final answer."""
    return chain.get("final_answer_norm", chain.get("final_answer", "")).strip()

def main():
    in_path, out_path = Path(INPUT_FILE), Path(OUTPUT_FILE)
    records = [json.loads(l) for l in in_path.open("r", encoding="utf-8") if l.strip()]
    total_chains = sum(len(r["generated"]) for r in records)
    kept, removed = 0, 0

    for rec in records:
        clean_chains = []
        for c in rec["generated"]:
            steps = c.get("steps", [])
            n = len(steps)
            is_noisy = c.get("is_noisy", False) or (n < MIN_STEPS or n > MAX_STEPS)
            if not is_noisy:
                # Ensure every chain has unified final_answer_used
                c["final_answer_used"] = get_final_answer(c)
                clean_chains.append(c)
            else:
                removed += 1
        rec["generated"] = clean_chains
        kept += len(clean_chains)

    # Filter out records with no valid chains left
    records = [r for r in records if r["generated"]]

    with out_path.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    print(f"Total chains before: {total_chains}")
    print(f"Chains removed: {removed} ({removed / total_chains * 100:.2f}%)")
    print(f"Chains kept: {kept}")
    print(f"Final questions retained: {len(records)}")

    # Step stats
    all_steps = [len(c["steps"]) for r in records for c in r["generated"]]
    mean_steps = mean(all_steps)
    sd_steps = stdev(all_steps) if len(all_steps) > 1 else 0
    print(f"Mean steps (cleaned): {mean_steps:.2f} ± {sd_steps:.2f}")
    print(f"[DONE] Wrote clean dataset → {out_path}")

if __name__ == "__main__":
    main()
