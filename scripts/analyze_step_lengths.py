# # quick snippet
# import json, statistics
# from collections import Counter
# lens=[]
# p="data/collected/gsm8k_training_chains_1000_llm_fixed.jsonl"
# for line in open(p):
#     rec=json.loads(line)
#     for c in rec["generated"]:
#         lens.append(len(c["steps"]))
# print("mean",statistics.mean(lens),"median",statistics.median(lens))
# print(Counter(lens).most_common(20))

#!/usr/bin/env python3
"""
analyze_step_lengths.py

Analyze step counts per reasoning chain and flag extremes for filtering.
"""

import json
from pathlib import Path
from statistics import mean, stdev
from collections import Counter

INPUT_FILE  = "data/collected/gsm8k_training_chains_1000_norm.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_norm_analyzed.jsonl"

MIN_STEPS, MAX_STEPS = 2, 12  # tune if needed

def main():
    in_path, out_path = Path(INPUT_FILE), Path(OUTPUT_FILE)
    records = [json.loads(l) for l in in_path.open("r", encoding="utf-8") if l.strip()]
    step_counts = []
    for rec in records:
        for c in rec.get("generated", []):
            n = len(c.get("steps", []))
            c["num_steps"] = n
            c["is_noisy"] = (n < MIN_STEPS or n > MAX_STEPS)
            if c.get("steps", []):
                redundant_checks = sum(1 for s in c["steps"] if "check" in s.lower())
                c["is_noisy"] = c["is_noisy"] or (redundant_checks > 4)
            step_counts.append(n)
    # stats
    mean_steps = mean(step_counts)
    sd_steps = stdev(step_counts) if len(step_counts) > 1 else 0
    hist = Counter(step_counts)
    print(f"Total chains: {len(step_counts)}")
    print(f"Mean steps: {mean_steps:.2f} ± {sd_steps:.2f}")
    print(f"Outliers (<{MIN_STEPS} or >{MAX_STEPS}): {sum(c['is_noisy'] for r in records for c in r['generated'])}")
    print("\nHistogram (first 15 bins):")
    for k in sorted(hist)[:15]:
        print(f"  {k}: {hist[k]}")
    # write updated file
    with out_path.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] Wrote analyzed dataset → {out_path}")

if __name__ == "__main__":
    main()

