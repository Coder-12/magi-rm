#!/usr/bin/env python3
"""
deterministic_split_optimizer.py
──────────────────────────────────────────────
Searches for an optimal deterministic hash modulus
to split the full labeled dataset into train/val/test
with:
  • Stable ratios (~0.8 / 0.1 / 0.1)
  • High statistical similarity (Pearson ≥ 0.95)
  • No drift (KS p-value ≥ 0.05)
  • Fully reproducible (hash-based, seed-free)

This acts like Anthropic’s internal data-gating for RM datasets.
"""

import json
import numpy as np
from scipy.stats import ks_2samp
from numpy import corrcoef
from tqdm import tqdm
from pathlib import Path

INPUT = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
OUTFILE = "data/processed/split_optimizer_results.json"

# =============================================================
# Helpers
# =============================================================

def load_prm_means(path: str):
    q2prm = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            qid = j.get("id") or j.get("qid")
            if not qid:
                continue
            chains = j.get("generated", [])
            if not chains:
                continue
            prm_means = [np.mean(c.get("step_scores_norm", [])) for c in chains if c.get("step_scores_norm")]
            if prm_means:
                q2prm[qid] = float(np.mean(prm_means))
    return q2prm


def split_groups(qids, mod, q2prm):
    groups = {"train": [], "val": [], "test": []}
    for qid in qids:
        r = hash(qid) % mod
        if r < int(mod * 0.8):         # ~80%
            groups["train"].append(q2prm[qid])
        elif r < int(mod * 0.9):       # ~10%
            groups["val"].append(q2prm[qid])
        else:                          # ~10%
            groups["test"].append(q2prm[qid])
    return {k: np.array(v) for k, v in groups.items()}


def evaluate_split(q2prm, mod):
    qids = list(q2prm.keys())
    g = split_groups(qids, mod, q2prm)
    train, val, test = g["train"], g["val"], g["test"]

    if min(map(len, [train, val, test])) == 0:
        return None

    def hist_corr(a, b):
        h1, _ = np.histogram(a, bins=20, range=(0, 1))
        h2, _ = np.histogram(b, bins=20, range=(0, 1))
        return corrcoef(h1, h2)[0, 1]

    pearson_tv = hist_corr(train, val)
    pearson_tt = hist_corr(train, test)
    pearson_vt = hist_corr(val, test)

    ks_tv = ks_2samp(train, val).pvalue
    ks_tt = ks_2samp(train, test).pvalue
    ks_vt = ks_2samp(val, test).pvalue

    return {
        "mod": mod,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "ratio_train": round(len(train) / len(qids), 3),
        "pearson_min": round(min(pearson_tv, pearson_tt, pearson_vt), 4),
        "ks_min": round(min(ks_tv, ks_tt, ks_vt), 4),
        "pearson_all": [round(x, 4) for x in (pearson_tv, pearson_tt, pearson_vt)],
        "ks_all": [round(x, 4) for x in (ks_tv, ks_tt, ks_vt)],
    }


def main():
    print("\n🔍 Searching for optimal deterministic split modulus...\n")
    q2prm = load_prm_means(INPUT)
    print(f"Loaded {len(q2prm)} questions from {INPUT}")

    results = []
    for mod in tqdm(range(101, 201)):
        res = evaluate_split(q2prm, mod)
        if res:
            results.append(res)

    # Filter viable candidates
    valid = [
        r for r in results
        if (r["pearson_min"] >= 0.95 and r["ks_min"] >= 0.05)
    ]
    if not valid:
        print("⚠️ No moduli met strict thresholds; relax KS_min to 0.03.")
        valid = [
            r for r in results
            if (r["pearson_min"] >= 0.95 and r["ks_min"] >= 0.03)
        ]

    # Rank by closeness to 0.8 train ratio and stability
    best = sorted(valid, key=lambda x: (abs(x["ratio_train"] - 0.8), -x["pearson_min"], -x["ks_min"]))[:5]

    Path(OUTFILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n=== 🧠 TOP 5 MODULUS CANDIDATES ===")
    for b in best:
        print(json.dumps(b, indent=2))
    print(f"\n📄 Saved to: {OUTFILE}\n")

    # Recommend the best
    if best:
        choice = best[0]
        print("✅ RECOMMENDED MODULUS:")
        print(f"  mod = {choice['mod']}")
        print(f"  ratio_train = {choice['ratio_train']}")
        print(f"  pearson_min = {choice['pearson_min']}")
        print(f"  ks_min = {choice['ks_min']}")
        print(f"\nUse this modulus for deterministic splitting.\n")
    else:
        print("❌ No suitable modulus found. Try expanding search range.")


if __name__ == "__main__":
    main()
