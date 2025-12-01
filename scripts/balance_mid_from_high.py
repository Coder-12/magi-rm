#!/usr/bin/env python3
"""
balance_mid_from_high.py

Conservatively rebalance dataset by converting some HIGH PRM chains into MID PRM chains.
- In-place modification (keeps total chains constant).
- Tags modified chains with "augmented_mid_balanced".
- Preserves chain-level ORM labels and final_answer.
- Dry-run supported.

Usage:
  python scripts/balance_mid_from_high.py \
    --input data/splits/gsm8k_rm_train_low_boosted.jsonl \
    --output data/splits/gsm8k_rm_train_FINAL_balanced.jsonl \
    --fraction 0.15 \
    --seed 42 \
"""

import argparse
import json
import random
import math
from statistics import mean
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def save_jsonl(path: str, recs: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chain_mean(chain: Dict[str, Any]) -> float:
    s = chain.get("step_scores_norm", [])
    return float(mean(s)) if s else 0.0

def recompute_chain_aggregates(chain: Dict[str, Any]) -> None:
    s = chain.get("step_scores_norm", []) or []
    # clamp
    s = [min(1.0, max(0.0, float(x))) for x in s]
    chain["step_scores_norm"] = s
    chain["step_scores_raw"] = [round(float(x) * 10.0, 4) for x in s]
    chain["prm_score_mean"] = float(mean(s)) if s else 0.0
    chain["prm_score_prod"] = float(math.prod(s)) if s else 0.0

def distribution_stats(recs: List[Dict[str, Any]], low_cut=0.30, mid_cut=0.70):
    total = 0
    low = mid = high = 0
    for rec in recs:
        for c in rec.get("generated", []):
            m = chain_mean(c)
            total += 1
            if m < low_cut:
                low += 1
            elif m < mid_cut:
                mid += 1
            else:
                high += 1
    return {"total": total, "low": low, "mid": mid, "high": high}

def degrade_one_chain_to_mid(chain: Dict[str, Any], mid_low: float=0.3, mid_high: float=0.7, clip_min=0.02):
    """
    Slightly reduce steps that are > mid_high so that average moves down into mid range.
    Conservative: only reduce values > mid_high by 0.10-0.25.
    """
    s = chain.get("step_scores_norm", []) or []
    if not s:
        return False
    new = list(s)
    # find indices with values > mid_high
    idxs = [i for i,v in enumerate(new) if v > mid_high]
    if not idxs:
        # if no step > mid_high, degrade the largest two steps
        sorted_idx = sorted(range(len(new)), key=lambda i: new[i], reverse=True)
        idxs = sorted_idx[:max(1, min(2, len(sorted_idx)))]
    # Apply conservative reductions
    for i in idxs:
        delta = random.uniform(0.10, 0.25)
        new[i] = max(clip_min, new[i] - delta)
    # If still above mid_high on average, reduce a few more but small
    attempts = 0
    while mean(new) >= mid_high and attempts < 3:
        # lower one of the previously untouched top steps
        top_idx = max(range(len(new)), key=lambda j: new[j])
        new[top_idx] = max(clip_min, new[top_idx] - random.uniform(0.06, 0.12))
        attempts += 1
    chain["step_scores_norm"] = [float(x) for x in new]
    recompute_chain_aggregates(chain)
    chain["labeling_status"] = "augmented_mid_balanced"
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input jsonl")
    parser.add_argument("--output", required=True, help="output jsonl")
    parser.add_argument("--fraction", type=float, default=0.15, help="fraction of HIGH chains to convert to MID (0..1)")
    parser.add_argument("--high_cut", type=float, default=0.70, help="threshold for HIGH bucket (prm mean >= high_cut)")
    parser.add_argument("--mid_low", type=float, default=0.30, help="lower bound of MID bucket")
    parser.add_argument("--mid_high", type=float, default=0.70, help="upper bound of MID bucket")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    random_seed = args.seed
    random = __import__("random")
    random.seed(random_seed)

    recs = load_jsonl(args.input)
    stats_before = distribution_stats(recs, low_cut=args.mid_low, mid_cut=args.mid_high)
    total = stats_before["total"]
    high_count = stats_before["high"]
    to_convert = int(round(high_count * args.fraction))
    print(f"Loaded {len(recs)} questions; total chains={total}; high_count={high_count}; converting target={to_convert}")

    # Collect references to high chains (rec_obj, chain_obj)
    high_refs = []
    for rec in recs:
        for c in rec.get("generated", []):
            if chain_mean(c) >= args.high_cut:
                high_refs.append((rec, c))

    if to_convert <= 0 or not high_refs:
        print("Nothing to convert; exiting.")
        if not args.dry_run:
            save_jsonl(args.output, recs)
        return

    # Choose random subset of high_refs to modify
    random.shuffle(high_refs)
    selected = high_refs[:to_convert]
    converted = 0
    for rec, chain in selected:
        ok = degrade_one_chain_to_mid(chain, mid_low=args.mid_low, mid_high=args.mid_high)
        if ok:
            converted += 1

    stats_after = distribution_stats(recs, low_cut=args.mid_low, mid_cut=args.mid_high)
    print("Converted:", converted)
    print("Before: total={total}, low={low}, mid={mid}, high={high}".format(**stats_before))
    print("After : total={total}, low={low}, mid={mid}, high={high}".format(**stats_after))
    print("Before pct: low={:.3f} mid={:.3f} high={:.3f}".format(
        stats_before["low"]/stats_before["total"],
        stats_before["mid"]/stats_before["total"],
        stats_before["high"]/stats_before["total"],
    ))
    print("After  pct: low={:.3f} mid={:.3f} high={:.3f}".format(
        stats_after["low"]/stats_after["total"],
        stats_after["mid"]/stats_after["total"],
        stats_after["high"]/stats_after["total"],
    ))

    if args.dry_run:
        print("Dry-run: not writing output.")
        return

    save_jsonl(args.output, recs)
    print("Saved balanced dataset to:", args.output)

if __name__ == "__main__":
    main()
