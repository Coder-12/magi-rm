#!/usr/bin/env python3
"""
check_rm_split_integrity_flat.py — Verify PRM/ORM split integrity for flattened RM datasets.

Checks:
  1. Split overlap (qid uniqueness)
  2. Record counts per split
  3. PRM step_targets stats (mean, stdev, entropy)
  4. ORM label distribution
  5. Cross-split similarity for PRM mean scores
  6. JSON report

Usage:
  python scripts/check_rm_split_integrity.py \
    --train_prm data/processed/prm_train.jsonl \
    --val_prm data/processed/prm_val.jsonl \
    --test_prm data/processed/prm_test.jsonl \
    --train_orm data/processed/orm_train.jsonl \
    --val_orm data/processed/orm_val.jsonl \
    --test_orm data/processed/orm_test.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime, UTC
from scipy.stats import pearsonr, ks_2samp, entropy

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prm_summary(name, data):
    """Summarize PRM flattened dataset."""
    prm_means = [float(np.mean(x["step_targets"])) for x in data if x.get("step_targets")]
    mean_ = float(np.mean(prm_means)) if prm_means else 0.0
    stdev_ = float(np.std(prm_means)) if prm_means else 0.0
    ent_ = float(entropy(np.histogram(prm_means, bins=10, range=(0, 1))[0] + 1e-10))
    print(f"\n=== PRM {name.upper()} ===")
    print(f"Examples: {len(data)}")
    print(f"Mean PRM mean: {mean_:.3f} | Stdev: {stdev_:.3f} | Entropy: {ent_:.3f}")
    return {"num_records": len(data), "mean": mean_, "stdev": stdev_, "entropy": ent_, "means": prm_means}


def orm_summary(name, data):
    """Summarize ORM flattened dataset."""
    labels = [x.get("label", 0) for x in data]
    cnt = Counter(labels)
    ratio = cnt.get(1, 0) / max(1, len(labels))
    print(f"\n=== ORM {name.upper()} ===")
    print(f"Examples: {len(labels)} | Label=1 ratio: {ratio:.3f}")
    return {"num_records": len(labels), "ratio": ratio}


def overlap(ids_a, ids_b):
    return len(set(ids_a) & set(ids_b))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_prm", required=True)
    ap.add_argument("--val_prm", required=True)
    ap.add_argument("--test_prm", required=True)
    ap.add_argument("--train_orm", required=True)
    ap.add_argument("--val_orm", required=True)
    ap.add_argument("--test_orm", required=True)
    ap.add_argument("--out", default="data/processed/rm_split_integrity_report.json")
    args = ap.parse_args()

    # Load
    prm_train = load_jsonl(args.train_prm)
    prm_val = load_jsonl(args.val_prm)
    prm_test = load_jsonl(args.test_prm)
    orm_train = load_jsonl(args.train_orm)
    orm_val = load_jsonl(args.val_orm)
    orm_test = load_jsonl(args.test_orm)

    # Extract qids
    get_qids = lambda d: [x.get("qid") for x in d if "qid" in x]
    qt, qv, qte = get_qids(prm_train), get_qids(prm_val), get_qids(prm_test)

    # Overlaps
    o_tv, o_tt, o_vt = overlap(qt, qv), overlap(qt, qte), overlap(qv, qte)
    print("\n=== SPLIT OVERLAP ===")
    print(f"Train∩Val: {o_tv}, Train∩Test: {o_tt}, Val∩Test: {o_vt}")
    overlap_exists = any([o_tv, o_tt, o_vt])

    # Summaries
    prm_splits = {
        "train": prm_summary("train", prm_train),
        "val": prm_summary("val", prm_val),
        "test": prm_summary("test", prm_test),
    }
    orm_splits = {
        "train": orm_summary("train", orm_train),
        "val": orm_summary("val", orm_val),
        "test": orm_summary("test", orm_test),
    }

    # Correlation and KS stats for PRM means
    def compare(a, b):
        """Compare two unpaired distributions of PRM mean scores."""
        if not a or not b:
            return {"pearson": None, "ks_pvalue": None}

        try:
            # Normalize to same-size histograms for comparable shapes
            hist_a, _ = np.histogram(a, bins=20, range=(0, 1), density=True)
            hist_b, _ = np.histogram(b, bins=20, range=(0, 1), density=True)
            # Use Pearson between histogram densities (now same shape)
            pearson_val = round(float(np.corrcoef(hist_a, hist_b)[0, 1]), 4)
            ks_val = round(float(ks_2samp(a, b).pvalue), 4)
            return {"pearson": pearson_val, "ks_pvalue": ks_val}
        except Exception as e:
            print(f"[WARN] Distribution compare failed: {e}")
            return {"pearson": None, "ks_pvalue": None}

    sim_stats = {
        "train_val": compare(prm_splits["train"]["means"], prm_splits["val"]["means"]),
        "train_test": compare(prm_splits["train"]["means"], prm_splits["test"]["means"]),
        "val_test": compare(prm_splits["val"]["means"], prm_splits["test"]["means"]),
    }

    # Verdict logic
    def ok(stat_value, threshold=0.9):
        return (stat_value is None) or (stat_value > threshold)

    verdict = (
        (not overlap_exists)
        and all(ok(v.get("pearson")) for v in sim_stats.values())
        and all(v.get("ks_pvalue", 1.0) > 0.05 for v in sim_stats.values())
    )

    print("\n=== CROSS-SPLIT PRM SIMILARITY ===")
    for k, v in sim_stats.items():
        print(f"{k}: Pearson={v['pearson']}, KS_pvalue={v['ks_pvalue']}")

    if verdict:
        print("\n✅ FINAL VERDICT: Split integrity PASS — distributions consistent & non-overlapping.")
    else:
        print("\n⚠️ FINAL VERDICT: REVIEW — check distribution drift or overlap.")

    # Save report
    for s in prm_splits.values():
        s.pop("means", None)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "overlap": {"train_val": o_tv, "train_test": o_tt, "val_test": o_vt},
        "prm": prm_splits,
        "orm": orm_splits,
        "similarity_stats": sim_stats,
        "integrity_pass": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved to {args.out}\n")


if __name__ == "__main__":
    main()
