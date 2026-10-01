#!/usr/bin/env python3
"""
check_normalization_integrity.py
──────────────────────────────────────────────
Stage 3.2.5 — Validation of numeric integrity for ORM + PRM splits.

Checks:
  • PRM step_targets ∈ [0, 1]
  • ORM labels ∈ {0, 1}
  • Detect NaNs / out-of-range values
  • Optionally auto-clip PRM values to [0, 1]
  • Generate summary JSON for reproducibility
"""

import argparse
import json
import numpy as np
from pathlib import Path
from loguru import logger


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def check_prm(prm_path: str, auto_clip=False):
    data = load_jsonl(prm_path)
    all_scores = []
    bad_count = 0

    for d in data:
        scores = np.array(d.get("step_targets", []), dtype=float)
        if scores.size == 0:
            continue

        # Detect invalids
        mask_bad = np.isnan(scores) | (scores < 0.0) | (scores > 1.0)
        num_bad = int(mask_bad.sum())
        if num_bad > 0:
            bad_count += num_bad
            if auto_clip:
                scores = np.clip(scores, 0.0, 1.0)
                d["step_targets"] = scores.tolist()

        all_scores.extend(scores.tolist())

    stats = {
        "examples": len(data),
        "total_values": len(all_scores),
        "mean": float(np.mean(all_scores)) if all_scores else None,
        "stdev": float(np.std(all_scores)) if all_scores else None,
        "min": float(np.min(all_scores)) if all_scores else None,
        "max": float(np.max(all_scores)) if all_scores else None,
        "out_of_range": int(bad_count),
    }

    if auto_clip and bad_count > 0:
        backup = Path(prm_path).with_suffix(".bak.jsonl")
        Path(prm_path).rename(backup)
        with open(prm_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        logger.warning(f"[AUTO-FIX] Clipped {bad_count} out-of-range PRM scores → wrote back to {prm_path}")
    return stats


def check_orm(orm_path: str):
    data = load_jsonl(orm_path)
    labels = []
    bad = 0

    for d in data:
        label = d.get("label", None)
        if label is None:
            bad += 1
            continue
        if label not in [0, 1]:
            bad += 1
        labels.append(label)

    stats = {
        "examples": len(data),
        "label_1_ratio": float(np.mean(labels)) if labels else 0,
        "invalid_labels": bad,
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_prm", required=True)
    parser.add_argument("--val_prm", required=True)
    parser.add_argument("--test_prm", required=True)
    parser.add_argument("--train_orm", required=True)
    parser.add_argument("--val_orm", required=True)
    parser.add_argument("--test_orm", required=True)
    parser.add_argument("--auto_clip", action="store_true", help="Clip out-of-range PRM values safely")
    parser.add_argument("--out", default="data/processed/rm_norm_check.json")
    args = parser.parse_args()

    logger.info("Checking normalization integrity for ORM + PRM splits...")
    report = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        prm = getattr(args, f"{split}_prm")
        orm = getattr(args, f"{split}_orm")
        logger.info(f"── {split.upper()} ────────────────────────────────")
        report[split]["PRM"] = check_prm(prm, auto_clip=args.auto_clip)
        report[split]["ORM"] = check_orm(orm)

    # Aggregate metrics
    total_bad = sum(report[s]["PRM"]["out_of_range"] for s in report)
    total_bad += sum(report[s]["ORM"]["invalid_labels"] for s in report)
    report["summary"] = {
        "total_outliers": total_bad,
        "verdict": "✅ PASS" if total_bad == 0 else "⚠️ REVIEW",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.success(f"Normalization integrity report saved to {args.out}")

    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
