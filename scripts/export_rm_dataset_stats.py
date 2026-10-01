#!/usr/bin/env python3
"""
export_rm_dataset_stats.py
──────────────────────────────────────────────
Stage 3.2.6 — Dataset summary manifest for RM training reproducibility.

Merges:
  - Split integrity results (Stage 3.2.2)
  - Normalization integrity (Stage 3.2.5)
  - Basic ORM/PRM stats (counts, ratios, stdevs)

Outputs:
  data/processed/rm_dataset_summary.json
"""

import json
import argparse
from datetime import datetime, UTC
from pathlib import Path
from loguru import logger


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"[WARN] Missing file: {path}")
        return None


def merge_reports(split_integrity_path, norm_check_path, output_path):
    split_report = load_json(split_integrity_path)
    norm_report = load_json(norm_check_path)

    merged = {
        "timestamp": datetime.now(UTC).isoformat(),
        "split_integrity": split_report or {},
        "normalization_check": norm_report or {},
    }

    # Summary rollup
    try:
        def safe_get(d, path, default=None):
            """Safely navigate nested dicts like split_report['prm']['train']['num_records']"""
            for p in path:
                if isinstance(d, dict) and p in d:
                    d = d[p]
                else:
                    return default
            return d

        merged["summary"] = {
            "num_train_examples": safe_get(split_report, ["prm", "train", "num_records"]),
            "num_val_examples": safe_get(split_report, ["prm", "val", "num_records"]),
            "num_test_examples": safe_get(split_report, ["prm", "test", "num_records"]),
            "train_mean_prm": safe_get(norm_report, ["train", "PRM", "mean"]),
            "val_mean_prm": safe_get(norm_report, ["val", "PRM", "mean"]),
            "test_mean_prm": safe_get(norm_report, ["test", "PRM", "mean"]),
            "train_label_ratio": safe_get(norm_report, ["train", "ORM", "label_1_ratio"]),
            "val_label_ratio": safe_get(norm_report, ["val", "ORM", "label_1_ratio"]),
            "test_label_ratio": safe_get(norm_report, ["test", "ORM", "label_1_ratio"]),
        }
    except Exception as e:
        logger.error(f"[ERROR] Failed summary rollup: {e}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    logger.success(f"✅ RM dataset summary manifest saved → {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_integrity", default="data/processed/rm_split_integrity_report.json")
    parser.add_argument("--norm_check", default="data/processed/rm_norm_check.json")
    parser.add_argument("--out", default="data/processed/rm_dataset_summary.json")
    args = parser.parse_args()

    logger.info("Merging reports into unified RM dataset summary...")
    merged = merge_reports(args.split_integrity, args.norm_check, args.out)
    logger.info(json.dumps(merged["summary"], indent=2))


if __name__ == "__main__":
    main()
