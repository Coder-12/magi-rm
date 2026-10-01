#!/usr/bin/env python3
"""
scripts/merge_batches.py

Merge multiple GSM8K reasoning chain batch files (JSONL) into a single dataset.
Ensures schema consistency, duplicate avoidance, and clean output.
"""

import os
import json
import argparse
from pathlib import Path

def merge_jsonl_files(input_files, output_file):
    seen_ids = set()
    total_records = 0
    merged = []

    for fpath in input_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping invalid JSON in {fpath}")
                    continue
                rid = rec.get("id")
                if rid in seen_ids:
                    print(f"[WARN] Duplicate ID {rid} found in {fpath}, skipping.")
                    continue
                seen_ids.add(rid)
                merged.append(rec)
                total_records += 1
        print(f"[INFO] Merged {fpath}: {len(seen_ids)} total so far")

    with open(output_file, "w", encoding="utf-8") as fout:
        for rec in merged:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Merge complete.")
    print(f"Total unique records: {total_records}")
    print(f"Output written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge batched GSM8K reasoning chain files")
    parser.add_argument("--input_dir", type=str, default="data/collected", help="Directory containing batch JSONL files")
    parser.add_argument("--pattern", type=str, default="gsm8k_training_chains_S", help="Prefix pattern for batch files")
    parser.add_argument("--output_file", type=str, default="data/collected/gsm8k_training_chains_merged_1000.jsonl")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted([
        str(p) for p in input_dir.glob(f"{args.pattern}*.jsonl")
        if p.is_file()
    ])

    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern}*.jsonl in {input_dir}")

    print(f"[INFO] Found {len(files)} batch files:")
    for f in files:
        print(f"  - {f}")

    merge_jsonl_files(files, args.output_file)

if __name__ == "__main__":
    main()
