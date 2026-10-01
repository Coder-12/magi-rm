#!/usr/bin/env python3
"""
scripts/add_gold_answers.py

Adds 'gold_answer' field to each question record in a generated GSM8K reasoning dataset
using the official GSM8K training split.

Safe, fast, and idempotent — preserves all other fields unchanged.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

INPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_final_for_training.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"

# 1. Load official GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")["train"]
gsm_map = {q.strip(): a.strip() for q, a in zip(gsm8k["question"], gsm8k["answer"])}

# 2. Process our generated dataset
added, missing = 0, 0
out_lines = []

with open(INPUT_FILE, "r", encoding="utf8") as fin:
    for line in tqdm(fin, desc="Adding gold answers"):
        rec = json.loads(line)
        q = rec["question"].strip()
        if q in gsm_map:
            rec["gold_answer"] = gsm_map[q]
            added += 1
        else:
            rec["gold_answer"] = None
            missing += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))

# 3. Write new dataset
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf8") as fout:
    fout.write("\n".join(out_lines) + "\n")

print(f"[DONE] Added gold answers for {added}/{added+missing} questions.")
print(f"Output written to: {OUTPUT_FILE}")
