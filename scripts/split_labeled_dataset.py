#!/usr/bin/env python3
"""
split_labeled_dataset.py
Split labeled RM dataset into train / val / test at question-level.
"""

import json, random
from pathlib import Path

INPUT = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
random.seed(RANDOM_SEED)

records = [json.loads(l) for l in open(INPUT, "r", encoding="utf-8") if l.strip()]
random.shuffle(records)
n = len(records)
n_train, n_val = int(n*TRAIN_RATIO), int(n*VAL_RATIO)

splits = {
    "train": records[:n_train],
    "val":   records[n_train:n_train+n_val],
    "test":  records[n_train+n_val:]
}

for name, recs in splits.items():
    with open(OUT_DIR / f"gsm8k_rm_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{name}] {len(recs)} questions written.")
