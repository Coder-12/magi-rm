#!/usr/bin/env python3
"""
scripts/validate_rm_loading.py

Quick checks:
 - load a few examples
 - tokenize and collate
 - print shapes and example entries
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from pathlib import Path
from src.data.rm_dataset import RMDataset, RMDataCollator, get_tokenizer
from torch.utils.data import DataLoader
from loguru import logger

def main(args):
    tokenizer = get_tokenizer(args.tokenizer_name, use_fast=True)
    ds = RMDataset(args.input)
    logger.info(f"Loaded {len(ds)} examples from {args.input}")
    collator = RMDataCollator(tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)
    batch = next(iter(dl))
    logger.info("Batch keys:", list(batch.keys()))
    logger.info("input_ids.shape:", batch["input_ids"].shape)
    logger.info("attention_mask.shape:", batch["attention_mask"].shape)
    if batch["orm_labels"] is not None:
        logger.info("orm_labels.shape:", batch["orm_labels"].shape, "examples:", batch["orm_labels"][:8].tolist())
    logger.info("step_targets.shape:", batch["step_targets"].shape)
    logger.info("step_mask.sum per row:", batch["step_mask"].sum(dim=1).tolist())
    logger.info("prm_mean:", batch["prm_mean"][:8] if batch["prm_mean"] is not None else None)
    logger.info("meta sample:", batch["meta"][0])
    logger.info("qid sample:", batch["qid"][0])
    logger.info("Sample prompt (truncated):")
    # logger.info(tokenizer.decode(batch["input_ids"][0][:200], skip_special_tokens=True)[:800])
    logger.info(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)[:1500])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/prm_train.jsonl")
    p.add_argument("--tokenizer_name", type=str, default="facebook/galactica-6.7b")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    args = p.parse_args()
    main(args)
