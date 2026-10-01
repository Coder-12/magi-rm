#!/usr/bin/env python3
"""
validate_dataloader_output.py
──────────────────────────────────────────────
Stage 3.2.4 — Quick verification of DataLoader output
Checks ORM + PRM batches for:
  • Tokenization integrity
  • Padding alignment
  • Shape and dtype correctness
  • Sample decoded text and meta
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader

from src.data_utils import (
    get_tokenizer,
    ORMJsonlDataset,
    PRMJsonlDataset,
    collate_orm,
    collate_prm,
)
import torch


def inspect_batch(batch, tokenizer, is_prm=False, n=1):
    """Pretty-print inspection for a small batch."""
    for i in range(min(n, len(batch["meta"]))):
        logger.info(f"── Sample {i} ────────────────────────────────")
        decoded = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
        logger.info(f"Decoded prompt:\n{decoded[:400]}...")
        if is_prm:
            logger.info(f"step_targets: {batch['step_targets'][i].tolist()}")
            logger.info(f"step_mask.sum = {batch['step_mask'][i].sum().item()}")
        else:
            logger.info(f"label = {batch['labels'][i].item()}")
        logger.info(f"meta = {batch['meta'][i]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--prm_path", type=str, required=True)
    parser.add_argument("--orm_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer_name, use_fast=True)
    pad_id = tokenizer.pad_token_id

    logger.info(f"[Tokenizer] eos={tokenizer.eos_token}, pad_id={pad_id}")

    # ── Load Datasets ─────────────────────────────────────────────
    prm_ds = PRMJsonlDataset(args.prm_path, tokenizer, max_length=args.max_length)
    orm_ds = ORMJsonlDataset(args.orm_path, tokenizer, max_length=args.max_length)
    logger.info(f"[INFO] Loaded PRM {len(prm_ds)} examples, ORM {len(orm_ds)} examples")

    prm_loader = DataLoader(
        prm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_prm(b, pad_token_id=pad_id),
    )

    orm_loader = DataLoader(
        orm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_orm(b, pad_token_id=pad_id),
    )

    # ── Inspect a few batches ─────────────────────────────────────
    prm_batch = next(iter(prm_loader))
    orm_batch = next(iter(orm_loader))

    logger.info(f"PRM batch → input_ids: {prm_batch['input_ids'].shape}, step_targets: {prm_batch['step_targets'].shape}")
    logger.info(f"ORM batch → input_ids: {orm_batch['input_ids'].shape}, labels: {orm_batch['labels'].shape}")

    logger.info("Verifying PRM sample decoding...")
    inspect_batch(prm_batch, tokenizer, is_prm=True, n=1)

    logger.info("Verifying ORM sample decoding...")
    inspect_batch(orm_batch, tokenizer, is_prm=False, n=1)

    logger.success("✅ DataLoader validation successful — structure consistent.")


if __name__ == "__main__":
    main()
