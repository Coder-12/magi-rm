#!/usr/bin/env python3
"""
data_utils.py
──────────────────────────────────────────────
Stage 3.2.3 — Dataset Loading & Collation for Reward Models (ORM + PRM)

Implements:
  • get_tokenizer – robust tokenizer init for Galactica/other LLMs.
  • ORMJsonlDataset – for chain-level classification (ORM).
  • PRMJsonlDataset – for step-level score regression (PRM).
  • Collation functions with dynamic padding.

Used by Stage 3.2.4 (`scripts/validate_dataloader_output.py`)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import  logger


# ------------------------------------------------------------
# Tokenizer Loader
# ------------------------------------------------------------
def get_tokenizer(tokenizer_name: str, use_fast: bool = True):
    """
    Safe tokenizer loader for models like Galactica/LLaMA without special tokens.
    Ensures eos, bos, and pad tokens are defined for batching and losses masking.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast, truncation_side="right")

    # ✅ Fix for legacy Galactica/LLaMA tokenizers missing EOS/BOS/PAD
    if tokenizer.eos_token is None:
        logger.info("[FIX] No eos_token found — adding '</s>' as eos_token.")
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.bos_token is None:
        logger.info("[FIX] No bos_token found — adding '<s>' as bos_token.")
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.pad_token is None:
        logger.info("[FIX] No pad_token found — using eos_token as pad_token (safe for decoder-only).")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set for causal LM best practices in RM training/validation
    tokenizer.padding_side = "left"  # Right-align sequences for efficient causal attention
    tokenizer.truncation_side = "left"  # Preserve end of sequence (critical for step_targets/labels)

    logger.info(f"[VERIFY] eos_token={tokenizer.eos_token}, pad_token={tokenizer.pad_token}, pad_token_id={tokenizer.pad_token_id}")
    return tokenizer


# ------------------------------------------------------------
# Base Dataset Loader
# ------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path = Path(path)
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ------------------------------------------------------------
# ORM Dataset – binary classification at chain level
# ------------------------------------------------------------
class ORMJsonlDataset(Dataset):
    """
    Each item contains:
       input_ids, attention_mask, label ∈ {0, 1}, meta (dict)
    """

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.samples = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ex = self.samples[idx]
        prompt = ex.get("prompt", "")
        label = int(ex.get("label", 0))
        meta = ex.get("meta", {})

        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
            "meta": meta,
        }


# ------------------------------------------------------------
# PRM Dataset – step-level score regression
# ------------------------------------------------------------
class PRMJsonlDataset(Dataset):
    """
    Each item contains:
       input_ids, attention_mask, step_targets (float[steps]), meta (dict)
    """

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.samples = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ex = self.samples[idx]
        prompt = ex.get("prompt", "")
        step_targets = ex.get("step_targets", [])
        meta = ex.get("meta", {})

        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        targets = torch.tensor(step_targets, dtype=torch.float)
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "step_targets": targets,
            "meta": meta,
        }


# ------------------------------------------------------------
# Collation Utilities
# ------------------------------------------------------------
def collate_orm(batch: List[Dict[str, Any]], pad_token_id: int):
    """Dynamic padding for ORM batch."""
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "meta": [b["meta"] for b in batch],
    }


def collate_prm(batch: List[Dict[str, Any]], pad_token_id: int):
    """Dynamic padding for PRM batch with variable step lengths."""
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    targets = [b["step_targets"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    # pad step targets to same length for batch processing
    max_steps = max(len(t) for t in targets)
    step_targets = torch.zeros(len(batch), max_steps)
    step_mask = torch.zeros_like(step_targets)
    for i, t in enumerate(targets):
        step_targets[i, : len(t)] = t
        step_mask[i, : len(t)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "step_targets": step_targets,
        "step_mask": step_mask,
        "meta": [b["meta"] for b in batch],
    }
