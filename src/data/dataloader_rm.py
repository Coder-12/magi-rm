"""
DataLoader for Dual-Head Reward Model (ORM + PRM)
=================================================

This module prepares training and validation batches for the hybrid ORM+PRM system.

Each JSONL record should include:
{
    "input_text": str,
    "orm_label": int,                    # 0 or 1
    "step_targets": [float],             # step-wise golds (0–1 scaled)
    "step_mask": [int],                  # 1 for valid steps
    "step_confidence": [float],          # optional (0–1), same length as step_targets
    "chain_confidence": float,           # optional (0–1)
    "prm_solution_gold": float           # optional aggregated target
}

Output batch (all torch.Tensors):
{
    "input_ids": (B,T),
    "attention_mask": (B,T),
    "step_targets": (B,S),
    "step_mask": (B,S),
    "step_confidence": (B,S),
    "chain_confidence": (B,),
    "orm_label": (B,),
    "prm_solution_gold": (B,)
}
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ==========================================================
# Dataset Definition
# ==========================================================
class RewardDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        max_length: int = 512,
        max_steps: Optional[int] = None,
    ):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

        self.data = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.data.append(json.loads(line))
                except Exception as e:
                    # best-effort: skip bad line but warn
                    print(f"[RewardDataset] Warning: failed to parse line in {path}: {e}")
                    continue
        if len(self.data) == 0:
            raise ValueError(f"No data loaded from {path}")

        # determine maximum number of reasoning steps
        if max_steps is not None:
            self.max_steps = max_steps
        else:
            self.max_steps = max(len(d.get("step_targets", [])) for d in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        item["input_text"] = item.get("input_text", item.get("prompt", ""))

        # -------------------------------
        # 1. Tokenize text
        # -------------------------------
        encoded = self.tokenizer(
            item["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # ensure tensors are contiguous for DDP (minor perf improvement)
        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.contiguous()

        # -------------------------------
        # 2. Step-level fields
        # -------------------------------
        step_targets = torch.tensor(item.get("step_targets", []), dtype=torch.float32)
        step_mask = torch.tensor(item.get("step_mask", [1] * len(step_targets)), dtype=torch.float32)

        # optional confidence
        step_confidence = torch.tensor(
            item.get("step_confidence", [1.0] * len(step_targets)), dtype=torch.float32
        )

        # pad / truncate to max_steps
        S = self.max_steps
        def pad_to_len(t: torch.Tensor, val: float = 0.0):
            if t.numel() >= S:
                return t[:S]
            pad = torch.full((S - t.numel(),), val, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        step_targets = pad_to_len(step_targets, 0.0)
        step_mask = pad_to_len(step_mask, 0.0)
        step_confidence = pad_to_len(step_confidence, 1.0)

        # -------------------------------
        # 3. Chain-level fields
        # -------------------------------
        orm_label = torch.tensor(item.get("orm_label", item.get("label", item.get("meta", {}).get("orm_label", 0))),
                                 dtype=torch.long)
        chain_conf = torch.tensor(item.get("chain_confidence", 1.0), dtype=torch.float32)
        prm_gold = torch.tensor(item.get("prm_solution_gold", 0.0), dtype=torch.float32)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            step_targets=step_targets,
            step_mask=step_mask,
            step_confidence=step_confidence,
            chain_confidence=chain_conf,
            orm_label=orm_label,
            prm_solution_gold=prm_gold,
        )


# ==========================================================
# Collate Function
# ==========================================================
@dataclass
class RewardBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    step_targets: torch.Tensor
    step_mask: torch.Tensor
    step_confidence: torch.Tensor
    chain_confidence: torch.Tensor
    orm_label: torch.Tensor
    prm_solution_gold: torch.Tensor


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that returns a plain dict of tensors.
    This ensures full compatibility with training and evaluation loops
    that expect dict-like batches.
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# ==========================================================
# Loader Factory
# ==========================================================
def create_dataloader(
    path: str,
    tokenizer_name: str,
    batch_size: int,
    max_length: int,
    max_steps: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    dataset = RewardDataset(path, tokenizer_name, max_length, max_steps)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# ==========================================================
# Unified dataset loader for ORM + PRM training
# ==========================================================
from torch.utils.data import ConcatDataset
import os


def load_rm_datasets(data_cfg: Dict[str, Any]):
    """
    Create unified train and validation datasets for dual-head RM training.

    data_cfg example:
    {
        "train_prm_path": "data/processed/confident/prm_train.jsonl",
        "train_orm_path": "data/orm_train.jsonl",
        "val_prm_path": "data/processed/confident/prm_val.jsonl",
        "val_orm_path": "data/orm_val.jsonl",
        "tokenizer_name": "microsoft/deberta-v3-base",
        "max_length": 512,
        "max_steps": 12
    }

    Returns:
        train_dataset, val_dataset : torch.utils.data.Dataset
    """

    tokenizer_name = data_cfg["tokenizer_name"]
    max_length = data_cfg.get("max_length", 512)
    max_steps = data_cfg.get("max_steps", None)

    # -------------------------------
    # Training datasets
    # -------------------------------
    train_datasets = []
    for key in ["train_prm_path", "train_orm_path"]:
        if key in data_cfg and os.path.exists(data_cfg[key]):
            ds = RewardDataset(
                path=data_cfg[key],
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                max_steps=max_steps,
            )
            train_datasets.append(ds)

    if not train_datasets:
        raise ValueError("No training datasets found in config.")

    # Merge PRM + ORM datasets
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

    # -------------------------------
    # Validation datasets
    # -------------------------------
    val_datasets = []
    for key in ["val_prm_path", "val_orm_path"]:
        if key in data_cfg and os.path.exists(data_cfg[key]):
            ds = RewardDataset(
                path=data_cfg[key],
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                max_steps=max_steps,
            )
            val_datasets.append(ds)

    val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

    # Sanity check for consistent max_steps across merged datasets
    if isinstance(train_dataset, ConcatDataset):
        max_steps_list = [ds.max_steps for ds in train_dataset.datasets]
        assert len(set(max_steps_list)) == 1, f"Inconsistent max_steps in train datasets: {max_steps_list}"
    if isinstance(val_dataset, ConcatDataset):
        max_steps_list = [ds.max_steps for ds in val_dataset.datasets]
        assert len(set(max_steps_list)) == 1, f"Inconsistent max_steps in val datasets: {max_steps_list}"

    return train_dataset, val_dataset

