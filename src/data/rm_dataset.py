"""
src/data/rm_dataset.py

RMDataset + RMDataCollator:
 - Supports ORM (binary) and PRM (variable-length step targets) training.
 - Works with HuggingFace tokenizer API.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import math

class RMDataset(Dataset):
    """
    Expect each line in JSONL to be:
      {
        "qid": "...", "chain_id": ..., "prompt": "<text>", "label": 0/1, "step_targets": [0.0..1.0], "prm_mean": 0.0,
        "meta": {...}
      }
    For ORM training feed `label`. For PRM training use `step_targets`.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf8") as f:
            self.data = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.data[idx]
        return rec


class RMDataCollator:
    """
    Collate: tokenize prompt -> pad -> create:
      input_ids, attention_mask,
      orm_labels (LongTensor) (optional),
      prm_step_targets (FloatTensor padded), step_mask (BoolTensor)
      prm_mean_targets (FloatTensor) (optional)
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]):
        prompts = [b["prompt"] for b in batch]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, return_token_type_ids=False)
        # ORM labels
        orm_labels = None
        if "label" in batch[0]:
            orm_labels = torch.tensor([int(b.get("label", 0)) for b in batch], dtype=torch.long)
        # PRM step targets (pad to max steps in batch)
        steps_list = [b.get("step_targets", []) for b in batch]
        max_steps = max(len(s) for s in steps_list) if steps_list else 0
        if max_steps == 0:
            step_targets = torch.zeros((len(batch), 0), dtype=torch.float32)
            step_mask = torch.zeros((len(batch), 0), dtype=torch.bool)
        else:
            step_targets = torch.zeros((len(batch), max_steps), dtype=torch.float32)
            step_mask = torch.zeros((len(batch), max_steps), dtype=torch.bool)
            for i, s in enumerate(steps_list):
                for j, val in enumerate(s):
                    step_targets[i, j] = float(val)
                    step_mask[i, j] = True
        # prm_mean target
        prm_mean = None
        if "prm_mean" in batch[0]:
            prm_mean = torch.tensor([float(b.get("prm_mean", 0.0)) for b in batch], dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "orm_labels": orm_labels,
            "step_targets": step_targets,
            "step_mask": step_mask,
            "prm_mean": prm_mean,
            "meta": [b.get("meta", {}) for b in batch],
            "qid": [b.get("qid") for b in batch],
            "chain_id": [b.get("chain_id") for b in batch]
        }


def get_tokenizer(tokenizer_name: str, use_fast: bool = True):
    """
    Safe tokenizer loader for models like Galactica/LLaMA without special tokens.
    Ensures eos, bos, and pad tokens are defined for batching and losses masking.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast, truncation_side="right")

    # ✅ Fix for legacy Galactica/LLaMA tokenizers missing EOS/BOS/PAD
    if tokenizer.eos_token is None:
        print("[FIX] No eos_token found — adding '</s>' as eos_token.")
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.bos_token is None:
        print("[FIX] No bos_token found — adding '<s>' as bos_token.")
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.pad_token is None:
        print("[FIX] No pad_token found — using eos_token as pad_token (safe for decoder-only).")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set for causal LM best practices in RM training/validation
    tokenizer.padding_side = "left"  # Right-align sequences for efficient causal attention
    tokenizer.truncation_side = "left"  # Preserve end of sequence (critical for step_targets/labels)

    print(f"[VERIFY] eos_token={tokenizer.eos_token}, pad_token={tokenizer.pad_token}, pad_token_id={tokenizer.pad_token_id}")
    return tokenizer
