# src/data_utils.py
"""
Data utilities for Reward Model training (ORM + PRM).

- ORMDataset: each example = (text: question + full chain) -> chain_label (0/1)
- PRMDataset: each example = (text: question + single step context) -> step_score (0..1)
- Collator: tokenizes (using provided tokenizer), pads, returns tensors.

Designed to allow small CPU tests and scale to accelerate multi-GPU runs.
"""

from typing import List, Dict, Any, Optional, Iterable
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


def _ensure_text(x) -> str:
    if isinstance(x, str):
        return x
    return str(x)


class ORMChainExample:
    def __init__(self, qid: str, question: str, chain_text: str, chain_label: int):
        self.qid = qid
        self.question = question
        self.chain_text = chain_text
        self.chain_label = int(chain_label)


class PRMStepExample:
    def __init__(self, qid: str, question: str, step_text: str, step_score: float, step_index: int, chain_id: int):
        self.qid = qid
        self.question = question
        self.step_text = step_text
        self.step_score = float(step_score)  # normalized 0..1
        self.step_index = int(step_index)
        self.chain_id = int(chain_id)


class ORMJsonlDataset(Dataset):
    """
    Load JSONL where each record corresponds to a question with 'generated' list of chains.
    Each chain must have: 'raw_text' or 'steps', and 'chain_label' (0/1).
    Produces ORMChainExample entries of length = number of chains.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        self.examples: List[ORMChainExample] = []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for rec in map(json.loads, f):
                qid = rec.get("id", "")
                question = rec.get("question", "")
                for chain in rec.get("generated", []):
                    chain_text = chain.get("raw_text") or "\n".join(chain.get("steps", []))
                    label = int(chain.get("chain_label", 0))
                    self.examples.append(ORMChainExample(qid, question, chain_text, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> ORMChainExample:
        return self.examples[idx]


class PRMJsonlDataset(Dataset):
    """
    Build PRM dataset by extracting per-step training examples.
    Each chain must have 'step_scores_raw' (0..10) or 'step_scores_norm' (0..1). If only raw (0..10),
    convert to norm by dividing by 10.
    """
    def __init__(self, path: str, convert_raw_to_norm: bool = True, min_steps: int = 1):
        self.path = Path(path)
        self.examples: List[PRMStepExample] = []
        self._load(convert_raw_to_norm, min_steps)

    def _load(self, convert_raw_to_norm: bool, min_steps: int):
        with open(self.path, "r", encoding="utf-8") as f:
            for rec in map(json.loads, f):
                qid = rec.get("id", "")
                question = rec.get("question", "")
                for chain_idx, chain in enumerate(rec.get("generated", [])):
                    steps = chain.get("steps", [])
                    # Get per-step scores: prefer step_scores_norm, else step_scores_raw/10
                    if "step_scores_norm" in chain and chain["step_scores_norm"]:
                        scores_norm = list(chain["step_scores_norm"])
                    elif "step_scores_raw" in chain and chain["step_scores_raw"]:
                        scores_norm = [float(s) / 10.0 for s in chain.get("step_scores_raw", [])]
                    else:
                        # skip chains with no step labels
                        continue
                    # If lengths mismatch, try to align min length
                    n = min(len(steps), len(scores_norm))
                    if n < min_steps:
                        continue
                    for i in range(n):
                        step_text = _ensure_text(steps[i])
                        step_score = float(scores_norm[i])
                        self.examples.append(PRMStepExample(qid, question, step_text, step_score, i, chain_idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> PRMStepExample:
        return self.examples[idx]


class RMDataCollator:
    """
    Tokenize text inputs: combines question + separator + content.
    For ORM: content = full chain text. Label is chain_label (0/1).
    For PRM: content = one step text (optionally include question context).
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512, include_question: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_question = include_question

    def _build_input(self, question: Optional[str], content: str) -> str:
        if self.include_question and question:
            return f"Question: {question}\nSolution:\n{content}\n"
        else:
            return content

    def collate_orm(self, batch: List[ORMChainExample]) -> Dict[str, torch.Tensor]:
        texts = [self._build_input(b.question, b.chain_text) for b in batch]
        toks = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([b.chain_label for b in batch], dtype=torch.float32)
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": labels}

    def collate_prm(self, batch: List[PRMStepExample]) -> Dict[str, torch.Tensor]:
        # For PRM we include question + step_text as model input and step_score as target (0..1)
        texts = [self._build_input(b.question, b.step_text) for b in batch]
        toks = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([b.step_score for b in batch], dtype=torch.float32)
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": labels}
