#!/usr/bin/env python3
"""
normalize_final_answers.py

Normalize numeric final answers in reasoning chains.
Adds field `final_answer_norm` for consistency before RM labeling.
"""

import json, re
from pathlib import Path
from statistics import mean

INPUT_FILE  = "data/collected/gsm8k_training_chains_1000_llm_fixed.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_norm.jsonl"

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def normalize_numeric(val: str) -> str:
    if not isinstance(val, str):
        return str(val)
    s = val.strip().replace(",", "")
    s = re.sub(r"[^0-9\.\-\+eE]", "", s)
    if not s:
        return ""
    try:
        num = float(s)
        # Convert to int if it's whole
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        else:
            return str(round(num, 6)).rstrip("0").rstrip(".")
    except Exception:
        # fallback: extract any numeric substring
        m = NUM_RE.search(s)
        return m.group(0) if m else s

def main():
    in_path, out_path = Path(INPUT_FILE), Path(OUTPUT_FILE)
    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    total, normalized = 0, 0
    for rec in records:
        for c in rec.get("generated", []):
            fa = c.get("final_answer", "")
            norm = normalize_numeric(fa)
            c["final_answer_norm"] = norm
            total += 1
            if norm and norm != fa:
                normalized += 1
    with out_path.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] Normalized {normalized}/{total} answers → {out_path}")

if __name__ == "__main__":
    main()
