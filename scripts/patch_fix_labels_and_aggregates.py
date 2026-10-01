#!/usr/bin/env python3
"""
patch_fix_labels_and_aggregates.py
- Aligns chain_label with gold_answer (deterministic).
- Recomputes and clamps step_scores_raw -> step_scores_norm.
- Recomputes prm_score_mean and prm_score_prod.
- Writes fixed output file.
"""

import json, math, re
from pathlib import Path
from statistics import mean
from datetime import datetime

INPUT = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_fixed_final.jsonl"
OUTPUT = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_patched_final.jsonl"
REPORT = "data/collected/patch_report_labels.json"

def normalize_answer_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    # remove common wrappers
    s = s.replace("$", "").replace(",", "")
    # strip "Final Answer:" or similar prefix
    s = re.sub(r'(?i)final\s*answer[:\-]?\s*', '', s).strip()
    # extract number if present
    m = re.search(r'[-+]?\d*\.\d+|\d+', s)
    if m:
        val = float(m.group(0))
        # prefer integer when it's integer-like
        return str(int(val)) if abs(val - round(val)) < 1e-9 else str(val)
    # lower-case and normalize white-space
    return re.sub(r'\s+', ' ', s).lower()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def product(iterable):
    p = 1.0
    for v in iterable:
        p *= v
    return p

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
total_q = 0
total_chains = 0
updated_chain_label = 0
fixed_aggregates = 0
clamped_scores = 0

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        total_q += 1
        rec = json.loads(line)
        gold = rec.get("gold_answer", "")
        norm_gold = normalize_answer_text(gold)
        for chain in rec.get("generated", []):
            total_chains += 1

            # 1) Normalize final and compute expected ORM label
            final = chain.get("final_answer", "")
            norm_final = normalize_answer_text(final)
            expected_label = 1 if (norm_final != "" and norm_final == norm_gold) else 0

            # If chain_label doesn't exist or differs, update deterministically
            if chain.get("chain_label", None) != expected_label:
                chain["chain_label"] = int(expected_label)
                chain["orm_score"] = float(expected_label)
                updated_chain_label += 1

            # 2) Ensure step_scores_raw exist and are floats; clamp to [0,10]
            raw_scores = chain.get("step_scores_raw", [])
            # If missing, try to create placeholders (unlikely here)
            if raw_scores is None:
                raw_scores = []

            raw_scores_f = []
            for v in raw_scores:
                fv = safe_float(v, default=None)
                if fv is None:
                    fv = 5.0  # neutral fallback
                # clamp
                fv_clamped = max(0.0, min(10.0, fv))
                if fv_clamped != fv:
                    clamped_scores += 1
                raw_scores_f.append(fv_clamped)

            # If score count mismatches steps count, pad / trim (paper-faithful)
            steps = chain.get("steps", []) or []
            if len(raw_scores_f) < len(steps):
                pad_val = mean(raw_scores_f) if raw_scores_f else 5.0
                raw_scores_f.extend([pad_val] * (len(steps) - len(raw_scores_f)))
            elif len(raw_scores_f) > len(steps):
                raw_scores_f = raw_scores_f[:len(steps)]

            # 3) Recompute normalized and aggregates
            norm_scores = [(s / 10.0) for s in raw_scores_f] if raw_scores_f else []
            prm_mean = float(mean(norm_scores)) if norm_scores else 0.0
            prm_prod = float(product(norm_scores)) if norm_scores else 0.0

            # update chain fields (overwrite stale values)
            chain["step_scores_raw"] = raw_scores_f
            chain["step_scores_norm"] = norm_scores
            chain["prm_score_mean"] = prm_mean
            chain["prm_score_prod"] = prm_prod

            # mark timestamp of fix
            ts = chain.get("timestamps", {})
            ts["patched_at"] = datetime.utcnow().isoformat() + "Z"
            chain["timestamps"] = ts
            fixed_aggregates += 1

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

report = {
    "input": INPUT,
    "output": OUTPUT,
    "total_questions": total_q,
    "total_chains": total_chains,
    "updated_chain_label": updated_chain_label,
    "fixed_aggregates": fixed_aggregates,
    "clamped_scores": clamped_scores,
    "patched_at": datetime.utcnow().isoformat() + "Z"
}
with open(REPORT, "w", encoding="utf-8") as rf:
    json.dump(report, rf, indent=2)
print("Patch complete. Report:", REPORT)
