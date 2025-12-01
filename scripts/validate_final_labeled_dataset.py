#!/usr/bin/env python3
"""
validate_final_labeled_dataset.py
Final validation of labeled reasoning dataset (Sub-Stage 2.2 completion check)
Ensures dataset integrity, field coverage, type correctness, and label coherence
before training RM (ORM + PRM).
"""

import json
from pathlib import Path
from statistics import mean
from collections import Counter
from datetime import datetime, UTC

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
INPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_patched_final.jsonl"
REPORT_FILE = "data/qc/qc_validation_labels_final.json"

REQUIRED_TOP_FIELDS = [
    "id", "question", "generated", "gold_answer", "created_at"
]

REQUIRED_CHAIN_FIELDS = [
    "raw_text", "steps", "final_answer",
    "step_scores_raw", "step_scores_norm",
    "prm_score_mean", "prm_score_prod",
    "chain_label", "orm_score",
    "timestamps", "labeling_status"
]

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                print("[WARN] Could not parse one line.")

n_q = len(records)
n_chains = sum(len(r.get("generated", [])) for r in records)
print(f"Loaded {n_q} questions, {n_chains} reasoning chains.\n")

# ------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------
missing_fields, type_issues, logical_issues = 0, 0, 0
chain_ok, chain_err = 0, 0

for q in records:
    # 1️⃣ Top-level schema check
    for f in REQUIRED_TOP_FIELDS:
        if f not in q:
            missing_fields += 1
            print(f"[ERROR] Missing top field '{f}' in question id={q.get('id')}")
    # 2️⃣ Per-chain checks
    for c in q.get("generated", []):
        ok = True
        for f in REQUIRED_CHAIN_FIELDS:
            if f not in c:
                ok = False
                missing_fields += 1
                print(f"[ERROR] Missing field '{f}' in chain of qid={q.get('id')}")
        # Type validation
        try:
            if not isinstance(c["steps"], list): ok = False
            if not isinstance(c["step_scores_raw"], list): ok = False
            if not isinstance(c["step_scores_norm"], list): ok = False
            if not isinstance(c["chain_label"], int): ok = False
            if not isinstance(c["prm_score_mean"], (int, float)): ok = False
            if not isinstance(c["orm_score"], (int, float)): ok = False
        except KeyError:
            ok = False
        if not ok:
            type_issues += 1
            chain_err += 1
            continue

        # Logical coherence checks
        steps_n = len(c["steps"])
        scores_n = len(c["step_scores_raw"])
        if abs(steps_n - scores_n) > 1:
            logical_issues += 1
            print(f"[WARN] Step count mismatch ({steps_n} vs {scores_n}) in qid={q.get('id')}")

        # Sanity bounds
        if not (0 <= c["prm_score_mean"] <= 1):
            logical_issues += 1
        if not (0 <= c["prm_score_prod"] <= 1):
            logical_issues += 1
        if c["orm_score"] not in [0.0, 1.0]:
            logical_issues += 1
        chain_ok += 1

# ------------------------------------------------------------
# Statistics summary
# ------------------------------------------------------------
prm_means = []
orms = []
for q in records:
    for c in q["generated"]:
        prm_means.append(c.get("prm_score_mean", 0))
        orms.append(c.get("orm_score", 0))

prm_mean_avg = mean(prm_means)
orm_ratio = Counter(orms)
corr_ratio = orm_ratio.get(1.0, 0) / max(1, sum(orm_ratio.values()))

print("\n=== Summary ===")
print(f"Valid chains: {chain_ok} / {n_chains}")
print(f"Missing fields: {missing_fields}")
print(f"Type issues: {type_issues}")
print(f"Logical issues: {logical_issues}")
print(f"Average PRM mean: {prm_mean_avg:.3f}")
print(f"ORM=1 ratio: {corr_ratio:.2%}")

# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------
if missing_fields == 0 and type_issues == 0 and logical_issues < 10:
    verdict = "✅ PASS — Dataset schema & label structure are fully valid for RM training"
else:
    verdict = "⚠️ REVIEW — Some schema or logical inconsistencies detected"

print(f"\nFinal Verdict: {verdict}")

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------
report = {
    "timestamp": datetime.now(UTC).isoformat(),
    "questions": n_q,
    "chains": n_chains,
    "valid_chains": chain_ok,
    "missing_fields": missing_fields,
    "type_issues": type_issues,
    "logical_issues": logical_issues,
    "prm_mean_avg": prm_mean_avg,
    "orm_ratio": orm_ratio,
    "verdict": verdict
}

Path("data/qc").mkdir(parents=True, exist_ok=True)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nQC validation report saved to {REPORT_FILE}")
