#!/usr/bin/env python3
"""
qc_labels_v2.py
Sub-Stage 2.3 — Post-label Quality & Sanity Validation for ORM + PRM

Validates the correctness, completeness, and distribution of labels
generated in Sub-stage 2.2 (colab_vllm_labeling_final / gpt4o-mini version).

Checks:
- Presence of ORM and PRM fields
- Step and chain label coverage
- Distribution histograms and correlations
- Mean/variance sanity vs. expectations
- Optional QC summary JSON report
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from statistics import mean, stdev
from datetime import datetime, UTC

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
INPUT_FILE = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
REPORT_FILE = "data/qc/qc_report_labels_v2.json"

# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                records.append(json.loads(line))
            except:
                pass

n_q = len(records)
n_chains = sum(len(r.get("generated", [])) for r in records)
print(f"Loaded {n_q} questions, {n_chains} labeled chains\n")

# ---------------------------------------------------------------
# Collect label stats
# ---------------------------------------------------------------
prm_means, prm_prods, orm_labels = [], [], []
step_counts = []

for q in records:
    for c in q.get("generated", []):
        s_norm = c.get("step_scores_norm", [])
        if s_norm:
            prm_means.append(float(c.get("prm_score_mean", 0.0)))
            prm_prods.append(float(c.get("prm_score_prod", 0.0)))
            step_counts.append(len(s_norm))
        if "chain_label" in c:
            orm_labels.append(int(c["chain_label"]))

# ---------------------------------------------------------------
# Basic coverage
# ---------------------------------------------------------------
missing_chain_labels = sum(1 for q in records for c in q["generated"] if "chain_label" not in c)
missing_step_scores = sum(1 for q in records for c in q["generated"] if not c.get("step_scores_norm"))

print(f"Missing chain labels: {missing_chain_labels}")
print(f"Missing step scores: {missing_step_scores}")

# ---------------------------------------------------------------
# ORM distribution
# ---------------------------------------------------------------
orm_counts = Counter(orm_labels)
total_orm = len(orm_labels)
print("\n=== ORM Label Distribution ===")
for k, v in sorted(orm_counts.items()):
    pct = 100 * v / total_orm if total_orm else 0
    print(f"  Label {k}: {v} ({pct:.2f}%)")

# ---------------------------------------------------------------
# PRM distributions
# ---------------------------------------------------------------
def describe(arr):
    return {
        "mean": mean(arr),
        "stdev": stdev(arr) if len(arr) > 1 else 0.0,
        "min": min(arr),
        "max": max(arr)
    } if arr else {}

desc_mean = describe(prm_means)
desc_prod = describe(prm_prods)

print("\n=== PRM Score Summary ===")
print(f"Mean  (avg of steps): {desc_mean}")
print(f"Product (aggregated): {desc_prod}")

print("\nStep count distribution (first 15 bins):")
step_hist = Counter(step_counts)
for k in sorted(step_hist)[:15]:
    print(f"  {k}: {step_hist[k]}")

# ---------------------------------------------------------------
# Correlation ORM↔PRM
# ---------------------------------------------------------------
if orm_labels and prm_means:
    corr = np.corrcoef(orm_labels, prm_means[:len(orm_labels)])[0, 1]
    print(f"\nCorrelation (ORM vs PRM mean): {corr:.3f}")

# ---------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------
coverage_ok = missing_chain_labels == 0 and missing_step_scores == 0
prm_mean_ok = 0.3 < desc_mean.get("mean", 0) < 0.9
orm_balance_ok = 25 < (100 * orm_counts.get(1, 0) / total_orm) < 85 if total_orm else False
corr_ok = abs(corr) > 0.2

status = all([coverage_ok, prm_mean_ok, orm_balance_ok, corr_ok])
verdict = "✅ PASS — Labeled data ready for RM training" if status else "⚠️ REVIEW — Some metrics outside ideal range"

print(f"\nFinal QC Verdict: {verdict}")

# ---------------------------------------------------------------
# Save report
# ---------------------------------------------------------------
report = {
    "timestamp": datetime.now(UTC).isoformat(),
    "questions": n_q,
    "chains": n_chains,
    "missing_chain_labels": missing_chain_labels,
    "missing_step_scores": missing_step_scores,
    "orm_distribution": orm_counts,
    "prm_mean_summary": desc_mean,
    "prm_prod_summary": desc_prod,
    "corr_orm_prm": corr,
    "verdict": verdict
}

Path("data/qc").mkdir(parents=True, exist_ok=True)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nQC report saved to {REPORT_FILE}")
