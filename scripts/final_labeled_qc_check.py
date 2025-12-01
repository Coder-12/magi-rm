#!/usr/bin/env python3
"""
final_labeled_qc_check.py
Performs full validation on labeled dataset before RM training.
Checks schema integrity, label distribution, PRM variance, ORM balance, and logical consistency.
"""

import json, math, statistics, re
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

LABELED_FILE = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
SUMMARY_FILE = "data/collected/final_rm_qc_summary.json"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def normalize_num(s):
    if s is None:
        return ""
    s = str(s).strip().replace("$", "").replace(",", "")
    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return str(round(float(m.group(0)), 3)) if m else s.lower()

records = list(load_jsonl(LABELED_FILE))
total_q = len(records)
total_chains = sum(len(r["generated"]) for r in records)
print(f"Loaded {total_q} questions ({total_chains} chains)\n")

missing_fields = 0
orm_ones, orm_zeros = 0, 0
prm_means, prm_prods, prm_stdevs = [], [], []
score_hist = Counter()

for rec in records:
    gold = normalize_num(rec.get("gold_answer", ""))
    for c in rec["generated"]:
        required = ["step_scores_raw", "prm_score_mean", "prm_score_prod", "chain_label"]
        if not all(k in c for k in required):
            missing_fields += 1
            continue
        step_scores = [float(x) for x in c.get("step_scores_raw", [])]
        if not step_scores: continue
        m = sum(step_scores)/len(step_scores)/10
        p = math.prod([s/10 for s in step_scores])
        prm_means.append(m)
        prm_prods.append(p)
        if len(step_scores)>1:
            prm_stdevs.append(statistics.stdev(step_scores)/10)
        for s in step_scores:
            score_hist[int(s)] += 1
        if c.get("chain_label",0)==1: orm_ones+=1
        else: orm_zeros+=1

# Aggregate metrics
prm_var = statistics.pstdev(prm_means) if prm_means else 0.0
orm_ratio = orm_ones/(orm_ones+orm_zeros) if orm_ones+orm_zeros>0 else 0.0
entropy = -sum((v/sum(score_hist.values()))*math.log(v/sum(score_hist.values())+1e-9)
               for v in score_hist.values())

summary = {
    "total_questions": total_q,
    "total_chains": total_chains,
    "missing_fields": missing_fields,
    "ORM_1_ratio": round(orm_ratio,3),
    "PRM_mean_avg": round(statistics.mean(prm_means),3),
    "PRM_stdev_avg": round(statistics.mean(prm_stdevs),3) if prm_stdevs else 0.0,
    "PRM_variance": round(prm_var,3),
    "PRM_entropy": round(entropy,3),
    "timestamp": datetime.now(timezone.utc).isoformat()
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nFinal QC Verdict:")
if (
    summary["missing_fields"]==0
    and 0.15<=summary["ORM_1_ratio"]<=0.35
    and 0.8<=summary["PRM_mean_avg"]<=0.95
    and summary["PRM_stdev_avg"]>=0.3
    and summary["PRM_entropy"]>=2.0
):
    print("✅ PASS — Dataset ready for ORM + PRM training.")
else:
    print("⚠️ REVIEW — Some metrics outside ideal range, inspect summary file.")
