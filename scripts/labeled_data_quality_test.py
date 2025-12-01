#!/usr/bin/env python3
"""
labeled_data_quality_test.py — Quality Testing for Labeled Chains (Post-Labeling Validation).
Validates ORM/PRM labels in LABELED_FILE against gold answers and heuristics.
Produces summary stats (including score histograms, variance/stdev, entropy) and flags issues for manual review.
Run after labeling to ensure data is ready for RM training (PRM + ORM).
"""

import json
import math
import re
import random
from statistics import mean, stdev
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime, timezone
from scipy.stats import entropy  # pip install scipy if needed

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
LABELED_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_patched_final.jsonl"  # Your labeled output
SUMMARY_FILE = "data/collected/gsm8k_labeled_quality_summary.json"  # Output summary
MANUAL_REVIEW_FILE = "data/collected/gsm8k_labeled_manual_review_samples.jsonl"  # Flagged chains for review
SAMPLE_SIZE_FOR_REVIEW = 20  # Number of chains to sample for manual review (prioritize issues)
LOG_FILE = "data/collected/gsm8k_quality_test_log.txt"  # Log pass/fail and details

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {path}")
    return records

def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_answer_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace("$", "").replace(",", "")
    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if m:
        val = float(m.group(0))
        return str(int(val)) if abs(val - round(val)) < 1e-9 else str(val)
    return s.lower()

def compute_entropy(scores: List[float]) -> float:
    if not scores:
        return 0.0
    norm_scores = [s + 1e-10 for s in scores]  # Avoid log(0)
    return entropy(norm_scores)

# ------------------------------------------------------------
# Quality Metrics Functions
# ------------------------------------------------------------
def validate_orm(chain: Dict[str, Any], gold: str) -> Dict[str, Any]:
    try:
        final_ans = chain.get("final_answer", "")
        norm_final = normalize_answer_text(final_ans)
        norm_gold = normalize_answer_text(gold)
        expected_label = 1 if norm_final == norm_gold and norm_gold != "" else 0
        actual_label = chain.get("chain_label", 0)
        return {
            "matches_gold": expected_label == 1,
            "label_correct": actual_label == expected_label,
            "status": chain.get("labeling_status", "unknown"),
            "final_answer": final_ans,
            "gold_answer": gold
        }
    except Exception as e:
        return {"label_correct": False, "reason": f"ORM error: {str(e)}"}

def validate_prm(chain: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw_scores = chain.get("step_scores_raw", [])
        num_steps = len(raw_scores)
        if num_steps == 0:
            return {"valid": False, "reason": "No step scores", "mean": 0.0, "prod": 0.0, "stdev": 0.0, "entropy": 0.0}
        # Range check
        in_range = all(0 <= s <= 10 for s in raw_scores)
        # Step count match (if "steps" key exists)
        expected_steps = len(chain.get("steps", []))
        steps_match = num_steps == expected_steps if expected_steps > 0 else True
        # Variance/stdev (flag low if <0.5 for multi-step)
        st_dev = stdev(raw_scores) if num_steps > 1 else 0.0
        variance_ok = st_dev >= 0.5 if num_steps > 1 else True
        # Aggregates match
        norm_scores = [s / 10.0 for s in raw_scores]
        calc_mean = mean(norm_scores)
        calc_prod = math.prod(norm_scores) if norm_scores else 0.0
        stored_mean = chain.get("prm_score_mean", 0.0)
        stored_prod = chain.get("prm_score_prod", 0.0)
        aggregates_match = abs(calc_mean - stored_mean) < 1e-6 and abs(calc_prod - stored_prod) < 1e-6
        # Entropy for confidence
        score_entropy = compute_entropy(norm_scores)
        return {
            "valid": in_range and steps_match and variance_ok and aggregates_match,
            "reason": "" if in_range and steps_match and variance_ok and aggregates_match else
                      f"Out of range: {not in_range}, Step mismatch: {not steps_match}, Low variance: {not variance_ok}, Aggregates mismatch: {not aggregates_match}",
            "num_steps": num_steps,
            "mean": stored_mean,
            "prod": stored_prod,
            "stdev": st_dev,
            "entropy": score_entropy
        }
    except Exception as e:
        return {"valid": False, "reason": f"PRM error: {str(e)}", "mean": 0.0, "prod": 0.0, "stdev": 0.0, "entropy": 0.0}

# ------------------------------------------------------------
# Main Quality Test
# ------------------------------------------------------------
def main():
    print(f"\n=== Quality Testing Labeled Data ({LABELED_FILE}) ===")
    try:
        records = load_jsonl(LABELED_FILE)
    except FileNotFoundError:
        print(f"Error: File {LABELED_FILE} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file - {str(e)}")
        return

    total_questions = len(records)
    total_chains = sum(len(r.get("generated", [])) for r in records)
    print(f"Loaded {total_questions} questions ({total_chains} chains)")

    orm_correct = 0
    prm_valid = 0
    heuristic_count = Counter()
    prm_means_correct = []
    prm_means_incorrect = []
    prm_stdevs = []
    prm_entropies = []
    score_histogram = Counter()  # Bin raw scores 0-10
    flagged_for_review = []  # Chains with issues

    for rec in records:
        gold = rec.get("gold_answer", "")
        chains = rec.get("generated", [])
        for chain in chains:
            orm_val = validate_orm(chain, gold)
            prm_val = validate_prm(chain)
            heuristic_count[chain.get("labeling_status", "ok")] += 1

            if orm_val.get("label_correct", False):
                orm_correct += 1
            if prm_val.get("valid", False):
                prm_valid += 1

            prm_stdevs.append(prm_val.get("stdev", 0.0))
            prm_entropies.append(prm_val.get("entropy", 0.0))
            for score in chain.get("step_scores_raw", []):
                score_histogram[int(score)] += 1

            if orm_val.get("matches_gold", False):
                prm_means_correct.append(prm_val.get("mean", 0.0))
            else:
                prm_means_incorrect.append(prm_val.get("mean", 0.0))

            # Flag if issues (add high mean on incorrect)
            if not orm_val.get("label_correct", True) or not prm_val.get("valid", True) or chain.get("labeling_status", "ok") != "ok" or (not orm_val.get("matches_gold", True) and prm_val.get("mean", 0.0) > 0.8):
                flagged = {
                    "question_id": rec["id"],
                    "chain": chain,
                    "orm_validation": orm_val,
                    "prm_validation": prm_val
                }
                flagged_for_review.append(flagged)

    # Summary Stats (enhanced)
    summary = {
        "total_questions": total_questions,
        "total_chains": total_chains,
        "orm_accuracy_pct": round(orm_correct / total_chains * 100, 2) if total_chains else 0.0,
        "prm_valid_pct": round(prm_valid / total_chains * 100, 2) if total_chains else 0.0,
        "heuristic_usage": dict(heuristic_count),
        "prm_mean_correct_chains": round(mean(prm_means_correct), 2) if prm_means_correct else 0.0,
        "prm_mean_incorrect_chains": round(mean(prm_means_incorrect), 2) if prm_means_incorrect else 0.0,
        "prm_avg_stdev": round(mean(prm_stdevs), 2) if prm_stdevs else 0.0,
        "prm_avg_entropy": round(mean(prm_entropies), 2) if prm_entropies else 0.0,
        "score_histogram": dict(sorted(score_histogram.items())),
        "flagged_count": len(flagged_for_review),
        "tested_at": datetime.now(timezone.utc).isoformat() + "Z"
    }

    # Save Summary and Flagged Samples
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {SUMMARY_FILE}")
    print(json.dumps(summary, indent=2))

    # Sample flagged for manual review (random for diversity)
    review_samples = random.sample(flagged_for_review, min(SAMPLE_SIZE_FOR_REVIEW, len(flagged_for_review))) if flagged_for_review else []
    append_jsonl(MANUAL_REVIEW_FILE, review_samples)
    print(f"\n{len(review_samples)} flagged chains saved to {MANUAL_REVIEW_FILE} for manual review.")

    # Log Pass/Fail to file
    log_msg = "\n✅ Data quality PASS: Labels robust for RM training." if (
        summary["orm_accuracy_pct"] > 95 and summary["prm_valid_pct"] > 90 and heuristic_count.get("ok", 0) / total_chains > 0.95 and summary["prm_avg_stdev"] > 0.5 and summary["prm_avg_entropy"] > 0.3
    ) else "\n⚠️ Data quality WARNING: Review flagged samples, heuristics, low variance/entropy."
    print(log_msg)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(log_msg + "\n" + json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()