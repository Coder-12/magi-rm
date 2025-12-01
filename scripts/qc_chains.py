import os
import json
import statistics
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from datetime import datetime
import re
import argparse

# Regex for validating extraction consistency
FINAL_ANSWER_RE = re.compile(r'Final Answer[:\-]?\s*(.+)$', flags=re.IGNORECASE | re.MULTILINE)


def evaluate_accuracy(preds, golds):
    """Compute exact-match accuracy between predictions and golds."""
    correct = 0
    for p, g in zip(preds, golds):
        # Normalize both sides (strip, remove $ signs, commas)
        p_norm = p.strip().replace('$', '').replace(',', '')
        g_norm = g.strip().replace('$', '').replace(',', '')
        try:
            if abs(float(p_norm) - float(g_norm)) < 1e-3:
                correct += 1
            elif p_norm == g_norm:
                correct += 1
        except Exception:
            if p_norm == g_norm:
                correct += 1
    return correct / len(preds) if preds else 0.0


def run_qc(input_file, k_expected, sample_correctness_check=100, output_dir="data/qc"):
    print("\n=== Data Quality & Sanity Check ===")
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)
    report = {"file": str(input_path), "timestamp": datetime.utcnow().isoformat() + "Z"}

    # ---- Step 1: Load Data ----
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} records")

    # ---- Step 2: Structural Sanity ----
    chain_counts = [len(r["generated"]) for r in data]
    avg_chains = statistics.mean(chain_counts)
    missing_final_ans = 0
    empty_steps = 0
    step_lens = []

    for r in data:
        for chain in r["generated"]:
            steps = chain.get("steps", [])
            step_lens.append(len(steps))
            if len(steps) == 0:
                empty_steps += 1
            final = chain.get("final_answer", "").strip()
            if not final:
                missing_final_ans += 1

    avg_steps = statistics.mean(step_lens) if step_lens else 0
    report["avg_chains_per_question"] = avg_chains
    report["avg_steps_per_chain"] = avg_steps
    report["missing_final_answers"] = missing_final_ans
    report["empty_steps"] = empty_steps

    print(f"Average chains per question: {avg_chains:.2f} (expected {k_expected})")
    print(f"Average steps per chain: {avg_steps:.2f} (expected 3–6)")
    print(f"Chains with empty steps: {empty_steps} ({empty_steps / (len(step_lens) + 1e-9) * 100:.2f}%)")
    print(f"Chains missing final answer: {missing_final_ans}")

    # ---- Step 3: Diversity Analysis ----
    unique_answers = [len(set(chain["final_answer"] for chain in r["generated"])) for r in data]
    avg_unique = statistics.mean(unique_answers)
    report["avg_unique_answers_per_question"] = avg_unique
    print(f"Average unique final answers per question: {avg_unique:.2f} (ideal: 2–4 @ temp=0.45)")

    # ---- Step 4: Correctness Spot Check ----
    ds = load_dataset("gsm8k", "main")["train"]
    gold_dict = {ex["question"]: ex["answer"].split("####")[-1].strip() for ex in ds}

    subsample = data[:sample_correctness_check]
    preds, golds = [], []
    for rec in subsample:
        q = rec["question"]
        gold = gold_dict.get(q)
        if not gold:
            continue
        chain_preds = [c["final_answer"] for c in rec["generated"]]
        maj_pred = Counter(chain_preds).most_common(1)[0][0]
        preds.append(maj_pred)
        golds.append(gold)

    overall_acc = evaluate_accuracy(preds, golds)
    report["sample_correctness_accuracy"] = overall_acc
    print(f"Sample correctness accuracy (majority vote): {overall_acc * 100:.2f}%")

    # ---- Step 5: Placeholder Integrity ----
    missing_labels = 0
    for rec in data:
        for chain in rec["generated"]:
            if "step_labels" not in chain or "chain_label" not in chain:
                missing_labels += 1
    report["missing_label_placeholders"] = missing_labels
    print(f"Missing label placeholders: {missing_labels}")

    # ---- Step 6: Final QC Verdict ----
    cond_structure = avg_chains == k_expected and empty_steps < 0.05 * len(step_lens)
    cond_diversity = 1.5 <= avg_unique <= 4.0
    cond_correctness = overall_acc >= 0.7
    cond_labels = missing_labels == 0
    passed = all([cond_structure, cond_diversity, cond_correctness, cond_labels])

    if passed:
        verdict = "✅ PASS — Data ready for RM training"
    elif cond_correctness >= 0.6:
        verdict = "⚠️ WARN — Usable but verify accuracy and regexes"
    else:
        verdict = "❌ FAIL — Needs regeneration (check prompt/temp)"

    report["verdict"] = verdict
    print(f"\nFinal QC Verdict: {verdict}")

    # ---- Step 7: Save report ----
    out_file = Path(output_dir) / f"qc_report_{input_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2)
    print(f"QC report saved to {out_file}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/collected/gsm8k_training_chains_1000.jsonl")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--sample_correctness_check", type=int, default=100)
    args = parser.parse_args()

    run_qc(args.input_file, args.k, args.sample_correctness_check)
