import os
import sys


# --- ensure project root is always importable ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import json
from pathlib import Path
from collections import Counter
import statistics
from datasets import load_dataset
from src.eval import normalize_answer, evaluate_predictions

def main(args):
    file_path = Path(args.input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load JSONL
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print("Warning: Invalid JSON line skipped")

    # Coarse Check 1: Structure Validation
    print("\n[Coarse Check 1: Structure Validation]")
    print(f"Total records: {len(data)} (expected: {args.subsample})")
    chain_counts = [len(rec['generated']) for rec in data]
    avg_chains = statistics.mean(chain_counts)
    print(f"Avg chains per questions: {avg_chains: .2f} (expected: {args.k})")
    if any(c != args.k for c in chain_counts):
        print("Anomaly: Inconsistent chain counts—refine collection for affected questions.")

    # Coarse Check 2: Extraction Quality
    print("\n[Coarse Check 2: Extraction Quality]")
    step_lens = [len(chain['steps']) for rec in data for chain in rec['generated']]
    avg_steps = statistics.mean(step_lens) if step_lens else 0
    print(f"Avg steps per chain: {avg_steps:.2f} (expected: 3-6 for GSM8K)")
    empty_steps_pct = (sum(1 for l in step_lens if l == 0) / len(step_lens)) * 100
    print(f"% chains with empty steps: {empty_steps_pct:.2f}% (should be <5%; if high, check STEP_RE)")
    final_ans_present = all(chain['final_answer'].strip() for rec in data for chain in rec['generated'])
    print(f"All chains have final_answer: {final_ans_present} (if False, check FINAL_ANSWER_RE)")

    # Fine Check 1: Diversity (per-question unique final answers)
    print("\n[Fine Check 1: Diversity Analysis]")
    unique_ans_per_q = [len(set(chain['final_answer'] for chain in rec['generated'])) for rec in data]
    avg_unique = statistics.mean(unique_ans_per_q)
    print(
        f"Avg unique final_answers per question: {avg_unique:.2f} (expected: 2-4 at temp=0.7; low = duplicates, high = high variance)")
    if avg_unique < 1.5:
        print("Anomaly: Low diversity—check temperature or model for determinism.")

    # Fine Check 2: Sample Correctness (spot-check vs gold; subsample for efficiency)
    print("\n[Fine Check 2: Correctness Spot-Check]")
    ds = load_dataset('gsm8k', 'main')['train']
    gold_dict = {row['question']: row['answer'].split('#### ')[-1].strip() for row in ds}  # Gold final answers

    # Subsample 100 questions for quick check
    subsample_data = data[:args.correctness_subsample]
    preds, golds = [], []
    chain_accuracies = []
    for rec in subsample_data:
        q = rec['question']
        gold = gold_dict.get(q)
        if gold is None:
            continue  # Skip if question not in train (unlikely)
        chain_preds = [chain['final_answer'] for chain in rec['generated']]
        chain_acc = evaluate_predictions(chain_preds, [gold] * len(chain_preds))
        chain_accuracies.append(chain_acc)
        # Aggregate per question (majority vote for demo)
        maj_pred = Counter(chain_preds).most_common(1)[0][0]
        preds.append(maj_pred)
        golds.append(gold)

    overall_acc = evaluate_predictions(preds, golds)
    avg_chain_acc = statistics.mean(chain_accuracies)
    print(f"Subsample size: {len(subsample_data)} questions ({len(subsample_data) * args.k} chains)")
    print(f"Avg per-chain accuracy: {avg_chain_acc * 100:.2f}% (expected: 70-80% for gpt-4o-mini)")
    print(f"Majority-vote accuracy: {overall_acc * 100:.2f}% (benchmark: should match or exceed base model)")
    if avg_chain_acc < 0.6:
        print("Anomaly: Low correctness—check prompt or model; refine generation.")

    # Orchestrator Decision
    print("\n[Orchestrator Decision]")
    cond1 = (avg_steps >= 3) and (empty_steps_pct < 5) and final_ans_present  # High majority quality
    cond2 = (avg_unique >= 2) and (avg_chain_acc >= 0.7)  # High confidence
    print(f"Condition 1 (Quality): {cond1}")
    print(f"Condition 2 (Confidence): {cond2}")
    if cond1 and cond2:
        print("Data classified as EASY ✅ — Proceed to 2.2 (Labeling).")
    else:
        print("Data classified as HARD 🔁 — Refine: Re-generate low-quality questions or tune extraction regex.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/collected/gsm8k_training_chains_1000.jsonl")
    parser.add_argument("--subsample", type=int, default=1000)  # From generation args
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--correctness_subsample", type=int, default=100, help="Questions to spot-check correctness")
    args = parser.parse_args()
    main(args)


