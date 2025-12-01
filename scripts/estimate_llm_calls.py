#!/usr/bin/env python3
"""
estimate_llm_calls.py
Estimate how many steps will trigger LLM fallback in PRM labeling.
"""

import re, json, argparse
from pathlib import Path

HEURISTIC_TOLERANCE = 1e-6
NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def safe_eval_arith(expr):
    expr = expr.replace(",", "")
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None

def heuristic_label_step(step):
    s = step.strip()
    if "=" in s:
        l, r = s.split("=", 1)
        lv, rv = safe_eval_arith(l), safe_eval_arith(r)
        if lv is not None and rv is not None:
            return abs(lv - rv) <= HEURISTIC_TOLERANCE
    if any(ch.isdigit() for ch in s) and any(op in s for op in "+-*/"):
        return True
    return False

def main(args):
    n_steps = 0
    n_heur = 0
    sample = 0
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.max_records and i >= args.max_records:
                break
            rec = json.loads(line)
            for c in rec.get("generated", []):
                for s in c.get("steps", []):
                    n_steps += 1
                    if heuristic_label_step(s):
                        n_heur += 1
            sample += 1

    frac_heur = n_heur / max(1, n_steps)
    frac_llm = 1 - frac_heur
    print(f"Sampled {sample} records → {n_steps} steps")
    print(f"Heuristic OK: {frac_heur*100:.2f}% | Needs LLM: {frac_llm*100:.2f}%")
    est_total_steps = args.total_chains * args.mean_steps
    print(f"→ Estimated total LLM calls ≈ {frac_llm*est_total_steps:.0f}")
    print(f"Expected cost (GPT-4.1): ${frac_llm*est_total_steps*0.0024:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/collected/gsm8k_training_chains_1000_clean.jsonl")
    p.add_argument("--max_records", type=int, default=100)
    p.add_argument("--total_chains", type=int, default=6000)
    p.add_argument("--mean_steps", type=float, default=7.1)
    main(p.parse_args())
