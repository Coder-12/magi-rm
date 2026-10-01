#!/usr/bin/env python3
"""
scripts/collect_chains_optimized.py

High-performance, production-ready data collector for GSM8K reasoning traces.
Generates k chains per question using an LLM (OpenAI client) and writes a JSONL file.

Key features:
- ThreadPoolExecutor parallelism across questions
- Buffered writes to disk (batch flush)
- Robust step/final-answer extraction (multiple patterns)
- Exponential-backoff retries on API errors
- UTC timestamps (timezone-aware)
- Configurable model, temperature, k, workers, subsample, etc.
- Uses src.prompts.SOLVER_PROMPT if available; otherwise falls back to internal prompt
"""

import os
import sys
import json
import time
import uuid
import random
import argparse
from datetime import datetime, UTC
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# LLM client (OpenAI)
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI client import failed; ensure openai package is installed.") from e

# optional prompts module (use internal fallback if not present)
try:
    from src import prompts as user_prompts
    SOLVER_PROMPT_TEMPLATE = getattr(user_prompts, "SOLVER_PROMPT", None)
except Exception:
    SOLVER_PROMPT_TEMPLATE = None

# Fallback solver prompt (optimized)
FALLBACK_SOLVER_PROMPT = """You are a careful, step-by-step reasoning assistant.

Solve the following math problem accurately. Follow these rules:
1. Clearly enumerate each reasoning step as "1.", "2.", "3.", etc.
2. Substitute all given numbers into formulas; do not leave symbols like 'C' or 'x'.
3. Show intermediate arithmetic where relevant.
4. Double-check your calculations before giving the final answer.
5. End with a single line exactly like: Final Answer: <numeric value>

If units are mentioned (like dollars, meters, etc.), include them in the final numeric answer.
If not, provide a plain numeric answer.

Problem:
{question}
"""

# Extraction regexes
import re
STEP_RE = re.compile(r'^\s*\d+\.\s*(.*)', flags=re.MULTILINE)
FINAL_ANSWER_RE = re.compile(
    r'(?:Final\s*Answer[:\-]?\s*|Answer[:\-]?\s*|The\s*answer\s*(?:is|=)\s*)(.+)$',
    flags=re.IGNORECASE | re.MULTILINE
)

def extract_steps_and_answer(text: str):
    """Return (steps:list[str], final_answer:str). Robust fallbacks."""
    # Find numbered steps first
    steps = STEP_RE.findall(text)
    if not steps:
        # fallback: split by non-empty lines and remove likely header lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # If first line is "Solution" or similar, drop it
        if lines and re.match(r'^(Solution|Steps|Answer)\b', lines[0], flags=re.IGNORECASE):
            lines = lines[1:]
        # assume last line is final answer candidate
        steps = lines if len(lines) <= 1 else lines[:-1] # treat last line as final answer line
    # final answer extraction
    m = FINAL_ANSWER_RE.search(text)
    final = m.group(1).strip() if m else (steps[-1].strip() if steps else text.strip())
    return steps, final

def generate_k_chains_for_question(client: OpenAI, model: str, prompt: str, k: int, temperature: float, max_tokens: int, per_call_retries: int = 2):
    """
    Generate k chains sequentially for one question. We keep sequential per-question generation
    to avoid over-parallelizing single-question requests and to ensure reproducibility.
    """
    chains = []
    for i in range(k):
        attempt = 0
        backoff = 1.0
        while True:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # assume first choice exists
                text = resp.choices[0].message.content.strip()
                chains.append(text)
                break
            except Exception as e:
                attempt += 1
                if attempt > per_call_retries:
                    # final fallback: append an empty chain placeholder
                    chains.append("Failed to generate chain after retries.")
                    break
                sleep_t = backoff + random.random() * 0.5
                time.sleep(sleep_t)
                backoff *= 2.0
    return chains

def collect_one_question(client: OpenAI, model: str, q: str, k: int, temperature: float, max_tokens: int, prompt_template: str, q_index: int):
    """Collect record for a single question (generate k chains, parse them)."""
    prompt = prompt_template.format(question=q)
    raw_chains = generate_k_chains_for_question(client, model, prompt, k, temperature, max_tokens)
    generated = []
    for cid, text in enumerate(raw_chains):
        steps, final = extract_steps_and_answer(text)
        generated.append({
            "chain_id": cid,
            "raw_text": text,
            "steps": steps,
            "final_answer": final,
            "step_labels": [None] * len(steps),
            "chain_label": None,
            "orm_score_placeholder": None,
            "timestamps": {"generated_at": datetime.now(UTC).isoformat()}
        })
    rec = {
        "id": str(uuid.uuid4()),
        "question": q,
        "dataset": "gsm8k",
        "model": model,
        "solver_k": k,
        "temperature": temperature,
        "generated": generated,
        "notes": "",
        "created_at": datetime.now(UTC).isoformat()
    }
    # Provide lightweight logging summary (not verbose)
    basic_stats = {
        "qid_index": q_index,
        "chains": len(generated),
        "steps_mean": (sum(len(c["steps"]) for c in generated) / (len(generated) or 1))
    }
    return rec, basic_stats

def main():
    parser = argparse.ArgumentParser(description="Optimized GSM8K reasoning chain collector")
    parser.add_argument("--output_dir", type=str, default="data/collected")
    parser.add_argument("--output_file", type=str, default="gsm8k_train_chains.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel question workers")
    parser.add_argument("--subsample", type=int, default=1000, help="How many questions from GSM8K train to collect (0=all)")
    parser.add_argument("--buffer_size", type=int, default=25, help="How many records to buffer before writing to disk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_pause_after_batch", type=float, default=1.5)
    parser.add_argument("--max_pause_after_batch", type=float, default=2.5)
    args = parser.parse_args()

    random.seed(args.seed)

    # Build prompt template (prefer user-provided if present)
    prompt_template = SOLVER_PROMPT_TEMPLATE if SOLVER_PROMPT_TEMPLATE else FALLBACK_SOLVER_PROMPT

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key)

    # Load GSM8K train split
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets library required (pip install datasets)") from e

    ds = load_dataset("gsm8k", "main")
    questions_all = ds["train"]["question"]
    n_questions = len(questions_all)
    print(f"Total number of Training GSM8K Questions: {n_questions}")
    if args.subsample and args.subsample > 0:
        questions = questions_all[900:900+args.subsample]
    else:
        questions = questions_all

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.output_file

    # Prepare thread pool
    workers = min(args.workers, max(1, os.cpu_count() or 4))
    print(f"[INFO] Collecting {len(questions)} questions with k={args.k}, workers={workers}, model={args.model}")
    print(f"[INFO] Output will be written to {out_file}")
    buffer = []
    written = 0
    stats_accum = {"questions": 0, "chains": 0, "avg_steps": []}

    # Submit jobs
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_file, "w", encoding="utf8") as fout:
        futures = {}
        for i, q in enumerate(questions):
            futures[ex.submit(collect_one_question, client, args.model, q, args.k, args.temperature, args.max_tokens, prompt_template, i)] = i

        completed = 0
        for fut in as_completed(futures):
            try:
                rec, basic_stats = fut.result()
            except Exception as e:
                print(f"[WARN] A question job failed: {e}")
                continue

            buffer.append(json.dumps(rec, ensure_ascii=False))
            stats_accum["questions"] += 1
            stats_accum["chains"] += len(rec["generated"])
            stats_accum["avg_steps"].append(basic_stats["steps_mean"])

            completed += 1
            # Periodic write
            if len(buffer) >= args.buffer_size or completed == len(futures):
                fout.write("\n".join(buffer) + "\n")
                fout.flush()
                written += len(buffer)
                buffer = []

            # Periodic status print
            if completed % 50 == 0 or completed == len(futures):
                mean_steps = sum(stats_accum["avg_steps"]) / len(stats_accum["avg_steps"]) if stats_accum["avg_steps"] else 0.0
                print(f"[PROGRESS] {completed}/{len(futures)} questions collected | total_chains={stats_accum['chains']} | mean_steps={mean_steps:.2f}")

            # Polite adaptive pause in small random bursts to reduce tiny rate-limit chance
            if completed % 200 == 0:
                pause = random.uniform(args.min_pause_after_batch, args.max_pause_after_batch)
                print(f"[INFO] Pausing {pause:.2f}s to avoid transient rate limits...")
                time.sleep(pause)

    # final aggregate print
    mean_steps = sum(stats_accum["avg_steps"]) / len(stats_accum["avg_steps"]) if stats_accum["avg_steps"] else 0.0
    print(f"\n[DONE] Collected {stats_accum['questions']} questions, {stats_accum['chains']} chains, avg steps {mean_steps:.2f}")
    print(f"Output file: {out_file}")

if __name__ == "__main__":
    main()
