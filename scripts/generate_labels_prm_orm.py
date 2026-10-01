#!/usr/bin/env python3
"""
gpt4o_labeling_final.py — FINAL (Sub-Stage 2.2) Label Generation using GPT-4o-mini.

Reliable, cost-efficient, and paper-faithful ORM + PRM labeling.
Preserves all heuristics, retries, and checkpoint logic from vLLM version.

Model: gpt-4o-mini
Features:
 - Paper-faithful PRM (0–10) and ORM (binary correctness)
 - Uses gold_answer for correctness comparison
 - Retry + exponential backoff
 - Checkpoint-safe (resumeable)
 - Dual PRM aggregations (mean + product)
 - Deterministic (temperature=0)
"""

import os
import re
import json
import time
import math
from statistics import mean
from pathlib import Path
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
INPUT_FILE = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_gpt4o.jsonl"
REPORT_FILE = "data/collected/gsm8k_label_report_gpt4o.jsonl"

QUESTIONS_PER_SAVE = 50
NUM_THREADS = 4
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5  # seconds base

# ------------------------------------------------------------
# Prompt Template
# ------------------------------------------------------------
STEP_JSON_PROMPT = """
You are an expert evaluator of step-by-step math reasoning.

Given the question, the reasoning chain (with numbered steps), and the gold final answer,
return ONLY a valid JSON object in this exact format:

{{
  "step_scores_raw": [s1, s2, ...],   // numbers 0–10 (10 = perfect)
  "chain_label": 0 or 1               // 1 if the final answer matches gold, else 0
}}

Rules:
- Output only valid JSON (no explanations or extra text).
- step_scores_raw length = number of steps in the chain.
- Use the gold answer to determine correctness.
Question:
{question}

Chain:
{chain}

Gold Answer:
{gold_answer}
""".strip()


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def append_jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def safe_parse_json(raw):
    """Safely extract a JSON object from LLM output."""
    try:
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1:
            return json.loads(txt[start:end+1])
    except Exception:
        pass
    return None

def heuristic_score_steps(steps):
    """Fallback if LLM labeling fails."""
    out = []
    for s in steps or []:
        s = (s or "").strip()
        if re.search(r"=", s) or (re.search(r"\d", s) and any(op in s for op in ["+", "-", "*", "/"])):
            out.append(8.0)
        elif len(s) < 20:
            out.append(6.0)
        else:
            out.append(5.0)
    return out or [5.0]

def normalize_answer_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace("$", "").replace(",", "")
    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if m:
        val = float(m.group(0))
        return str(int(val)) if abs(val - round(val)) < 1e-9 else str(val)
    return s.lower()

# ------------------------------------------------------------
# Core Labeling Function
# ------------------------------------------------------------
def label_one_question(client, q_obj):
    q_text = q_obj["question"]
    gold = q_obj.get("gold_answer", "").strip()
    results = []

    for chain in q_obj["generated"]:
        steps = chain.get("steps", [])
        chain_text = "\n".join(steps) if steps else chain.get("raw_text", "")
        prompt = STEP_JSON_PROMPT.format(question=q_text, chain=chain_text, gold_answer=gold)

        parsed, status = None, None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw = resp.choices[0].message.content.strip()
                parsed = safe_parse_json(raw)
                if parsed and "step_scores_raw" in parsed and "chain_label" in parsed:
                    status = "ok"
                    break
                else:
                    status = "parse_error"
            except Exception as e:
                status = f"llm_error: {e}"
            time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))

        if not parsed or status != "ok":
            step_scores_raw = heuristic_score_steps(steps)
            chain_label = 1 if normalize_answer_text(chain.get("final_answer", "")) == normalize_answer_text(gold) else 0
            status = status or "heuristic"
        else:
            step_scores_raw = [max(0.0, min(10.0, float(s))) for s in parsed["step_scores_raw"]]
            chain_label = int(parsed["chain_label"])

        step_scores_norm = [s / 10.0 for s in step_scores_raw]
        prm_mean = float(mean(step_scores_norm))
        prm_prod = float(math.prod(step_scores_norm))
        orm_score = float(chain_label)

        chain["step_scores_raw"] = step_scores_raw
        chain["step_scores_norm"] = step_scores_norm
        chain["prm_score_mean"] = prm_mean
        chain["prm_score_prod"] = prm_prod
        chain["orm_score"] = orm_score
        chain["chain_label"] = chain_label
        chain["labeling_status"] = status
        chain.setdefault("timestamps", {})["labeled_at"] = datetime.now(UTC).isoformat()

        results.append(chain)

    return {
        "id": q_obj["id"],
        "question": q_obj["question"],
        "gold_answer": gold,
        "generated": results,
        "created_at": q_obj["created_at"],
        "labeled_at": datetime.now(UTC).isoformat(),
    }

# ------------------------------------------------------------
# Main Orchestrator
# ------------------------------------------------------------
def main():
    print(f"\n=== Labeling (Sub-Stage 2.2) with {MODEL_NAME} ===")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completed_ids = set()

    if Path(OUTPUT_FILE).exists():
        for line in open(OUTPUT_FILE):
            try:
                completed_ids.add(json.loads(line)["id"])
            except:
                continue

    todo = [q for q in load_jsonl(INPUT_FILE) if q["id"] not in completed_ids]
    total = len(todo)
    print(f"Total to process: {total} (skipping {len(completed_ids)} already done)")

    if total == 0:
        print("✅ Nothing to label.")
        return

    buffer, processed = [], 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        futures = {pool.submit(label_one_question, client, q): q["id"] for q in todo}
        for fut in tqdm(as_completed(futures), total=total, desc="Labeling"):
            try:
                result = fut.result()
                buffer.append(result)
                processed += 1
            except Exception as e:
                print(f"[WARN] Thread failed: {e}")

            if len(buffer) >= QUESTIONS_PER_SAVE or processed == total:
                print(
                    f"{datetime.now(UTC).isoformat()} | Processed {processed}/{total} | Threads active: {NUM_THREADS}")
                append_jsonl(OUTPUT_FILE, buffer)
                append_jsonl(REPORT_FILE, [{"checkpoint": len(buffer), "timestamp": datetime.now(UTC).isoformat()}])
                print(f"[CHECKPOINT] Saved {len(buffer)} → {OUTPUT_FILE}")
                buffer = []
                time.sleep(2)  # 👈 throttle to stay underrate limits

    duration = (time.time() - start) / 60
    print(f"\n✅ Labeled {processed}/{total} questions in {duration:.2f} min")
    print(f"Output: {OUTPUT_FILE}\n")

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
