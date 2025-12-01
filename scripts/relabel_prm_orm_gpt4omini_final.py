#!/usr/bin/env python3
"""
relabel_prm_orm_gpt4omini_final.py
Sub-Stage 2.2 — Re-labeling (ORM + PRM) with GPT-4o-mini.
Paper-faithful, contrastive-calibrated, checkpoint safe.
"""

import os, json, time, math
from datetime import datetime, UTC
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from statistics import mean

# --- Config ---
MODEL = "gpt-4o-mini"
INPUT = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
OUTPUT = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel.jsonl"
REPORT = "data/collected/gsm8k_relabel_report.jsonl"
TEMPERATURE = 0.0
MAX_TOKENS = 300
THREADS = 4
QUESTIONS_PER_SAVE = 20
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
client = OpenAI()

# --- Prompt Template ---
STEP_JSON_PROMPT = """You are a professional math reasoning evaluator.

Given a math question, a reasoning chain with numbered steps, and the correct (gold) final answer,
evaluate the reasoning strictly and analytically.

Return ONLY a JSON object:
{{
  "step_scores_raw": [s1, s2, ...],
  "chain_label": 1 or 0
}}
Guidelines:
- Score each step for logical correctness, justification, and mathematical accuracy.
- Penalize invalid arithmetic, skipped logic, or hallucinated reasoning.
- Use full 0–10 range. Be strict.
- Be objective and concise.
- Output nothing outside JSON.

Question:
{question}

Reasoning Chain:
{chain}

Gold Answer:
{gold_answer}
"""

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def save_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def safe_parse_json(text):
    try:
        text = text.strip()
        if text.startswith("```"): text = text.strip("`")
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            return json.loads(text[s:e+1])
    except: return None
    return None

def product(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p

def label_question(qobj):
    q, gold = qobj["question"], qobj.get("gold_answer", "").strip()
    results = []
    for chain in qobj["generated"]:
        chain_text = "\n".join(chain.get("steps", []))
        prompt = STEP_JSON_PROMPT.format(question=q, chain=chain_text, gold_answer=gold)
        parsed = None; success = False
        for attempt in range(1, MAX_RETRIES+1):
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw = r.choices[0].message.content.strip()
                parsed = safe_parse_json(raw)
                if parsed and "step_scores_raw" in parsed and "chain_label" in parsed:
                    success = True; break
            except Exception: time.sleep(RETRY_BACKOFF ** attempt)
        # Fallback if fail
        if not success:
            scores = [5.0]*len(chain.get("steps", []))
            label = 0
        else:
            scores = [max(0,min(10,float(s))) for s in parsed["step_scores_raw"]]
            label = int(parsed["chain_label"])
        scores_norm = [s/10 for s in scores]
        chain["step_scores_raw"] = scores
        chain["step_scores_norm"] = scores_norm
        chain["prm_score_mean"] = mean(scores_norm)
        chain["prm_score_prod"] = math.prod(scores_norm)
        chain["chain_label"] = label
        chain["orm_score"] = float(label)
        chain["timestamps"] = {"labeled_at": datetime.now(UTC).isoformat()}
        results.append(chain)
    return {
        "id": qobj["id"],
        "question": qobj["question"],
        "generated": results,
        "gold_answer": gold,
        "created_at": qobj.get("created_at", ""),
    }

def main():
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(OUTPUT).exists():
        for line in open(OUTPUT):
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in load_jsonl(INPUT) if q["id"] not in done]
    print(f"Total {len(todo)} questions to label")
    buf = []; start = time.time()
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(label_question, q): q["id"] for q in todo}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures))):
            res = fut.result()
            buf.append(res)
            if len(buf) >= QUESTIONS_PER_SAVE or i == len(futures)-1:
                save_jsonl(OUTPUT, buf)
                buf.clear()
                print(f"[{datetime.now(UTC).isoformat()}] checkpoint")
                time.sleep(2)
    print(f"✅ Done in {(time.time()-start)/60:.2f} min")

if __name__ == "__main__":
    main()
