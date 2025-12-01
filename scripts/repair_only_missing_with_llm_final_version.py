#!/usr/bin/env python3
"""
repair_only_missing_with_llm_final_version.py

Fix ONLY chains with empty or missing final_answer in the original dataset
using GPT-4.1 (deterministic, low-cost calls). Supports dry-run mode.

Usage (dry run first):
python scripts/repair_only_missing_with_llm_final_version.py \
  --orig_file data/collected/gsm8k_training_chains_1000.jsonl \
  --output_file data/collected/gsm8k_training_chains_1000_llm_fixed.jsonl \
  --report_file data/collected/missing_final_llm_report.jsonl \
  --dry_run

To actually fix (after verifying):
add --force
"""

import json
import time
import re
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

# ====== CONFIG ======
MODEL = "gpt-4.1"
TEMP = 0.0
MAX_TOK = 80
SLEEP = 0.3  # seconds between API calls
RETRIES = 2

PROMPT_TEMPLATE = """You are a precise numeric answer extractor. Given a single chain-of-thought solution, return only a JSON object with a single key "final_answer" whose value is the final numeric answer in canonical form (no currency symbols, no commas, no units) OR the string "NONE" if no final numeric answer can be confidently extracted.

Rules:
- Prefer explicitly stated final lines like "Final Answer:", "Therefore, x = 18", "Hence, 34".
- If the chain contains a final equation (e.g., "20 - t = 2"), solve it algebraically for the variable and return the numeric value (e.g., "18").
- If the chain computes a final numeric sum such as "5 + 8 + 10 + 0 = 23", return 23.
- Do NOT return intermediate step numbers (like '1.' or '2.') if they are step indices.
- Return the numeric answer as a number (e.g., 18) or a numeric string (e.g., "18").
- Return ONLY a JSON object, no explanation.

Example Output
{{"final_answer": 23}}

Chain:
{chain}
"""

JSON_RE = re.compile(r'\{.*\}', re.DOTALL)
NUM_RE = re.compile(r'[-+]?\d[\d,\.]*')


# ====== HELPERS ======
def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_llm_output(text: str):
    """Extract numeric answer from JSON or text."""
    try:
        m = JSON_RE.search(text)
        if m:
            obj = json.loads(m.group(0))
            ans = str(obj.get("final_answer", "")).strip()
            if ans.lower() != "none":
                return ans
    except Exception:
        pass
    m2 = NUM_RE.findall(text)
    if m2:
        return m2[-1].replace(",", "").strip()
    return ""


def call_llm(client, chain_text):
    """Make one robust GPT call with retries."""
    prompt = PROMPT_TEMPLATE.format(chain=chain_text)
    for attempt in range(RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP,
                max_tokens=MAX_TOK,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == RETRIES:
                print(f"[ERROR] LLM failed after {RETRIES} retries: {e}")
                return ""
            time.sleep(1 + attempt)


# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Execute LLM calls; required to actually call the API")
    args = parser.parse_args()

    orig_path = Path(args.orig_file)
    out_path = Path(args.output_file)
    rep_path = Path(args.report_file)

    if not orig_path.exists():
        raise FileNotFoundError(f"File not found: {orig_path}")

    records = load_jsonl(orig_path)
    total = sum(len(r.get("generated", [])) for r in records)

    # identify missing chains
    missing = []
    for ri, rec in enumerate(records):
        for ci, c in enumerate(rec.get("generated", [])):
            fa = c.get("final_answer")
            if fa is None or (isinstance(fa, str) and fa.strip() == ""):
                missing.append((ri, ci, rec["id"]))
    print(f"[INFO] Loaded {len(records)} records ({total} chains)")
    print(f"[INFO] Found {len(missing)} missing final_answer chains")

    if not missing:
        print("[INFO] No missing answers found. Nothing to fix.")
        return

    for i, (_, _, rid) in enumerate(missing[:50]):
        print(f"  {i+1}. record_id={rid}")

    if args.dry_run and not args.force:
        print("\n💡 DRY RUN MODE — No API calls made.")
        print("Use --force to actually repair missing answers.")
        return

    # initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    client = OpenAI(api_key=api_key)

    repaired = 0
    report = []

    for ri, ci, rid in missing:
        chain = records[ri]["generated"][ci]
        raw_text = chain.get("raw_text", "").strip()
        if not raw_text:
            continue

        print(f"[LLM] ({repaired+1}/{len(missing)}) record_id={rid} chain_id={ci}")
        llm_out = call_llm(client, raw_text)
        extracted = parse_llm_output(llm_out)

        if extracted:
            chain["final_answer"] = extracted
            repaired += 1
            status = "fixed"
        else:
            status = "failed"

        report.append({
            "record_id": rid,
            "chain_id": ci,
            "status": status,
            "final_answer": extracted,
            "llm_output": llm_out,
        })
        time.sleep(SLEEP)

    # write outputs
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "calls": len(missing),
        "repaired": repaired,
    }
    with open(rep_path, "w", encoding="utf-8") as frep:
        frep.write(json.dumps({"meta": meta}) + "\n")
        for r in report:
            frep.write(json.dumps(r) + "\n")

    print(f"\n✅ Done — {repaired}/{len(missing)} repaired successfully.")
    print(f"Output: {out_path}")
    print(f"Report: {rep_path}")


if __name__ == "__main__":
    main()
