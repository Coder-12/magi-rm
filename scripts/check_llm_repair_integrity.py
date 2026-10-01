#!/usr/bin/env python3
"""
check_llm_repair_integrity.py

Verifies that all previously missing final_answer fields
are now correctly filled in the LLM-fixed dataset.
"""

import json
from pathlib import Path

ORIG = Path("data/collected/gsm8k_training_chains_1000.jsonl")
FIXED = Path("data/collected/gsm8k_training_chains_1000_llm_fixed.jsonl")

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def find_missing(records):
    missing = []
    for ri, r in enumerate(records):
        for ci, c in enumerate(r.get("generated", [])):
            fa = c.get("final_answer")
            if fa is None or (isinstance(fa, str) and fa.strip() == ""):
                missing.append((ri, ci, r["id"]))
    return missing

def main():
    orig = load_jsonl(ORIG)
    fixed = load_jsonl(FIXED)

    missing_orig = find_missing(orig)
    missing_fixed = find_missing(fixed)

    print(f"[INFO] Original file had {len(missing_orig)} missing final_answer entries.")
    print(f"[INFO] LLM-fixed file has {len(missing_fixed)} missing final_answer entries.")

    # verify overlap by record_id
    orig_ids = {(r["id"], ci) for ri, ci, _id in missing_orig for r in [orig[ri]]}
    fixed_ids = {(r["id"], ci) for ri, ci, _id in missing_fixed for r in [fixed[ri]]}

    repaired = orig_ids - fixed_ids
    still_missing = orig_ids & fixed_ids

    print(f"\n✅ Repaired entries: {len(repaired)}")
    print(f"❌ Still missing: {len(still_missing)}")

    if still_missing:
        print("\nThese are still missing:")
        for rid, cid in sorted(list(still_missing))[:20]:
            print(f"  record_id={rid}, chain_id={cid}")

    print("\n--- Spot check of a few repaired answers ---")
    for i, (rid, cid) in enumerate(list(repaired)[:]):
        rec = next((r for r in fixed if r["id"] == rid), None)
        if not rec: continue
        chain = rec["generated"][cid]
        print(f"[{i+1}] id={rid}, chain_id={cid}, final_answer={chain.get('final_answer')}")

if __name__ == "__main__":
    main()
