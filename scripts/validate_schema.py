#!/usr/bin/env python3
import json
from pathlib import Path

INPUT = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"

REQUIRED_CHAIN_FIELDS = {"chain_id","raw_text","steps","final_answer_used","timestamps"}
REQUIRED_RECORD_FIELDS = {"id","question","generated"}

def main():
    total = valid = 0
    with open(INPUT,"r",encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
                assert REQUIRED_RECORD_FIELDS.issubset(rec)
                for c in rec["generated"]:
                    assert REQUIRED_CHAIN_FIELDS.issubset(c)
                    assert isinstance(c["steps"], list) and all(isinstance(s,str) for s in c["steps"])
                valid += 1
            except Exception as e:
                print(f"[WARN] Invalid record {total}: {e}")
    print(f"[DONE] Valid records: {valid}/{total} ({valid/total*100:.1f}%)")

if __name__ == "__main__":
    main()
