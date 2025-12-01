#!/usr/bin/env python3
import json, hashlib
from pathlib import Path

INPUT  = "data/collected/gsm8k_training_chains_1000_clean.jsonl"
OUTPUT = "data/collected/gsm8k_training_chains_1000_dedup.jsonl"

def chain_fingerprint(chain):
    key = (chain.get("raw_text","") + "|" + chain.get("final_answer_used","")).strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def main():
    seen = set(); kept = removed = 0
    with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            unique = []
            for c in rec["generated"]:
                fp = chain_fingerprint(c)
                if fp not in seen:
                    seen.add(fp); unique.append(c)
                else:
                    removed += 1
            kept += len(unique)
            rec["generated"] = unique
            if unique:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[DONE] Deduplicated chains: kept={kept}, removed={removed}")
    print(f"Output → {OUTPUT}")

if __name__ == "__main__":
    main()
