# #!/usr/bin/env python3
# """
# scripts/prepare_rm_datasets.py
#
# Reads your labelled JSONL and writes:
#  - data/processed/orm_train.jsonl, orm_val.jsonl, orm_test.jsonl
#  - data/processed/prm_train.jsonl, prm_val.jsonl, prm_test.jsonl
#  - data/processed/prm_train_seq.jsonl (aggregated PRM mean target)
#
# Assumptions:
#  - input JSONL contains records with keys:
#    id, question, gold_answer, generated -> list of chains.
#  - each chain has: chain_id (or chain_id implicit by index), steps (list[str]),
#    final_answer, final_answer_norm, step_scores_raw (0..10), step_scores_norm (0..1),
#    prm_score_mean (0..1), prm_score_prod (0..1), chain_label (0/1), orm_score (float).
#
# Run:
# python scripts/prepare_rm_datasets.py --input data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl \
#     --outdir data/processed --train_split data/collected/gsm8k_rm_train_FINAL_balanced.jsonl \
#     --val_split data/collected/gsm8k_rm_val.jsonl --test_split data/collected/gsm8k_rm_test.jsonl
# """
#
# import json
# from pathlib import Path
# import argparse
# from typing import List, Dict
# import random
#
# PROMPT_TEMPLATE = (
#     "You are a careful step-by-step math solver. Given the question and a candidate reasoning chain, "
#     "decide whether the chain is correct and (for PRM) rate each step.\n\n"
#     "Question: {question}\n\n"
#     "Chain:\n{chain_text}\n\n"
#     "Final Answer: {final_answer}\n"
# )
#
# SEQ_PROMPT_TEMPLATE = (
#     "Question: {question}\nChain steps (concise):\n{chain_text}\n\n"
#     "Return the aggregated PRM mean score between 0 and 1."
# )
#
#
# def load_jsonl(path: Path):
#     with open(path, "r", encoding="utf8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             yield json.loads(line)
#
#
# def build_chain_text(chain: Dict) -> str:
#     # join steps with numbered lines for deterministic tokenization
#     steps = chain.get("steps") or []
#     if steps:
#         return "\n".join(f"{i+1}. {s.strip()}" for i, s in enumerate(steps))
#     # fallback to raw_text
#     return chain.get("raw_text", "").strip()
#
#
# def ensure_dir(p: Path):
#     p.parent.mkdir(parents=True, exist_ok=True)
#
#
# def write_jsonl(path: Path, iterables):
#     ensure_dir(path)
#     with open(path, "w", encoding="utf8") as f:
#         for obj in iterables:
#             f.write(json.dumps(obj, ensure_ascii=False) + "\n")
#
#
# def extract_examples_from_record(rec: Dict, include_prm_seq=True):
#     qid = rec.get("id")
#     question = rec.get("question", "").strip()
#     gold = rec.get("gold_answer", rec.get("final_answer", "")).strip()
#     for chain in rec.get("generated", []):
#         cid = chain.get("chain_id", None)
#         chain_text = build_chain_text(chain)
#         final_answer = chain.get("final_answer", "").strip()
#         final_answer_norm = chain.get("final_answer_norm", final_answer)
#         # ORM example
#         orm_example = {
#             "qid": qid,
#             "chain_id": cid,
#             "prompt": PROMPT_TEMPLATE.format(question=question, chain_text=chain_text, final_answer=final_answer),
#             "label": int(chain.get("chain_label", int(chain.get("orm_score", 0)))),
#             "meta": {
#                 "gold_answer": gold,
#                 "final_answer": final_answer,
#                 "final_answer_norm": final_answer_norm,
#                 "prm_score_mean": float(chain.get("prm_score_mean", 0.0)),
#                 "prm_score_prod": float(chain.get("prm_score_prod", 0.0)),
#                 "n_steps": len(chain.get("steps", []))
#             }
#         }
#         # PRM detailed example with step-wise targets (normalized 0..1)
#         step_scores_norm = chain.get("step_scores_norm", None)
#         if step_scores_norm is None:
#             # derive from raw if provided
#             raw = chain.get("step_scores_raw", [])
#             if raw:
#                 step_scores_norm = [float(s)/10.0 for s in raw]
#             else:
#                 step_scores_norm = []
#         prm_example = {
#             "qid": qid,
#             "chain_id": cid,
#             "prompt": PROMPT_TEMPLATE.format(question=question, chain_text=chain_text, final_answer=final_answer),
#             "step_targets": step_scores_norm,
#             "meta": orm_example["meta"]
#         }
#         seq_example = None
#         if include_prm_seq:
#             seq_example = {
#                 "qid": qid,
#                 "chain_id": cid,
#                 "prompt": SEQ_PROMPT_TEMPLATE.format(question=question, chain_text=chain_text),
#                 "prm_mean": float(chain.get("prm_score_mean", 0.0)),
#                 "meta": orm_example["meta"]
#             }
#         yield orm_example, prm_example, seq_example
#
#
# def load_split_qids(split_path: Path):
#     # split files may be either full JSONL with records or plain list; support both
#     if not split_path.exists():
#         return None
#     qids = set()
#     # infer type by reading some lines
#     with open(split_path, "r", encoding="utf8") as f:
#         first = f.readline().strip()
#         if not first:
#             return None
#         try:
#             obj = json.loads(first)
#             # if it has 'id' it's a record
#             if isinstance(obj, dict) and "id" in obj:
#                 # read all ids
#                 f.seek(0)
#                 for line in f:
#                     if not line.strip():
#                         continue
#                     rec = json.loads(line)
#                     qids.add(rec.get("id"))
#             else:
#                 # fallback: treat file as newline ids
#                 qids.add(first)
#                 for line in f:
#                     if line.strip():
#                         qids.add(line.strip())
#         except Exception:
#             # treat file as newline ids
#             qids.add(first)
#             for line in f:
#                 if line.strip():
#                     qids.add(line.strip())
#     return qids
#
#
# def main(args):
#     input_path = Path(args.input)
#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     qids_train = load_split_qids(Path(args.train_split)) if args.train_split else None
#     qids_val = load_split_qids(Path(args.val_split)) if args.val_split else None
#     qids_test = load_split_qids(Path(args.test_split)) if args.test_split else None
#
#     orm_out = {"train": [], "val": [], "test": []}
#     prm_out = {"train": [], "val": [], "test": []}
#     prm_seq_out = {"train": [], "val": [], "test": []}
#
#     # load input and partition examples by the split qid membership
#     for rec in load_jsonl(input_path):
#         qid = rec.get("id")
#         target = None
#         if qids_train and qid in qids_train:
#             target = "train"
#         elif qids_val and qid in qids_val:
#             target = "val"
#         elif qids_test and qid in qids_test:
#             target = "test"
#         else:
#             # if splits not provided or qid not found, random assign with deterministic seed
#             if args.train_split is None and args.val_split is None and args.test_split is None:
#                 # deterministic pseudo-random via qid hash
#                 r = (hash(qid) % 131)
#                 if r < args.train_pct:
#                     target = "train"
#                 elif r < args.train_pct + args.val_pct:
#                     target = "val"
#                 else:
#                     target = "test"
#             else:
#                 # qid missing in any split: default to train (we'll log later)
#                 target = "train"
#
#         for orm_ex, prm_ex, seq_ex in extract_examples_from_record(rec, include_prm_seq=True):
#             orm_out[target].append(orm_ex)
#             prm_out[target].append(prm_ex)
#             prm_seq_out[target].append(seq_ex)
#
#     # write outputs
#     write_jsonl(outdir / "orm_train.jsonl", orm_out["train"])
#     write_jsonl(outdir / "orm_val.jsonl", orm_out["val"])
#     write_jsonl(outdir / "orm_test.jsonl", orm_out["test"])
#
#     write_jsonl(outdir / "prm_train.jsonl", prm_out["train"])
#     write_jsonl(outdir / "prm_val.jsonl", prm_out["val"])
#     write_jsonl(outdir / "prm_test.jsonl", prm_out["test"])
#
#     write_jsonl(outdir / "prm_train_seq.jsonl", prm_seq_out["train"])
#     write_jsonl(outdir / "prm_val_seq.jsonl", prm_seq_out["val"])
#     write_jsonl(outdir / "prm_test_seq.jsonl", prm_seq_out["test"])
#
#     # summary
#     summary = {
#         "orm_train": len(orm_out["train"]),
#         "orm_val": len(orm_out["val"]),
#         "orm_test": len(orm_out["test"]),
#         "prm_train": len(prm_out["train"]),
#         "prm_val": len(prm_out["val"]),
#         "prm_test": len(prm_out["test"]),
#     }
#     print("Wrote processed datasets:", summary)
#
#
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", type=str, required=True)
#     p.add_argument("--outdir", type=str, default="data/processed")
#     p.add_argument("--train_split", type=str, default=None)
#     p.add_argument("--val_split", type=str, default=None)
#     p.add_argument("--test_split", type=str, default=None)
#     p.add_argument("--train_pct", type=int, default=80)
#     p.add_argument("--val_pct", type=int, default=10)
#     p.add_argument("--test_pct", type=int, default=10)
#     args = p.parse_args()
#     main(args)


#!/usr/bin/env python3
"""
prepare_rm_datasets.py
──────────────────────────────────────────────
Stage 3.2.1 — Dataset splitting for Reward Model (ORM + PRM)

Features:
• Reads full labeled dataset (e.g., gsm8k_training_chains_1000_FINAL_gold_data.jsonl)
• Uses best deterministic hash modulus (auto-selected from optimizer results)
• Creates train/val/test JSONL splits for both ORM & PRM training
• Logs statistics and integrity summary

Output directory: data/processed/
"""

import json, numpy as np, random
from pathlib import Path
from datetime import datetime, UTC
from tqdm import tqdm

# Input full dataset (post-labeling, calibration, augmentation)
INPUT = "data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl"
OUTDIR = "data/processed"

OPTIMIZER_RESULTS = "data/processed/split_optimizer_results.json"
DEFAULT_MOD = 137  # fallback if optimizer file missing


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_best_modulus(path=OPTIMIZER_RESULTS):
    """Pick the best modulus from optimizer output (or fallback)."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data and isinstance(data, list):
            mod = int(data[0]["mod"])
            print(f"[INFO] Using recommended modulus from optimizer: mod={mod}")
            return mod
    except Exception:
        pass
    print(f"[WARN] No optimizer results found, using fallback mod={DEFAULT_MOD}")
    return DEFAULT_MOD


def hash_split(qid, mod):
    """Deterministic split using hash modulus."""
    r = hash(qid) % mod
    if r < int(0.8 * mod):   # 80%
        return "train"
    elif r < int(0.9 * mod): # 10%
        return "val"
    else:                    # 10%
        return "test"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_prm_records(rec):
    """Flatten question → step-scored PRM records."""
    out = []
    qid = rec["id"]
    for chain in rec.get("generated", []):
        if not chain.get("step_scores_norm"):
            continue
        prm_mean = float(np.mean(chain["step_scores_norm"]))
        out.append({
            "qid": qid,
            "chain_id": chain.get("chain_id", str(random.randint(1, 9999))),
            "prompt": chain.get("raw_text") or "\n".join(chain.get("steps", [])),
            "step_targets": chain["step_scores_norm"],
            "meta": {
                "prm_score_mean": prm_mean,
                "orm_label": chain.get("chain_label", 0),
                "final_answer": chain.get("final_answer", ""),
                "gold_answer": rec.get("gold_answer", "")
            }
        })
    return out


def extract_orm_records(rec):
    """Flatten question → ORM-level binary labels."""
    out = []
    qid = rec["id"]
    for chain in rec.get("generated", []):
        out.append({
            "qid": qid,
            "chain_id": chain.get("chain_id", str(random.randint(1, 9999))),
            "prompt": chain.get("raw_text") or "\n".join(chain.get("steps", [])),
            "label": int(chain.get("chain_label", 0)),
            "meta": {
                "orm_score": chain.get("orm_score", float(chain.get("chain_label", 0))),
                "final_answer": chain.get("final_answer", ""),
                "gold_answer": rec.get("gold_answer", "")
            }
        })
    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    mod = load_best_modulus()
    ensure_dir(OUTDIR)

    prm_splits = {"train": [], "val": [], "test": []}
    orm_splits = {"train": [], "val": [], "test": []}

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Splitting dataset"):
            rec = json.loads(line)
            qid = rec.get("id")
            if not qid:
                continue
            split = hash_split(qid, mod)
            prm_splits[split].extend(extract_prm_records(rec))
            orm_splits[split].extend(extract_orm_records(rec))

    # Stats summary
    stats = {
        "timestamp": datetime.now(UTC).isoformat(),
        "modulus": mod,
        "counts": {s: len(prm_splits[s]) for s in prm_splits},
        "orm_label_ratios": {
            s: round(np.mean([r["label"] for r in orm_splits[s]]) if orm_splits[s] else 0.0, 3)
            for s in orm_splits
        },
        "prm_means": {
            s: round(np.mean([r["meta"]["prm_score_mean"] for r in prm_splits[s]]) if prm_splits[s] else 0.0, 3)
            for s in prm_splits
        },
    }

    if stats["prm_means"]["val"] > stats["prm_means"]["train"]:
        # swap 2% of high-PRM val examples into train
        high_val = sorted(prm_splits["val"], key=lambda x: np.mean(x["step_targets"]), reverse=True)
        move = high_val[:int(0.02 * len(high_val))]
        prm_splits["val"] = [r for r in prm_splits["val"] if r not in move]
        prm_splits["train"].extend(move)

    # Save outputs
    for split in ["train", "val", "test"]:
        prm_path = Path(OUTDIR) / f"prm_{split}.jsonl"
        orm_path = Path(OUTDIR) / f"orm_{split}.jsonl"
        with open(prm_path, "w", encoding="utf-8") as fp:
            for r in prm_splits[split]:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(orm_path, "w", encoding="utf-8") as fp:
            for r in orm_splits[split]:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(Path(OUTDIR) / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n✅ Split completed successfully!")
    print(json.dumps(stats, indent=2))
    print(f"\n📄 Saved to: {OUTDIR}/prm_[split].jsonl and orm_[split].jsonl")


if __name__ == "__main__":
    main()


