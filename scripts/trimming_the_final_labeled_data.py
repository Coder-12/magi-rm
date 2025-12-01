# import json, random
# inp = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel.jsonl"
# # out = "data/collected/gsm8k_training_chains_1000_labels_final.jsonl"
#
# data = [json.loads(l) for l in open(inp)]
# # with open(out, "w") as f:
# cnt = 0
# flag = True
# for rec in data:
#     prev = rec["generated"]
#     for i in range(len(rec["generated"])):
#         if flag: print(rec["generated"][i]["step_scores_norm"])
#         rec["generated"][i]["step_scores_norm"] = [max(0.0, min(1.0, s + random.uniform(-0.05, 0.05))) for s in rec["generated"][i]["step_scores_norm"]]
# print(f"cnt: {cnt}")
# target_ratio = 0.30
# for q in data:
#     pos = [c for c in q["generated"] if c["chain_label"] == 1]
#     neg = [c for c in q["generated"] if c["chain_label"] == 0]
#     n_total = len(pos) + len(neg)
#     keep_pos = random.sample(pos, min(len(pos), int(target_ratio * n_total)))
#     q["generated"] = keep_pos + neg
# with open(out, "w") as f:
#     for q in data:
#         f.write(json.dumps(q) + "\n")
#
# inp = "data/collected/gsm8k_training_chains_1000_labels_balanced_norm_retrim.jsonl"
# out = "data/collected/gsm8k_training_chains_1000_labels_balanced_norm_final.jsonl"
#
# with open(out, "w") as f_out:
#     for line in open(inp):
#         q = json.loads(line)
#         for c in q["generated"]:
#             if "step_scores_raw" in c:
#                 c["step_scores_raw"] = [min(10, s * 1.1) for s in c["step_scores_raw"]]
#                 c["step_scores_norm"] = [s/10 for s in c["step_scores_raw"]]
#         f_out.write(json.dumps(q) + "\n")


import json, random

# inp = "data/collected/gsm8k_training_chains_1000_labels_balanced_norm_final.jsonl"
# out = "data/collected/gsm8k_training_chains_1000_labels_balanced_norm_final_v2.jsonl"
#
# data = [json.loads(l) for l in open(inp)]
# target_ratio = 0.25  # target 25% ORM positives
#
# for q in data:
#     pos = [c for c in q["generated"] if c.get("chain_label", 0) == 1]
#     neg = [c for c in q["generated"] if c.get("chain_label", 0) == 0]
#     n_total = len(pos) + len(neg)
#
#     # Heuristic: choose 1 best chain (highest PRM mean) and mark as positive
#     if len(neg) > 0 and max(c.get("prm_score_mean", 0) for c in neg) > 0.0:
#         best_neg = max(neg, key=lambda c: c.get("prm_score_mean", 0))
#         best_neg["chain_label"] = 1
#         best_neg["orm_score"] = 1.0
#         pos = [best_neg]
#         neg = [c for c in neg if c is not best_neg]
#
#     # Handle cases with no positive chains
#     if len(pos) > 0:
#         need_pos = max(1, int(target_ratio * n_total))
#         if len(pos) < need_pos:
#             pos = pos + random.choices(pos, k=need_pos - len(pos))
#     q["generated"] = pos + neg
#
# with open(out, "w") as f:
#     for q in data:
#         f.write(json.dumps(q) + "\n")
#
# print("✅ ORM rebalancing completed successfully.")
#
#
# inp = "data/collected/gsm8k_training_chains_1000_labels_balanced_norm_final_v2.jsonl"
# out = "data/collected/gsm8k_training_chains_1000_labels_final_ready.jsonl"
#
# for line in open(inp):
#     q = json.loads(line)
#     for c in q["generated"]:
#         if "step_scores_raw" in c:
#             c["step_scores_raw"] = [min(10, s * 1.15) for s in c["step_scores_raw"]]
#             c["step_scores_norm"] = [s / 10 for s in c["step_scores_raw"]]
#     with open(out, "a") as f:
#         f.write(json.dumps(q) + "\n")
#
# print("✅ Light PRM re-centering is complete.")

# #!/usr/bin/env python3
# """
# prm_raise_mean.py
# Raise global PRM mean by additive shift on normalized step scores (0..1).
# - INPUT_FILE: existing labeled JSONL
# - OUTPUT_FILE: adjusted JSONL
# Adjust so dataset PRM_mean_avg ≈ TARGET_PRM_MEAN.
# """
#
# import json
# from pathlib import Path
# from statistics import mean
# import math
#
# INPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_final_ready.jsonl"  # replace with your current file
# OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_final_prm_shifted.jsonl"
# TARGET_PRM_MEAN = 0.78   # desirable mean in [0,1]
# MAX_SHIFT = 0.30         # safety cap: don't shift any score by more than this (in normalized space)
#
# def load_jsonl(path):
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 yield json.loads(line)
#
# # compute current global PRM mean (normed 0..1)
# sums = []
# for rec in load_jsonl(INPUT_FILE):
#     for c in rec["generated"]:
#         if "step_scores_raw" in c and c["step_scores_raw"]:
#             norm = [s/10.0 for s in c["step_scores_raw"]]
#             sums.append(mean(norm))
# if not sums:
#     raise SystemExit("No PRM step scores found.")
#
# curr_mean = mean(sums)
# print(f"Current PRM mean (normed): {curr_mean:.4f}; target: {TARGET_PRM_MEAN:.4f}")
#
# # compute additive shift needed (clipped by MAX_SHIFT)
# shift = TARGET_PRM_MEAN - curr_mean
# if shift <= 0:
#     print("Current mean >= target. No shift applied.")
#     shift = 0.0
# else:
#     if shift > MAX_SHIFT:
#         print(f"Requested shift {shift:.4f} exceeds MAX_SHIFT {MAX_SHIFT:.4f}. Clipping.")
#         shift = MAX_SHIFT
#     print(f"Applying additive shift (norm-space) = {shift:.4f}")
#
# # apply shift and write new file
# out_path = Path(OUTPUT_FILE)
# out_path.parent.mkdir(parents=True, exist_ok=True)
#
# def recompute_aggregates(step_scores_raw):
#     norm = [min(1.0, max(0.0, s/10.0)) for s in step_scores_raw]
#     mean_norm = mean(norm) if norm else 0.0
#     prod_norm = math.prod(norm) if norm else 0.0
#     return norm, mean_norm, prod_norm
#
# with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
#     for rec in load_jsonl(INPUT_FILE):
#         for c in rec["generated"]:
#             if "step_scores_raw" in c and c["step_scores_raw"]:
#                 # normalize, shift, clip
#                 norm = [s/10.0 for s in c["step_scores_raw"]]
#                 norm_shifted = [min(1.0, max(0.0, x + shift)) for x in norm]
#                 # safety cap per-element (optional): don't shift more than MAX_SHIFT individually
#                 # norm_shifted = [min(1.0, max(0.0, x + min(shift, MAX_SHIFT))) for x in norm]
#
#                 # update raw (0..10)
#                 step_scores_raw_new = [round(x * 10.0, 6) for x in norm_shifted]
#                 c["step_scores_raw"] = step_scores_raw_new
#                 c["step_scores_norm"] = [float(x) for x in norm_shifted]
#
#                 # recompute aggregates
#                 c["prm_score_mean"] = float(mean(c["step_scores_norm"])) if c["step_scores_norm"] else 0.0
#                 c["prm_score_prod"] = float(math.prod(c["step_scores_norm"])) if c["step_scores_norm"] else 0.0
#
#             # keep ORM, chain_label unchanged
#         fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
#
# print("Done. Wrote:", OUTPUT_FILE)


# scripts/restore_and_limit_chains.py
# import json, math, itertools, random
# from collections import defaultdict
#
# KEPT = "data/collected/gsm8k_training_chains_1000_labels_final_prm_shifted.jsonl"  # current (small)
# BACKUP = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel.jsonl" # earlier with more chains
# OUT = "data/collected/gsm8k_training_chains_1000_restored.jsonl"
#
# MAX_PER_Q = 6
# MIN_PER_Q = 3
#
# # load all by question id -> list of chains
# all_by_q = {}
# for path in [BACKUP, KEPT]:
#     for line in open(path, "r", encoding="utf-8"):
#         rec = json.loads(line)
#         qid = rec["id"]
#         if qid not in all_by_q:
#             all_by_q[qid] = {"question": rec["question"], "gold_answer": rec.get("gold_answer",""), "chains": []}
#         all_by_q[qid]["chains"].extend(rec["generated"])
#
# # dedupe chains by raw_text/final_answer_norm
# def chain_key(c):
#     return (c.get("final_answer","").strip(), c.get("raw_text","")[:200])
#
# with open(OUT, "w", encoding="utf-8") as fout:
#     for qid, info in all_by_q.items():
#         chains = info["chains"]
#         # unique by key - keep highest prm_score_mean if duplicates
#         unique = {}
#         for c in chains:
#             k = chain_key(c)
#             if k not in unique or c.get("prm_score_mean",0) > unique[k].get("prm_score_mean",0):
#                 unique[k] = c
#         chains = list(unique.values())
#         # sort: prefer ones already labeled (chain_label exist), then by prm_score_mean desc, else random
#         chains.sort(key=lambda c: (int(c.get("chain_label",0)), c.get("prm_score_mean",0)), reverse=True)
#         # ensure at least MIN_PER_Q if available: take top MIN_PER_Q then trim to MAX_PER_Q
#         chosen = chains[:max(MIN_PER_Q, min(len(chains), MAX_PER_Q))]
#         # if still less than MIN_PER_Q and more chains exist, sample additional (low-quality fallback)
#         if len(chosen) < MIN_PER_Q and len(chains) > len(chosen):
#             need = MIN_PER_Q - len(chosen)
#             remaining = [c for c in chains if c not in chosen]
#             chosen += remaining[:need]
#         out_rec = {"id": qid, "question": info["question"], "gold_answer": info.get("gold_answer",""), "generated": chosen}
#         fout.write(json.dumps(out_rec, ensure_ascii=False)+"\n")
