from collections import Counter, defaultdict
import json

path = "data/splits/gsm8k_rm_train_balanced_final.jsonl"
bins = defaultdict(int)
counts = Counter()
for l in open(path, "r"):
    j = json.loads(l)
    for c in j["generated"]:
        for s in c.get("step_scores_norm", []):
            b = round(s, 1)
            bins[b] += 1
            if s < 0.3: counts["low"] += 1
            elif s < 0.7: counts["mid"] += 1
            else: counts["high"] += 1

total = sum(bins.values())
print(f"Low:{counts['low']/total:.2%}, Mid:{counts['mid']/total:.2%}, High:{counts['high']/total:.2%}")

# if counts['low']/total < 10.0:
#     import random, json
#
#     INPUT = "data/splits/gsm8k_rm_train.jsonl"
#     OUTPUT = "data/splits/gsm8k_rm_train_aug.jsonl"
#     AUG_FACTOR = 0.2  # duplicate 20% of low-score chains
#
#     records = [json.loads(l) for l in open(INPUT) if l.strip()]
#     low_chains = [(r, c) for r in records for c in r["generated"]
#                   if any(s < 0.3 for s in c.get("step_scores_norm", []))]
#     to_add = random.sample(low_chains, int(len(low_chains) * AUG_FACTOR))
#
#     for r, c in to_add:
#         c_new = json.loads(json.dumps(c))
#         c_new["step_scores_norm"] = [min(1.0, max(0.0, x + random.uniform(-0.05, 0.05)))
#                                      for x in c_new["step_scores_norm"]]
#         c_new["step_scores_raw"] = [round(x * 10, 2) for x in c_new["step_scores_norm"]]
#         r["generated"].append(c_new)
#
#     with open(OUTPUT, "w", encoding="utf-8") as f:
#         for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
#
#     print("Augmented low-score chains ->", OUTPUT)

# import json, random
# from pathlib import Path
#
# INPUT = "data/splits/gsm8k_rm_train_aug.jsonl"
# OUTPUT = "data/splits/gsm8k_rm_train_balanced.jsonl"
# LOW_THRESH, MID_THRESH = 0.3, 0.7
# LOW_TARGET, MID_TARGET = 0.20, 0.35  # Target proportions
#
# data = [json.loads(l) for l in open(INPUT) if l.strip()]
#
# all_steps = [(r, c) for r in data for c in r["generated"]]
# low = [(r, c) for r, c in all_steps if any(s < LOW_THRESH for s in c.get("step_scores_norm", []))]
# mid = [(r, c) for r, c in all_steps if any(LOW_THRESH <= s < MID_THRESH for s in c.get("step_scores_norm", []))]
#
# def upsample(pool, factor):
#     new = []
#     for r, c in random.choices(pool, k=int(len(pool)*factor)):
#         c_new = json.loads(json.dumps(c))
#         c_new["step_scores_norm"] = [
#             min(1.0, max(0.0, s + random.uniform(-0.05, 0.05))) for s in c_new["step_scores_norm"]
#         ]
#         c_new["step_scores_raw"] = [round(s*10, 2) for s in c_new["step_scores_norm"]]
#         c_new["labeling_status"] = "augmented"
#         r["generated"].append(c_new)
#         new.append(c_new)
#     return new
#
# # Compute augmentation factors
# total_chains = len(all_steps)
# low_factor = max(0.0, (LOW_TARGET - len(low)/total_chains) / (len(low)/total_chains))
# mid_factor = max(0.0, (MID_TARGET - len(mid)/total_chains) / (len(mid)/total_chains))
#
# print(f"Low-factor: {low_factor:.2f}, Mid-factor: {mid_factor:.2f}")
# upsampled = []
# upsampled += upsample(low, low_factor)
# upsampled += upsample(mid, mid_factor)
#
# Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
# with open(OUTPUT, "w", encoding="utf-8") as f:
#     for r in data:
#         f.write(json.dumps(r, ensure_ascii=False) + "\n")
#
# print(f"[DONE] Added {len(upsampled)} new chains → {OUTPUT}")
