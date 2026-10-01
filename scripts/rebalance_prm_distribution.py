#!/usr/bin/env python3
"""
rebalance_prm_distribution.py

Safe dataset-side rebalance for PRM step-score bins:
- Upsample low/mid chains (duplicate + small noise).
- Degrade a fraction of high chains into mid (softly).
- Mark augmented chains with labeling_status = "augmented".
- Preserve ORM chain_label and final_answer.
"""

import json, random, math, copy
from pathlib import Path
from statistics import mean

# === CONFIG ===
INPUT = "data/splits/gsm8k_rm_train_aug.jsonl"
OUTPUT = "data/splits/gsm8k_rm_train_balanced_final.jsonl"
SEED = 42
random.seed(SEED)

TARGET_LOW = 0.18   # 18%
TARGET_MID = 0.32   # 32%
TARGET_HIGH = 0.50  # 50%

LOW_THRESH = 0.3
MID_THRESH = 0.7

# augmentation caps
MAX_DUPLICATE_PER_Q = 4     # avoid explosive duplicates per question
MAX_DEGRADE_PROPORTION = 0.25  # at most degrade 25% of high chains

NOISE_LOW = 0.04    # for upsampled low chains: ±noise
NOISE_MID = 0.03
DEGRADE_RANGE = ( -0.18, -0.06 )  # subtract to degrade a few step scores (norm space)

# === HELPERS ===
def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs

def save_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chain_bin(chain):
    scores = chain.get("step_scores_norm", [])
    if not scores:
        return "mid"
    avg = mean(scores)
    if avg < LOW_THRESH:
        return "low"
    if avg < MID_THRESH:
        return "mid"
    return "high"

def clone_and_perturb_chain(chain, noise_scale):
    c = copy.deepcopy(chain)
    s = c.get("step_scores_norm", [])
    if not s:
        return c
    new = []
    for x in s:
        # add small noise (clipped)
        delta = random.uniform(-noise_scale, noise_scale)
        nx = min(1.0, max(0.0, x + delta))
        new.append(nx)
    c["step_scores_norm"] = [float(x) for x in new]
    c["step_scores_raw"] = [round(x * 10.0, 4) for x in new]
    c["prm_score_mean"] = float(mean(new)) if new else 0.0
    c["prm_score_prod"] = float(math.prod(new)) if new else 0.0
    c["labeling_status"] = "augmented"
    return c

def degrade_high_to_mid(chain):
    # Softly reduce some step scores to move avg into mid-range
    c = copy.deepcopy(chain)
    s = c.get("step_scores_norm", [])
    if not s:
        return c
    n = len(s)
    # choose 30-60% of steps (prefer longer chains) to degrade
    k = max(1, int(min(n, max(1, round(n * random.uniform(0.3,0.6))))))
    idxs = random.sample(range(n), k)
    new = s[:]
    for i in idxs:
        delta = random.uniform(DEGRADE_RANGE[0], DEGRADE_RANGE[1])  # negative
        new[i] = min(1.0, max(0.0, new[i] + delta))
    # if avg still >= MID_THRESH, degrade slightly more
    avg = mean(new)
    iter_count = 0
    while avg >= MID_THRESH and iter_count < 3:
        i = random.choice(list(range(n)))
        new[i] = min(1.0, max(0.0, new[i] + random.uniform(DEGRADE_RANGE[0], DEGRADE_RANGE[1])))
        avg = mean(new); iter_count += 1
    c["step_scores_norm"] = [float(x) for x in new]
    c["step_scores_raw"] = [round(x*10.0,4) for x in new]
    c["prm_score_mean"] = float(mean(new))
    c["prm_score_prod"] = float(math.prod(new))
    c["labeling_status"] = "augmented_degraded"
    return c

# === MAIN ===
def rebalance():
    recs = load_jsonl(INPUT)
    all_chains = []
    for r in recs:
        for c in r["generated"]:
            all_chains.append((r["id"], r, c))

    total = len(all_chains)
    # categorize
    bins = {"low": [], "mid": [], "high": []}
    for qid, r, c in all_chains:
        b = chain_bin(c)
        bins[b].append((qid, r, c))

    cur_low = len(bins["low"]) / total
    cur_mid = len(bins["mid"]) / total
    cur_high = len(bins["high"]) / total
    print(f"Current: Low={cur_low:.3f}, Mid={cur_mid:.3f}, High={cur_high:.3f} (total={total})")

    # compute target counts
    target_low_cnt = int(round(TARGET_LOW * total))
    target_mid_cnt = int(round(TARGET_MID * total))
    target_high_cnt = int(round(TARGET_HIGH * total))
    print("Targets (counts):", target_low_cnt, target_mid_cnt, target_high_cnt)

    augmented = 0
    # 1) Upsample low and mid (duplicate some chains)
    def upsample_pool(pool, target_cnt, noise_scale):
        nonlocal augmented
        cur_cnt = len(pool)
        need = max(0, target_cnt - cur_cnt)
        if need == 0:
            return []
        # cap per-question duplicates
        augmented_items = []
        pool_choices = pool.copy()
        random.shuffle(pool_choices)
        idx = 0
        while need > 0 and pool_choices:
            qid, r, c = pool_choices[idx % len(pool_choices)]
            # don't duplicate more than MAX_DUPLICATE_PER_Q times from same q
            times_from_q = sum(1 for _,rr,cc in augmented_items if rr["id"] == qid)
            if times_from_q < MAX_DUPLICATE_PER_Q:
                new_c = clone_and_perturb_chain(c, noise_scale)
                # append cloned chain to question's generated list
                r["generated"].append(new_c)
                augmented_items.append((qid, r, new_c))
                augmented += 1
                need -= 1
            idx += 1
            if idx > 100000:
                break
        return augmented_items

    # upsample low
    upsample_pool(bins["low"], target_low_cnt, NOISE_LOW)
    # upsample mid
    upsample_pool(bins["mid"], target_mid_cnt, NOISE_MID)

    # recompute totals after upsampling
    all_chains = []
    for r in recs:
        for c in r["generated"]:
            all_chains.append((r["id"], r, c))
    total2 = len(all_chains)
    bins2 = {"low": [], "mid": [], "high": []}
    for qid, r, c in all_chains:
        bins2[chain_bin(c)].append((qid, r, c))

    print(f"After upsampling: total={total2}; low={len(bins2['low'])}, mid={len(bins2['mid'])}, high={len(bins2['high'])}")

    # 2) Degrade some fraction of high -> mid if still needed
    # compute current counts and deficits
    cur_low2 = len(bins2["low"]); cur_mid2 = len(bins2["mid"]); cur_high2 = len(bins2["high"])
    need_low2 = max(0, target_low_cnt - cur_low2)
    need_mid2 = max(0, target_mid_cnt - cur_mid2)
    # degrade up to MAX_DEGRADE_PROPORTION of high pool
    high_pool = bins2["high"].copy()
    random.shuffle(high_pool)
    max_degrade = int(len(high_pool) * MAX_DEGRADE_PROPORTION)
    degrade_count = min(max_degrade, need_mid2 + max(0, need_low2))
    degraded = 0
    idx = 0
    while degraded < degrade_count and idx < len(high_pool):
        qid, r, c = high_pool[idx]
        # degrade only if degrated chain won't change final answer (we don't change final_answer)
        new_c = degrade_high_to_mid(c)
        # ensure we didn't accidentally drop mean below 0 (safe)
        r["generated"].append(new_c)
        degraded += 1
        augmented += 1
        idx += 1

    print(f"Degraded {degraded} high chains into mid-ish chains (added)")

    # final recompute
    final_chains = []
    for r in recs:
        for c in r["generated"]:
            final_chains.append((r["id"], r, c))
    total_final = len(final_chains)
    final_bins = {"low": [], "mid": [], "high": []}
    for qid, r, c in final_chains:
        final_bins[chain_bin(c)].append((qid, r, c))

    print(f"Final totals: {total_final} chains")
    print(f"Final distribution (counts): low={len(final_bins['low'])}, mid={len(final_bins['mid'])}, high={len(final_bins['high'])}")
    print(f"Final distribution (pct): low={len(final_bins['low'])/total_final:.3f}, mid={len(final_bins['mid'])/total_final:.3f}, high={len(final_bins['high'])/total_final:.3f}")
    print(f"Augmented total chains added: {augmented}")

    save_jsonl(OUTPUT, recs)
    print("Saved balanced dataset to:", OUTPUT)

if __name__ == "__main__":
    rebalance()
