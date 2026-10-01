#!/usr/bin/env python3
"""
boost_low_prm_chains_fixed.py

Robust, safe augmentation to increase fraction of "low" PRM chains.
- First upsample (duplicate + small noise) from existing low pool (capped per question).
- Then degrade some mid/high chains (conservative) if still needed.
- Strong failsafes to avoid infinite loops.
- Marks augmented chains with labeling_status.

python scripts/boost_low_prm_chains.py \                                                                                    ─╯
  --input data/splits/gsm8k_rm_train_aug.jsonl \
  --output data/splits/gsm8k_rm_train_low_boosted.jsonl \
  --target_low 0.18 \
  --seed 42
"""

import argparse, json, random, copy, math, sys, time
from statistics import mean
from pathlib import Path

# === Configurable defaults ===
DEFAULT_NOISE_SCALE = 0.04
DEFAULT_LOW_THRESH = 0.30
DEFAULT_MAX_DUP_PER_Q = 4
DEFAULT_MAX_DEGRADE_FRACTION = 0.15
DEFAULT_TARGET_LOW = 0.18
DEFAULT_SEED = 42
DEFAULT_MAX_TOTAL_AUG = 20000   # hard stop
DEFAULT_MAX_ATTEMPTS_PER_PHASE = 200000

# === Helpers ===
def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def save_jsonl(path, recs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chain_avg_norm(chain):
    s = chain.get("step_scores_norm", [])
    return mean(s) if s else 0.0

def is_low(chain, low_thresh):
    return chain_avg_norm(chain) < low_thresh

def clone_with_noise(chain, noise_scale):
    c = copy.deepcopy(chain)
    s = c.get("step_scores_norm", [])
    if not s:
        return c
    new = []
    for x in s:
        delta = random.uniform(-noise_scale, noise_scale)
        nx = min(1.0, max(0.0, x + delta))
        new.append(nx)
    c["step_scores_norm"] = [float(x) for x in new]
    c["step_scores_raw"] = [round(x * 10.0, 4) for x in new]
    c["prm_score_mean"] = float(mean(new)) if new else 0.0
    c["prm_score_prod"] = float(math.prod(new)) if new else 0.0
    c["labeling_status"] = "augmented"
    return c

def degrade_to_low(chain, low_thresh):
    """Safely degrade chain to push average below low_thresh, with failsafe clamps."""
    c = copy.deepcopy(chain)
    s = c.get("step_scores_norm", [])
    if not s:
        return c
    n = len(s)
    new = s[:]
    max_attempts = 5
    attempts = 0
    while mean(new) >= low_thresh and attempts < max_attempts:
        k = max(1, int(0.4 * n))
        idxs = random.sample(range(n), k)
        for i in idxs:
            delta = random.uniform(0.15, 0.40)
            new[i] = max(0.0, new[i] - delta)
        attempts += 1
    # Failsafe: if still not low, clamp a subset to very low range
    if mean(new) >= low_thresh:
        for i in range(n):
            # lower any value above low_thresh aggressively but not to zero
            if new[i] > low_thresh + 0.05:
                new[i] = max(0.0, new[i] - random.uniform(0.25, 0.5))
    # final clamp safety
    new = [min(1.0, max(0.0, float(x))) for x in new]
    c["step_scores_norm"] = new
    c["step_scores_raw"] = [round(x * 10.0, 4) for x in new]
    c["prm_score_mean"] = float(mean(new)) if new else 0.0
    c["prm_score_prod"] = float(math.prod(new)) if new else 0.0
    c["labeling_status"] = "augmented_degraded"
    return c

# === Main process ===
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target_low", type=float, default=DEFAULT_TARGET_LOW)
    p.add_argument("--low_thresh", type=float, default=DEFAULT_LOW_THRESH)
    p.add_argument("--noise_scale", type=float, default=DEFAULT_NOISE_SCALE)
    p.add_argument("--max_dup_per_q", type=int, default=DEFAULT_MAX_DUP_PER_Q)
    p.add_argument("--max_degrade_fraction", type=float, default=DEFAULT_MAX_DEGRADE_FRACTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_total_aug", type=int, default=DEFAULT_MAX_TOTAL_AUG)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)

    recs = load_jsonl(args.input)
    # Flatten chains for counts
    all_chains = []
    for rec in recs:
        for c in rec.get("generated", []):
            all_chains.append((rec, c))
    total = len(all_chains)
    if total == 0:
        print("No chains found in input; exiting.")
        return

    low_pool = [(r,c) for (r,c) in all_chains if is_low(c, args.low_thresh)]
    mid_pool = [(r,c) for (r,c) in all_chains if not is_low(c, args.low_thresh) and chain_avg_norm(c) < 0.7]
    high_pool = [(r,c) for (r,c) in all_chains if chain_avg_norm(c) >= 0.7]

    cur_low = len(low_pool)
    target_low_cnt = int(round(args.target_low * total))
    need = max(0, target_low_cnt - cur_low)
    print(f"Total chains={total}; cur_low={cur_low}; target_low={target_low_cnt}; need_add={need}")
    if need == 0:
        print("Already at or above target low fraction. Nothing to do.")
        save_jsonl(args.output, recs)
        return

    # Safety trackers
    augmented = 0
    per_q_dup = {}  # qid -> dup count

    # PHASE 1: Upsample low pool (duplicates with noise)
    print("Phase 1: Upsample existing low chains (duplicate + noise).")
    if len(low_pool) == 0:
        print("No low chains available to upsample.")
    else:
        attempts = 0
        # Shuffle candidate list once
        candidates = low_pool.copy()
        random.shuffle(candidates)
        # iterate over candidates in round-robin, but bounded attempts
        i = 0
        while need > 0 and attempts < DEFAULT_MAX_ATTEMPTS_PER_PHASE and augmented < args.max_total_aug:
            rec, chain = candidates[i % len(candidates)]
            qid = rec["id"]
            if per_q_dup.get(qid, 0) < args.max_dup_per_q:
                newc = clone_with_noise(chain, args.noise_scale)
                rec["generated"].append(newc)
                per_q_dup[qid] = per_q_dup.get(qid, 0) + 1
                augmented += 1
                need -= 1
            # increment indices and attempts
            i += 1
            attempts += 1
            # if we've looped more than N times without progress and per-q caps reached, break
            if attempts % 10000 == 0:
                # recompute available candidates that still can be duplicated
                candidates = [(r,c) for (r,c) in low_pool if per_q_dup.get(r["id"], 0) < args.max_dup_per_q]
                if not candidates:
                    print("No more duplicable low candidates (per-question caps reached). Breaking upsample phase.")
                    break
        print(f"Phase 1 ended: augmented={augmented}, need_remaining={need}, attempts={attempts}")

    # PHASE 2: Degrade some mid/high chains if still need
    if need > 0:
        print("Phase 2: Degrade mid/high chains to low until need satisfied (conservative).")
        pool = mid_pool + high_pool
        random.shuffle(pool)
        max_degrade = int(len(pool) * args.max_degrade_fraction)
        degraded = 0
        attempts = 0
        for rec, chain in pool:
            if need <= 0 or degraded >= max_degrade or augmented >= args.max_total_aug:
                break
            # degrade and append clone
            newc = degrade_to_low(chain, args.low_thresh)
            # safety: ensure the degradation actually made it low
            if chain_avg_norm(newc) >= args.low_thresh:
                # degrade didn't succeed; skip
                attempts += 1
                if attempts > DEFAULT_MAX_ATTEMPTS_PER_PHASE:
                    print("Too many failed degrade attempts; breaking.")
                    break
                continue
            rec["generated"].append(newc)
            degraded += 1
            augmented += 1
            need -= 1
        print(f"Phase 2 ended: degraded={degraded}, need_remaining={need}")

    # Final summary & save
    # compute final distribution
    final_chains = [(r,c) for r in recs for c in r.get("generated", [])]
    final_total = len(final_chains)
    low_final = sum(1 for (r,c) in final_chains if is_low(c, args.low_thresh))
    mid_final = sum(1 for (r,c) in final_chains if not is_low(c, args.low_thresh) and chain_avg_norm(c) < 0.7)
    high_final = final_total - low_final - mid_final
    print(f"Final totals: total={final_total}, low={low_final}, mid={mid_final}, high={high_final}")
    print(f"Final pct: low={low_final/final_total:.3f}, mid={mid_final/final_total:.3f}, high={high_final/final_total:.3f}")
    print(f"Augmented total chains added: {augmented}")

    if args.dry_run:
        print("Dry run mode: not writing output.")
        return

    save_jsonl(args.output, recs)
    print("Saved balanced dataset to:", args.output)
    if need > 0:
        print(f"WARNING: Could not fully satisfy need. need_remaining={need}. Consider lowering per-q cap or increasing degrade fraction.")

if __name__ == "__main__":
    main()
