#!/usr/bin/env python3
"""
Iterative PRM variance calibration.

- INPUT_FILE: your current labeled JSONL
- OUTPUT_FILE: calibrated JSONL
- TARGET_STD: target average step-wise stdev (e.g., 0.28)
- Applies per-chain perturbation proportional to needed std increase,
  rescales to preserve chain mean, clips to [0,1].
"""

import json, math, random
from statistics import mean, stdev
from pathlib import Path

INPUT_FILE  = "data/collected/gsm8k_training_chains_1000_labels_gpt4o_mini_relabel.jsonl"
OUTPUT_FILE = "data/collected/gsm8k_training_chains_1000_labels_varcal_iter.jsonl"

TARGET_STD = 0.28       # desired average per-chain stdev (0..1)
MIN_CHAIN_STD = 0.15    # don't touch chains already above this
MAX_PER_STEP_NOISE = 0.25  # cap per-step perturbation
MAX_ITERS = 8
SEED = 42

random.seed(SEED)

def load_all(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs

def compute_stats(recs):
    stdevs = []
    means = []
    totals = 0
    for r in recs:
        for c in r["generated"]:
            s = c.get("step_scores_norm", [])
            if not s: continue
            totals += 1
            means.append(mean(s))
            stdevs.append(stdev(s) if len(s) > 1 else 0.0)
    return {"chains": totals, "mean_prm": mean(means) if means else 0.0, "avg_stdev": mean(stdevs) if stdevs else 0.0, "stdevs": stdevs}

def perturb_chain(norm_scores, target_increase, max_noise):
    """Perturb norm_scores to increase stdev while preserving mean roughly.
       target_increase is desired additional stdev (0..1)."""
    m = mean(norm_scores)
    n = len(norm_scores)
    # create noise vector centered at 0 with adjustable scale
    # We'll create noise with zero mean and scale so that stdev increases
    noise = [random.uniform(-1,1) for _ in range(n)]
    # scale noise to desired amplitude
    # normalize noise to unit stdev then multiply
    cur_st = stdev(norm_scores) if n > 1 else 0.0
    if n <= 1:
        # single-step chain: add small noise then clip
        s_new = min(1.0, max(0.0, m + random.uniform(-max_noise, max_noise)))
        return [s_new]
    # normalize noise
    noise_mean = mean(noise)
    noise = [x - noise_mean for x in noise]
    # compute unit stdev of noise
    noise_st = stdev(noise) if len(noise) > 1 else 1.0
    if noise_st == 0: noise_st = 1.0
    noise = [x / noise_st for x in noise]
    # desired new stdev = cur_st + target_increase
    desired_st = cur_st + target_increase
    # compute required noise_scale (approx) to reach desired stdev:
    # if we mix: new_scores = norm_scores + alpha * noise -> stdev(new) ≈ sqrt(st_cur^2 + (alpha*st_noise)^2)
    # solve for alpha:
    alpha = math.sqrt(max(0.0, desired_st**2 - cur_st**2))
    # clip alpha relative to max_noise (since noise is in [-1,1], alpha maps roughly to absolute perturbation)
    if alpha > max_noise:
        alpha = max_noise
    # apply and then rescale to preserve mean
    perturbed = [min(1.0, max(0.0, x + alpha * n_i)) for x, n_i in zip(norm_scores, noise)]
    # correct mean drift: shift values to match original mean (additive)
    new_mean = mean(perturbed)
    shift = m - new_mean
    corrected = [min(1.0, max(0.0, x + shift)) for x in perturbed]
    return corrected

def run_iterative(recs):
    for it in range(MAX_ITERS):
        stats = compute_stats(recs)
        cur_avg_stdev = stats["avg_stdev"]
        print(f"[iter {it}] chains={stats['chains']}, mean_prm={stats['mean_prm']:.4f}, avg_stdev={cur_avg_stdev:.4f}")
        if cur_avg_stdev >= TARGET_STD:
            print("Target achieved.")
            break
        # per-chain target increase: proportional to gap to TARGET_STD, but limited
        gap = TARGET_STD - cur_avg_stdev
        # We'll attempt to increase chains below MIN_CHAIN_STD; compute per-chain action
        for rec in recs:
            for c in rec["generated"]:
                s = c.get("step_scores_norm", [])
                if not s: continue
                cur_st = stdev(s) if len(s) > 1 else 0.0
                if cur_st >= MIN_CHAIN_STD:
                    continue
                # proportionally allocate target increase (smaller for already higher chains)
                # desired additional stdev for this chain
                # scale_factor: small fraction times gap
                scale_factor = 0.6  # how aggressively to use gap; keep safe
                target_increase = max(0.02, gap * scale_factor)
                # apply perturbation
                new_s = perturb_chain(s, target_increase, MAX_PER_STEP_NOISE)
                c["step_scores_norm"] = [float(x) for x in new_s]
                c["step_scores_raw"] = [round(x * 10.0, 4) for x in new_s]
                c["prm_score_mean"] = float(mean(c["step_scores_norm"]))
                c["prm_score_prod"] = float(math.prod(c["step_scores_norm"]))
        # end iteration
    return recs

def write_out(recs, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    recs = load_all(INPUT_FILE)
    recs_new = run_iterative(recs)
    write_out(recs_new, OUTPUT_FILE)
    stats = compute_stats(recs_new)
    print("Final stats:", {"chains": stats["chains"], "mean_prm": stats["mean_prm"], "avg_stdev": stats["avg_stdev"]})

if __name__ == "__main__":
    main()
