#!/usr/bin/env python3
"""
inspect_rm_splits.py
──────────────────────────────────────────────
Stage 3.2.2 — Inspect & Visualize PRM + ORM splits

Analyzes train/val/test splits produced by Stage 3.2.1
and generates a detailed statistical summary with
optional visual histograms (PRM score, ORM ratio,
step-count distribution, entropy).

Output:
  • Console summary (per split)
  • JSON report → data/processed/rm_split_analysis.json
  • (Optional) Matplotlib plots → data/processed/rm_split_plots/
"""

import os, json, numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, UTC

# Paths ---------------------------------------------------------
DATA_DIR = Path("data/processed")
OUT_JSON = DATA_DIR / "rm_split_analysis.json"
OUT_PLOTS = DATA_DIR / "rm_split_plots"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": {
        "prm": DATA_DIR / "prm_train.jsonl",
        "orm": DATA_DIR / "orm_train.jsonl",
    },
    "val": {
        "prm": DATA_DIR / "prm_val.jsonl",
        "orm": DATA_DIR / "orm_val.jsonl",
    },
    "test": {
        "prm": DATA_DIR / "prm_test.jsonl",
        "orm": DATA_DIR / "orm_test.jsonl",
    },
}


# Helpers -------------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def summarize_prm(prm_data):
    """Compute summary stats for PRM split."""
    means, steps = [], []
    for r in prm_data:
        step_scores = r.get("step_targets", [])
        if step_scores:
            means.append(np.mean(step_scores))
            steps.append(len(step_scores))
    if not means:
        return {}

    hist, bins = np.histogram(means, bins=10, range=(0, 1))
    hist_pct = (hist / np.sum(hist)).round(3).tolist()
    return {
        "num_examples": len(prm_data),
        "mean": round(np.mean(means), 3),
        "stdev": round(np.std(means), 3),
        "entropy": round(entropy(hist + 1e-9, base=2), 3),
        "step_mean": int(np.mean(steps)),
        "step_median": int(np.median(steps)),
        "step_min": int(np.min(steps)),
        "step_max": int(np.max(steps)),
        "hist_bins": bins.tolist(),
        "hist_pct": hist_pct,
    }


def summarize_orm(orm_data):
    """Compute ORM label balance."""
    labels = [r.get("label", 0) for r in orm_data]
    return {
        "num_examples": len(labels),
        "label_1_ratio": round(np.mean(labels), 3) if labels else 0.0,
    }


# Plotting ------------------------------------------------------
def plot_histograms(split, prm_summary, orm_summary):
    """Save histograms for visual check."""
    bins, hist_pct = prm_summary["hist_bins"], prm_summary["hist_pct"]
    plt.figure(figsize=(6, 4))
    plt.bar(bins[:-1], hist_pct, width=0.08)
    plt.title(f"PRM Score Distribution ({split})")
    plt.xlabel("PRM Mean Score")
    plt.ylabel("Fraction")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"prm_hist_{split}.png")
    plt.close()

    plt.figure(figsize=(3, 3))
    plt.bar(["label 0", "label 1"],
            [1 - orm_summary["label_1_ratio"], orm_summary["label_1_ratio"]],
            color=["#888", "#3b82f6"])
    plt.title(f"ORM Label Ratio ({split})")
    plt.ylabel("Fraction")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"orm_ratio_{split}.png")
    plt.close()


# Main ----------------------------------------------------------
def main():
    report = {"timestamp": datetime.now(UTC).isoformat()}

    for split, paths in SPLITS.items():
        prm_data = load_jsonl(paths["prm"])
        orm_data = load_jsonl(paths["orm"])

        prm_summary = summarize_prm(prm_data)
        orm_summary = summarize_orm(orm_data)

        report[split] = {
            "PRM": prm_summary,
            "ORM": orm_summary,
        }

        # Console summary
        print(f"\n=== {split.upper()} SPLIT ===")
        print(f"PRM examples = {prm_summary['num_examples']}, ORM examples = {orm_summary['num_examples']}")
        print(f"PRM mean = {prm_summary['mean']} | stdev = {prm_summary['stdev']} | entropy = {prm_summary['entropy']}")
        print(f"ORM label 1 ratio = {orm_summary['label_1_ratio']}")
        print(f"Step count avg = {prm_summary['step_mean']} | median = {prm_summary['step_median']}")
        print("-" * 60)

        # Plots
        plot_histograms(split, prm_summary, orm_summary)

    # Save JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Inspection complete — report saved to {OUT_JSON}")
    print(f"📊 Histograms in {OUT_PLOTS}/")


if __name__ == "__main__":
    main()
