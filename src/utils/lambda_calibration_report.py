"""
Lambda Calibration Report Generator
===================================

Generates a publication-grade PNG report comparing λ weights
(before vs after calibration) and gradient magnitude statistics.

Inputs:
    - pre_calibration (dict): original λs and/or gradient stats
    - post_calibration (dict): tuned λs and/or gradient stats
    - output_path (str): where to save the generated PNG

Optional:
    - include_grad_stats: include per-loss gradient magnitude bars
    - meta_info: dict with run metadata (date, config checksum, dataset, etc.)

Output:
    -> reports/calibration_report.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def generate_lambda_calibration_report(
    pre_lambdas: dict,
    post_lambdas: dict,
    pre_grad_stats: dict = None,
    post_grad_stats: dict = None,
    output_path: str = "reports/calibration_report.png",
    meta_info: dict = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract and align lambda keys
    labels = list(pre_lambdas.keys())
    pre_vals = [pre_lambdas.get(k, 0.0) for k in labels]
    post_vals = [post_lambdas.get(k, 0.0) for k in labels]

    # Setup figure
    fig, axes = plt.subplots(
        2 if pre_grad_stats else 1,
        1,
        figsize=(9, 6 if pre_grad_stats else 4),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # ------------------------------------------------------------------
    # 1️⃣ Lambda comparison (Before vs After)
    # ------------------------------------------------------------------
    ax = axes[0]
    x = np.arange(len(labels))
    bar_width = 0.35

    ax.bar(x - bar_width / 2, pre_vals, width=bar_width, label="Before", alpha=0.7)
    ax.bar(x + bar_width / 2, post_vals, width=bar_width, label="After", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("λ Value")
    ax.set_title("λ Calibration — Before vs After")
    ax.legend()
    ax.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # 2️⃣ Gradient magnitude section (optional)
    # ------------------------------------------------------------------
    if pre_grad_stats:
        ax2 = axes[1]
        grad_labels = list(pre_grad_stats.keys())
        pre_grad = [pre_grad_stats[k]["mean"] for k in grad_labels]
        post_grad = [post_grad_stats[k]["mean"] for k in grad_labels] if post_grad_stats else pre_grad

        ax2.bar(grad_labels, pre_grad, alpha=0.6, label="Before")
        if post_grad_stats:
            ax2.bar(grad_labels, post_grad, alpha=0.6, label="After")
        ax2.set_ylabel("Mean |grad|")
        ax2.set_title("Gradient Magnitude per Loss Component")
        ax2.legend()
        ax2.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # 🧾 Metadata footer
    # ------------------------------------------------------------------
    if meta_info:
        meta_text = " | ".join([f"{k}: {v}" for k, v in meta_info.items()])
        fig.text(0.01, 0.01, meta_text, fontsize=8, ha="left", va="bottom", alpha=0.7)
    else:
        fig.text(
            0.01,
            0.01,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=8,
            ha="left",
            va="bottom",
            alpha=0.7,
        )

    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✅ Calibration report saved → {output_path}")
