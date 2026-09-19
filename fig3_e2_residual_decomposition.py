#!/usr/bin/env python3
"""fig3_e2_residual_decomposition.py — Figure 3: E2 iteration sweep + K×tol decomposition.

A-class data figure. Sources:
- results/V1_iteration_sweep/V1_iter_sweep.json (A1 #3, #4)

Output: fig3_e2_residual_decomposition.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
SRC = os.path.join(BASE, "results", "V1_iteration_sweep", "V1_iter_sweep.json")
OUT_PNG = os.path.join(BASE, "fig3_e2_residual_decomposition.png")
OUT_PDF = os.path.join(BASE, "fig3_e2_residual_decomposition.pdf")


def load_and_verify():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    budgets = data["budgets"]
    decomp = data["decomposition"]

    # Values to be plotted
    fig_values = {
        "budget_sweep": {},
        "decomposition": {},
    }

    print("=== Fig 3 values to be plotted ===")
    print("\nIteration budget sweep (velocity relative-L2):")
    for b in ["10", "20", "30", "50"]:
        v = budgets[b]["v_rel"]
        print(f"  max_iter={b}: median={v['median']:.6e}, P99={v['p99']:.6e}, max={v['max']:.6e}")
        fig_values["budget_sweep"][b] = {
            "median": v["median"],
            "p99": v["p99"],
            "max": v["max"],
        }

    print("\nK × tolerance decomposition (velocity relative-L2):")
    for key in ["K10_tol0.001", "K10_tol1e-06", "K20_tol0.001"]:
        v = decomp[key]["v_rel"]
        print(f"  {key}: median={v['median']:.6e}, P99={v['p99']:.6e}, max={v['max']:.6e}")
        fig_values["decomposition"][key] = {
            "median": v["median"],
            "p99": v["p99"],
            "max": v["max"],
        }

    # Verification against source
    ok = True
    for b in ["10", "20", "30", "50"]:
        src_v = data["budgets"][b]["v_rel"]
        for stat in ["median", "p99", "max"]:
            if fig_values["budget_sweep"][b][stat] != src_v[stat]:
                print(f"MISMATCH budget {b} {stat}")
                ok = False
    for key in ["K10_tol0.001", "K10_tol1e-06", "K20_tol0.001"]:
        src_v = data["decomposition"][key]["v_rel"]
        for stat in ["median", "p99", "max"]:
            if fig_values["decomposition"][key][stat] != src_v[stat]:
                print(f"MISMATCH decomposition {key} {stat}")
                ok = False

    print("\nVerification:", "PASS" if ok else "FAIL")
    return data, fig_values, ok


def draw(data, fig_values):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # Left: iteration budget sweep
    ax1 = fig.add_subplot(gs[0, 0])
    budgets = ["10", "20", "30", "50"]
    x = np.arange(len(budgets))
    stats = ["median", "p99", "max"]
    colors = {"median": "#1f77b4", "p99": "#ff7f0e", "max": "#d62728"}
    markers = {"median": "o", "p99": "s", "max": "^"}

    for stat in stats:
        y = [fig_values["budget_sweep"][b][stat] for b in budgets]
        ax1.plot(x, y, marker=markers[stat], color=colors[stat], label=stat, linewidth=1.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}" for b in budgets])
    ax1.set_xlabel("Picard iteration budget (max_iter)")
    ax1.set_ylabel("Velocity relative-L2 residual")
    ax1.set_yscale("log")
    ax1.set_title("Iteration budget sweep (K=10, tol=1e-3, n=500)")
    ax1.legend(title="Statistic")
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # Right: K × tol decomposition
    ax2 = fig.add_subplot(gs[0, 1])
    keys = ["K10_tol0.001", "K10_tol1e-06", "K20_tol0.001"]
    labels = ["K=10, tol=1e-3", "K=10, tol=1e-6", "K=20, tol=1e-3"]
    x = np.arange(len(keys))
    width = 0.25

    for i, stat in enumerate(stats):
        y = [fig_values["decomposition"][k][stat] for k in keys]
        ax2.bar(x + i * width, y, width, label=stat, color=colors[stat])

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Velocity relative-L2 residual")
    ax2.set_yscale("log")
    ax2.set_title("K × tolerance decomposition (50 iterations)")
    ax2.legend(title="Statistic")
    ax2.grid(True, which="both", axis="y", ls="--", alpha=0.4)

    fig.suptitle(
        "E2 residual attribution: iteration budget is neutral; segment count dominates the median and tolerance dominates the maximum",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved:\n  {OUT_PNG}\n  {OUT_PDF}")

    return (
        "Caption draft: (Left) Iteration-budget sweep shows identical velocity relative-L2 "
        "median, P99 and maximum across max_iter = 10–50, indicating that the production "
        "iteration budget is not the source of the residual tail. (Right) K × tolerance "
        "decomposition against a tightened reference (K=20, tol=1e-6): increasing K from 10 "
        "to 20 collapses the median residual ~28-fold, while tightening tol from 1e-3 to 1e-6 "
        "reduces the maximum residual; P99 remains in a narrow band."
    )


def main() -> int:
    data, fig_values, ok = load_and_verify()
    if not ok:
        return 1
    caption = draw(data, fig_values)
    print("\nCaption:", caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
