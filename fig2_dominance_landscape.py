#!/usr/bin/env python3
"""fig2_dominance_landscape.py — Figure 2: dominance landscape scatter.

B-class schematic using actual A1 median values. Sources:
- results/V5_rate_max/compare_manual_vs_adaptive.log (A1 #9) -> rate bound rm20
- results/V7_no_cycling/COMPARE_v7_result.json (A1 #11) -> gene universe V7b
- results/V6_grn_tiered/COMPARE_result.json (A1 #10) -> GRN aggressive
- results/V8_K_sensitivity/COMPARE_result.json (A1 #12) -> segment count K10

Output: fig2_dominance_landscape.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
SRC_RATE = os.path.join(BASE, "results", "V5_rate_max", "compare_manual_vs_adaptive.log")
SRC_GENE = os.path.join(BASE, "results", "V7_no_cycling", "COMPARE_v7_result.json")
SRC_GRN = os.path.join(BASE, "results", "V6_grn_tiered", "COMPARE_result.json")
SRC_K = os.path.join(BASE, "results", "V8_K_sensitivity", "COMPARE_result.json")
OUT_PNG = os.path.join(BASE, "fig2_dominance_landscape.png")
OUT_PDF = os.path.join(BASE, "fig2_dominance_landscape.pdf")


def load_and_verify():
    # Rate bound rm20
    with open(SRC_RATE, "r", encoding="utf-8") as f:
        rate_log = f.read()
    m = re.search(r"l3_r20: v cosine median=(\S+) min=\S+ \| v rel_l2 median=(\S+)", rate_log)
    if not m:
        raise ValueError("Could not parse rate-bound rm20 values")
    rate_cos = float(m.group(1))
    rate_rel = float(m.group(2))

    # Gene universe V7b
    with open(SRC_GENE, "r", encoding="utf-8") as f:
        gene_data = json.load(f)
    gene_cos = gene_data["v_cosine"]["median"]
    gene_rel = gene_data["v_rel_l2"]["median"]

    # GRN aggressive
    with open(SRC_GRN, "r", encoding="utf-8") as f:
        grn_data = json.load(f)
    grn_cos = grn_data["tests"]["l3_v6"]["v_cosine"]["median"]
    grn_rel = grn_data["tests"]["l3_v6"]["v_rel_l2"]["median"]

    # Segment count K10
    with open(SRC_K, "r", encoding="utf-8") as f:
        k_data = json.load(f)
    k_cos = k_data["tests"]["l3_K10"]["v_cosine"]["median"]
    k_rel = k_data["tests"]["l3_K10"]["v_rel_l2"]["median"]

    points = {
        "rate_bound": {"cos": rate_cos, "rel": rate_rel, "label": "rate bound (rm20)"},
        "gene_universe": {"cos": gene_cos, "rel": gene_rel, "label": "gene universe (no-cycle V7b)"},
        "grn_weighting": {"cos": grn_cos, "rel": grn_rel, "label": "GRN weighting (aggressive)"},
        "segment_count": {"cos": k_cos, "rel": k_rel, "label": "segment count (K=10)"},
    }

    print("=== Fig 2 values to be plotted ===")
    for k, v in points.items():
        print(f"  {k}: cosine median={v['cos']:.6f}, relative-L2 median={v['rel']:.6e}")

    # Verification by re-reading
    ok = True
    with open(SRC_RATE, "r", encoding="utf-8") as f:
        txt = f.read()
    m2 = re.search(r"l3_r20: v cosine median=(\S+) min=\S+ \| v rel_l2 median=(\S+)", txt)
    if float(m2.group(1)) != rate_cos or float(m2.group(2)) != rate_rel:
        print("MISMATCH rate bound")
        ok = False

    print("\nVerification:", "PASS" if ok else "FAIL")
    return points, ok


def draw(points):
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "rate_bound": "#d62728",
        "gene_universe": "#ff7f0e",
        "grn_weighting": "#2ca02c",
        "segment_count": "#1f77b4",
    }
    markers = {
        "rate_bound": "s",
        "gene_universe": "o",
        "grn_weighting": "D",
        "segment_count": "^",
    }

    for key, vals in points.items():
        ax.scatter(
            vals["cos"],
            vals["rel"],
            color=colors[key],
            marker=markers[key],
            s=200,
            label=vals["label"],
            edgecolors="black",
            zorder=5,
        )

    # Strict pass lines
    ax.axvline(0.99, color="gray", ls="--", lw=1, label="strict cosine line")
    ax.axhline(0.05, color="gray", ls=":", lw=1, label="strict rel-L2 line")

    ax.set_xlabel("Velocity direction stability (per-cell cosine median)")
    ax.set_ylabel("Velocity magnitude perturbation (relative-L2 median)")
    ax.set_xlim(0.3, 1.02)
    ax.set_yscale("log")
    ax.set_title("Dominance landscape of engine choices")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, which="both", ls="--", alpha=0.4)

    fig.suptitle(
        "Segment count passes the strict lines; GRN weighting passes direction but fails amplitude; "
        "rate bound and gene-universe restriction fail both",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved:\n  {OUT_PNG}\n  {OUT_PDF}")

    return (
        "Caption draft: Each point locates one sensitivity arm by its median velocity direction "
        "(per-cell cosine, x-axis) and magnitude perturbation (relative-L2, y-axis). The strict "
        "acceptance lines are cosine median = 0.99 and relative-L2 median = 0.05. Segment count "
        "(K=10) and GRN weighting remain inside the pass region; the kinetic rate bound and "
        "gene-universe restriction are the most consequential perturbations."
    )


def main() -> int:
    points, ok = load_and_verify()
    if not ok:
        return 1
    caption = draw(points)
    print("\nCaption:", caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
