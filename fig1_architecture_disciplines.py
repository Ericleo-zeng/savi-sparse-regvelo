#!/usr/bin/env python3
"""fig1_architecture_disciplines.py — Figure 1: engine architecture + three validation disciplines.

B-class schematic draft. No numerical data. Layout-only placeholder.

Output: fig1_architecture_disciplines.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
OUT_PNG = os.path.join(BASE, "fig1_architecture_disciplines.png")
OUT_PDF = os.path.join(BASE, "fig1_architecture_disciplines.pdf")


def draw():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", lw=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Title (no "Figure 1." prefix)
    ax.text(7, 6.7, "SAVI engine architecture and validation disciplines",
            ha="center", va="center", fontsize=14, weight="bold")

    # Engine pipeline (left side)
    ax.text(3.5, 6.2, "Engine pipeline", ha="center", fontsize=12, weight="bold")

    box(1.5, 4.8, 4.0, 0.9, "L1: steady-state init\n+ per-gene gate", "#e6f3ff")
    arrow(3.5, 4.8, 3.5, 4.2)
    box(1.5, 3.2, 4.0, 0.9, "L2: decoupled per-gene\nkinetic fit (β, γ)", "#e6f3ff")
    arrow(3.5, 3.2, 3.5, 2.6)
    box(1.5, 1.6, 4.0, 0.9, "L3: K-segment closed-form\nPicard propagation", "#e6f3ff")
    arrow(3.5, 1.6, 3.5, 1.0)
    box(1.5, 0.3, 4.0, 0.6, "outputs: v = β⊙u − γ⊙s\nanalytic Jacobian", "#d1e7dd")

    # Fallback annotation
    ax.text(6.0, 2.0, "Krylov / full-dynamics\nfallback routing", ha="left", fontsize=8,
            style="italic", color="#555555")

    # Disciplines (right side)
    ax.text(10.5, 6.2, "Three validation disciplines", ha="center", fontsize=12, weight="bold")

    box(8.0, 4.8, 5.0, 0.9,
        "Reference discipline\nuse only checkpoints refitted\non identical data",
        "#fff3cd")
    box(8.0, 3.2, 5.0, 0.9,
        "Provenance discipline\nrecord requested vs actual rate bounds\n+ SHA-256 fingerprints",
        "#fff3cd")
    box(8.0, 1.6, 5.0, 0.9,
        "Completeness discipline\nrequire L3 metadata completion marker;\nreject half-written outputs",
        "#fff3cd")

    # Single clean connection annotation, vertically centered between columns
    ax.annotate(
        "validated by the three disciplines →",
        xy=(8.0, 3.5),
        xytext=(6.0, 3.5),
        fontsize=10,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")

    return (
        "Caption draft: The SAVI engine comprises three levels (L1–L3). L1 computes steady-state "
        "initial values and a per-gene gate; L2 fits per-gene kinetics; L3 propagates cells over K "
        "segments with Picard iteration and fallback routing, producing velocity and an analytic Jacobian. "
        "Three validation disciplines (Reference, Provenance, Completeness) audit the pipeline and are "
        "described in §2.3."
    )


def main() -> int:
    caption = draw()
    print("\nCaption:", caption)
    print("\nNote: B-class schematic draft; no numerical values; layout and labels are structural placeholders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
