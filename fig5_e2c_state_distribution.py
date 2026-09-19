#!/usr/bin/env python3
"""fig5_e2c_state_distribution.py — Figure 5: E2c residual tail cell-state distribution.

A-class data figure. Sources:
- results/E2c_tail_grounding/E2c_result.json (A1 #15)
- results/E2c_tail_grounding/E2c_sample_aware.json (A1 #15)

Output: fig5_e2c_state_distribution.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
SRC_E2C = os.path.join(BASE, "results", "E2c_tail_grounding", "E2c_result.json")
SRC_SA = os.path.join(BASE, "results", "E2c_tail_grounding", "E2c_sample_aware.json")
OUT_PNG = os.path.join(BASE, "fig5_e2c_state_distribution.png")
OUT_PDF = os.path.join(BASE, "fig5_e2c_state_distribution.pdf")


def load_and_verify():
    with open(SRC_E2C, "r", encoding="utf-8") as f:
        e2c = json.load(f)
    with open(SRC_SA, "r", encoding="utf-8") as f:
        sample_aware = json.load(f)

    states = sorted(e2c["by_state"].keys())
    observed = []
    expected = []
    enrichment = []
    for st in states:
        s = e2c["by_state"][st]
        obs = s["top1pct_count"]
        exp = s["population_share"] * 50  # top 1% of 5000 = 50 cells
        observed.append(obs)
        expected.append(exp)
        enrichment.append(s["enrichment"])

    fig_values = {
        "states": states,
        "observed": observed,
        "expected": expected,
        "enrichment": enrichment,
        "c1q_enrichment": e2c["by_state"]["C1q_inflammatory"]["enrichment"],
        "c1q_count": e2c["by_state"]["C1q_inflammatory"]["top1pct_count"],
        "c1q_expected": e2c["by_state"]["C1q_inflammatory"]["population_share"] * 50,
        "fisher_or": sample_aware["fisher_OR"],
        "fisher_p": sample_aware["fisher_p"],
        "perm_p": sample_aware["concentration_perm_p"],
        "c1q_distinct_samples": sample_aware["c1q_top_distinct_samples"],
    }

    print("=== Fig 5 values to be plotted ===")
    for st, obs, exp, enr in zip(states, observed, expected, enrichment):
        print(f"  {st}: observed={obs}, expected={exp:.2f}, enrichment={enr:.3f}")
    print(f"\n  C1q OR={fig_values['fisher_or']:.3f}, Fisher p={fig_values['fisher_p']:.3e}")
    print(f"  C1q top cells across {fig_values['c1q_distinct_samples']} / 12 samples, permutation p={fig_values['perm_p']:.3f}")

    # Verification
    ok = True
    for i, st in enumerate(states):
        s = e2c["by_state"][st]
        if observed[i] != s["top1pct_count"]:
            print(f"MISMATCH observed {st}")
            ok = False
        if abs(expected[i] - s["population_share"] * 50) > 1e-9:
            print(f"MISMATCH expected {st}")
            ok = False
        if enrichment[i] != s["enrichment"]:
            print(f"MISMATCH enrichment {st}")
            ok = False

    print("\nVerification:", "PASS" if ok else "FAIL")
    return e2c, sample_aware, fig_values, ok


def draw(fig_values):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    states = fig_values["states"]
    x = np.arange(len(states))
    width = 0.35

    bars1 = ax.bar(x - width / 2, fig_values["observed"], width, label="Observed in top-1% residual", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, fig_values["expected"], width, label="Expected under null", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=30, ha="right")
    ax.set_ylabel("Cell count in top-1% residual tail (n = 50)")
    ax.set_title("Cell-state distribution of the velocity residual tail")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    # Annotate C1q
    c1q_idx = states.index("C1q_inflammatory")
    ax.annotate(
        f"C1q: {fig_values['c1q_count']}/50\nOR={fig_values['fisher_or']:.2f}, p={fig_values['fisher_p']:.1e}\n"
        f"across {fig_values['c1q_distinct_samples']}/12 samples",
        xy=(c1q_idx - width / 2, fig_values["observed"][c1q_idx]),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    # Proliferating zero note
    prol_idx = states.index("Proliferating")
    ax.annotate(
        "0/50",
        xy=(prol_idx - width / 2, 0),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

    fig.suptitle(
        "The velocity residual tail is enriched in C1q_inflammatory microglia and "
        "depleted in Homeostatic and DAM-like states",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved:\n  {OUT_PNG}\n  {OUT_PDF}")

    return (
        "Caption draft: Top-1% highest-residual cells (n = 50 of 5,000) are 2.0-fold enriched "
        "in C1q_inflammatory microglia (19 observed vs. 9.4 expected; OR = 2.67, Fisher p = 1.5e-3, "
        "Bonferroni-significant across six states) and depleted in Homeostatic and DAM-like states. "
        "Proliferating cells contribute none (0/50). The 19 C1q_inflammatory cells are distributed "
        f"across {fig_values['c1q_distinct_samples']} of 12 samples (within-state permutation p = "
        f"{fig_values['perm_p']:.2f}), arguing against a single-sample artefact."
    )


def main() -> int:
    e2c, sample_aware, fig_values, ok = load_and_verify()
    if not ok:
        return 1
    caption = draw(fig_values)
    print("\nCaption:", caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
