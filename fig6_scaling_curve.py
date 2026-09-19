#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig6_scaling_curve.py — 绘制 B2 规模扩展曲线。

读取 results/B2_scaling/B2_result.json，生成 log-log 双轴曲线图：
    - 左 y 轴：runtime_sec
    - 右 y 轴：peak_gpu_gb
区分 SAVI 成功 / OOM 或失败 / scVelo 竞品 / A1#13 参考点。

输出：
    validation_Sparse_regvelo/fig6_scaling_curve.png (300 dpi)
    validation_Sparse_regvelo/fig6_scaling_curve.pdf

用法：
    cd /home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo
    python fig6_scaling_curve.py
"""
from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import config as C

RESULT_JSON = os.path.join(C.RESULTS_DIR, "B2_scaling", "B2_result.json")
OUT_PNG = os.path.join(C.VALIDATION_DIR, "fig6_scaling_curve.png")
OUT_PDF = os.path.join(C.VALIDATION_DIR, "fig6_scaling_curve.pdf")


def load_result() -> dict[str, Any]:
    if not os.path.isfile(RESULT_JSON):
        raise FileNotFoundError(f"未找到 {RESULT_JSON}，请先运行 B2_scaling_collect.py")
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def split_points(runs: list[dict[str, Any]], x_key: str) -> dict[str, list[dict[str, Any]]]:
    """将 runs 按状态分组：success / oom_failure / competitor / reference。"""
    groups: dict[str, list[dict[str, Any]]] = {
        "success": [],
        "oom_failure": [],
        "competitor": [],
        "reference": [],
    }
    for r in runs:
        status = (r.get("status") or "").lower()
        if r.get("type") == "scvelo":
            groups["competitor"].append(r)
        elif r.get("source") == "A1#13":
            # 仅保留成功的 A1#13 参考点；OOM 信息已移至 figure_caption_note
            if status != "oom" and status != "failure":
                ref = dict(r)
                if x_key == "n_cells":
                    ref["n_cells"] = r.get("n_cells")
                elif x_key == "n_genes":
                    ref["n_genes"] = r.get("G")
                groups["reference"].append(ref)
        elif status == "success":
            groups["success"].append(r)
        else:
            groups["oom_failure"].append(r)
    return groups


def extract_xy(runs: list[dict[str, Any]], x_key: str, y_key: str):
    """返回可用于 matplotlib 的 x, y 列表，跳过 None。"""
    xs, ys = [], []
    for r in runs:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def extract_x(runs: list[dict[str, Any]], x_key: str) -> np.ndarray:
    """仅提取横坐标，用于 OOM 等无 y 值的标记。"""
    xs = []
    for r in runs:
        x = r.get(x_key)
        if x is not None:
            xs.append(float(x))
    return np.asarray(xs)


def plot_panel(
    ax_runtime,
    ax_mem,
    groups: dict[str, list[dict[str, Any]]],
    x_key: str,
    title: str,
    xlabel: str,
) -> None:
    """绘制单个子图：左轴 runtime，右轴 memory。"""
    ax_runtime.set_xscale("log")
    ax_runtime.set_yscale("log")
    ax_mem.set_yscale("log")
    ax_runtime.set_title(title, fontsize=11)
    ax_runtime.set_xlabel(xlabel, fontsize=10)
    ax_runtime.set_ylabel("Runtime (s)", color="tab:blue", fontsize=10)
    ax_mem.set_ylabel("Peak GPU memory (GB)", color="tab:red", fontsize=10)

    # SAVI success
    x, y = extract_xy(groups["success"], x_key, "runtime_sec")
    if len(x):
        ax_runtime.plot(x, y, "o-", color="tab:blue", label="SAVI success (time)")
    x, y = extract_xy(groups["success"], x_key, "peak_gpu_gb")
    if len(x):
        ax_mem.plot(x, y, "s--", color="tab:red", label="SAVI success (mem)")

    # OOM / failure：放在左轴底部，用向下三角
    xf, _ = extract_xy(groups["oom_failure"], x_key, "runtime_sec")
    if len(xf):
        y_floor = ax_runtime.get_ylim()[0]
        ax_runtime.plot(xf, [y_floor] * len(xf), "v", color="tab:gray",
                        markersize=8, clip_on=False, label="OOM / failure")

    # scVelo competitor
    x, y = extract_xy(groups["competitor"], x_key, "runtime_sec")
    if len(x):
        ax_runtime.plot(x, y, "^--", color="tab:green", label="scVelo (time)")

    # A1#13 reference memory star on right axis
    x, y = extract_xy(groups["reference"], x_key, "peak_gpu_gb")
    if len(x):
        ax_mem.plot(x, y, "*", color="tab:orange", markersize=12,
                    label="A1#13 reference (mem)")

    # 右轴刻度：普通小数，保留 1 位，不使用科学计数法 / ×10⁰
    ax_mem.set_yscale("log")
    ax_mem.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:.1f}"))
    ax_mem.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 10])
    ax_mem.set_yticklabels(["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "10.0"])

    # 美化
    ax_runtime.tick_params(axis="y", labelcolor="tab:blue")
    ax_mem.tick_params(axis="y", labelcolor="tab:red")
    ax_runtime.grid(True, which="both", ls="--", alpha=0.4)


def main() -> int:
    result = load_result()
    scaling = result.get("scaling_runs", [])
    competitor = result.get("competitor_runs", [])
    references = result.get("a1_references", [])

    # 按子图分组
    cell_groups = split_points([r for r in scaling if r.get("type") == "cell"] + references, "n_cells")
    gene_groups = split_points([r for r in scaling if r.get("type") == "gene"] + references, "n_genes")
    # 竞品按 n_cells 分
    cell_groups["competitor"] = [r for r in competitor]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1_mem = ax1.twinx()
    ax2_mem = ax2.twinx()

    plot_panel(ax1, ax1_mem, cell_groups, "n_cells",
               "SAVI L3 cell scaling (G=17,714, K=10)",
               "Number of cells")
    plot_panel(ax2, ax2_mem, gene_groups, "n_genes",
               "SAVI L3 gene scaling (N=5,000, K=10)",
               "Number of genes")

    # 合并图例放在下方
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1m, labels1m = ax1_mem.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2m, labels2m = ax2_mem.get_legend_handles_labels()
    all_lines = lines1 + lines1m + lines2 + lines2m
    all_labels = labels1 + labels1m + labels2 + labels2m
    # 去重，保持顺序
    seen = set()
    uniq = []
    for ln, lb in zip(all_lines, all_labels):
        if lb not in seen:
            seen.add(lb)
            uniq.append((ln, lb))
    fig.legend(*zip(*uniq), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    # 图注/说明文字
    note = (
        "scVelo: dynamical-modeling representative, no GRN constraint, same hardware"
    )
    fig.text(0.5, -0.10, note, ha="center", fontsize=9, style="italic")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] saved {OUT_PNG} and {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
