#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""08_v7_compare.py — V7 专用对比：不同基因宇宙（G 不同）的 L3 输出按基因名交集对齐。

背景：06_compare_conditions 要求 v shape 一致，V7 剔除增殖基因后 G=17657
≠ 参考的 17714，无法使用。本脚本按基因名交集（此处 = V7 保留的 17657 基因）
对齐两个速度场后逐细胞对比。细胞必须同一批（V7 子集用同一 cell_index 切出）。

判据（方向性检查，较 06 宽松——V7 是变量重要性问题而非敏感性 Pass/Fail）：
    common-gene v cosine median > 0.95 -> 增殖基因非速度场主驱动
    0.90–0.95 -> 部分驱动；< 0.90 -> 强驱动

用法：
    python 08_v7_compare.py \
        --ref results/V5_rate_max/l3_adaptive \
        --test results/V7_no_cycling/l7_v7 \
        --gene-list /mnt/t9/.../gene_universe_used.txt \
        --manifest /mnt/t9/.../sparse_engine_input_v7_nocycling/export_manifest_v7.json \
        --tag V7_no_cycling

输出：results/{tag}/COMPARE_v7_result.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

import config as C

COS_DIRECTIONAL = 0.95   # 方向性检查阈值（宽松于 06 的 0.99）


def _ensure_complete_l3(d: str) -> None:
    """完整性守卫（同 06 v2）：l3_metadata.json 缺失 = 半成品，拒绝。"""
    if not os.path.exists(os.path.join(d, "l3_metadata.json")):
        raise SystemExit(f"[FATAL] {d} 缺少 l3_metadata.json：疑似半成品，拒绝对比。")


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 不同基因宇宙对比（交集基因对齐）")
    ap.add_argument("--ref", required=True, help="参考 L3 输出目录（全宇宙）")
    ap.add_argument("--test", required=True, help="待对比 L3 输出目录（剔除后宇宙）")
    ap.add_argument("--gene-list", required=True, help="原始 gene_universe_used.txt")
    ap.add_argument("--manifest", required=True,
                    help="V7 export manifest（提供实际剔除基因清单）")
    ap.add_argument("--tag", default=None, help="输出目录名（默认 test 目录名）")
    args = ap.parse_args()

    _ensure_complete_l3(args.ref)
    _ensure_complete_l3(args.test)

    # ---- 基因映射：ref=全宇宙；test=剔除后（export 保留原顺序）
    genes_all = [x.strip() for x in open(args.gene_list, encoding="utf-8")
                 if x.strip()]
    removed = set(json.load(open(args.manifest, encoding="utf-8"))
                  ["cycling_hit_in_universe"])
    idx = {g: i for i, g in enumerate(genes_all)}
    test_genes = [g for g in genes_all if g not in removed]
    ref_idx = np.array([idx[g] for g in test_genes])   # test 基因在 ref 中的列号

    # ---- 加载交集基因上的速度场
    v_ref = np.asarray(np.load(os.path.join(args.ref, "v.npy"), mmap_mode="r")
                       )[:, ref_idx]
    v_tst = np.asarray(np.load(os.path.join(args.test, "v.npy"), mmap_mode="r"))
    if v_ref.shape != v_tst.shape:
        raise SystemExit(f"[FATAL] 对齐后 shape 仍不一致: ref {v_ref.shape} vs "
                         f"test {v_tst.shape}（细胞子集或基因列表有误）")
    N, G_common = v_ref.shape

    cos = C.cosine_per_cell(v_tst, v_ref)
    rel = C.rel_l2_per_cell(v_tst, v_ref)
    result = {
        "ref": args.ref, "test": args.test,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        "design": ("V7 方向性对比：不同基因宇宙，按基因名交集对齐后逐细胞对比；"
                   "各 arm 均用各自 adaptive rate_max（生产口径）；"
                   "A 方案（操作化）：test arm 含重归一化效应"),
        "N_cells": int(N),
        "G_common": int(G_common),
        "G_ref": int(len(genes_all)),
        "G_test": int(len(test_genes)),
        "genes_removed": sorted(removed),
        "v_cosine": C.summarize(cos),
        "v_cosine_min": float(cos.min()),
        "v_rel_l2": C.summarize(rel),
        "threshold": {"directional_cosine_median": COS_DIRECTIONAL},
    }
    med = result["v_cosine"]["median"]
    if med > COS_DIRECTIONAL:
        verdict = "PASS（增殖基因非主驱动）"
    elif med > 0.90:
        verdict = "PARTIAL（部分驱动）"
    else:
        verdict = "DRIVEN（强驱动：结论依赖周期基因）"
    result["verdict"] = verdict

    print(f"[V7CMP] common genes: {G_common} (ref {len(genes_all)}, "
          f"test {len(test_genes)}, removed {len(removed)})")
    print(f"[V7CMP] v cosine median={med:.5f} min={result['v_cosine_min']:.5f} | "
          f"v rel_l2 median={result['v_rel_l2']['median']:.3e}")
    print(f"[V7CMP] 判决: {verdict}")

    tag = args.tag or os.path.basename(args.test.rstrip("/"))
    out_dir = os.path.join(C.RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "COMPARE_v7_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[V7CMP] -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
