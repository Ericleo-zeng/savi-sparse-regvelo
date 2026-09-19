#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10_v7a_no_renorm_input.py — E3：V7a 对照臂输入（列子集，无重归一化）。

V7b（已跑）= 剔除 57 周期基因 + 重算 moments（含重归一化）。
V7a（本脚本）= 生产 Ms/Mu 直接按保留基因列子集（归一化保持生产口径），
              隔离"基因剔除"与"重归一化"两个效应。
W 复用 V7 的 W.npz（同一保留宇宙、同一基因序）；X_scVI 为 per-cell 量，
与 V7 完全相同。
输出到 results/_subset/ 下，命名 *_v7a.npy。
"""
import json
import numpy as np

VAL = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
PROD = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input"
GENES = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/gene_universe_used.txt"
V7 = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input_v7_nocycling"

genes_all = [x.strip() for x in open(GENES, encoding="utf-8") if x.strip()]
removed = set(json.load(open(f"{V7}/export_manifest_v7.json", encoding="utf-8"))
              ["cycling_hit_in_universe"])
keep_idx = np.array([i for i, g in enumerate(genes_all) if g not in removed])
print(f"保留基因: {len(keep_idx)} / {len(genes_all)} "
      f"(剔除 {len(genes_all) - len(keep_idx)})")

# 校验：V7 的 W 维度与保留数一致
from scipy.sparse import load_npz
assert load_npz(f"{V7}/W.npz").shape[0] == len(keep_idx)

idx = np.load(f"{VAL}/results/_subset/cell_index_n5000.npy")
for name in ("Ms", "Mu"):
    a = np.load(f"{PROD}/{name}.npy", mmap_mode="r")     # (74984, 17714)
    sub = np.asarray(a[idx][:, keep_idx])                # 先切行(同细胞)再切列
    np.save(f"{VAL}/results/_subset/{name}_sub_n5000_v7a.npy",
            sub.astype(np.float32))
    print(f"{name}_v7a {sub.shape}")

# X_scVI per-cell，与 V7b 完全相同 → 直接复制
import shutil
shutil.copy(f"{VAL}/results/_subset/X_sub_n5000_v7.npy",
            f"{VAL}/results/_subset/X_sub_n5000_v7a.npy")
print("X_scVI_v7a = 复用 X_sub_n5000_v7.npy（per-cell 量，与基因子集无关）")
print("done: Ms/Mu/X_sub_n5000_v7a.npy 就绪；L2 时 --w 用 V7 的 W.npz")
