#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""config.py — validation_Sparse_regvelo 集中路径配置与共享工具。

所有验证脚本从这里读取路径；如需改路径，只改本文件。
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np

# ================================================================ 路径配置
# 引擎仓库（代码 + 既有产出）
PROJECT_DIR = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/sparse_regvelo"

# 本验证工程
VALIDATION_DIR = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")

# production 输入数据（由 export_for_sparse_engine.py 生成）
ENGINE_INPUT_DIR = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input"
MS_PATH = os.path.join(ENGINE_INPUT_DIR, "Ms.npy")
MU_PATH = os.path.join(ENGINE_INPUT_DIR, "Mu.npy")
X_SCVI_PATH = os.path.join(ENGINE_INPUT_DIR, "X_scVI.npy")
W_NPZ_PATH = os.path.join(ENGINE_INPUT_DIR, "W.npz")

# L2 checkpoint 候选。2026-09-14 盘点确认存在两条 production 链路：
#   链路A（旧，rate_max 固定）：{PROJECT_DIR}/l2_checkpoint (副本)/ → l3_output/（不完整）
#   链路B（新，adaptive rate_max，完整）：{ENGINE_OUTPUT_DIR}/l2_checkpoint_adaptive/
#        → {ENGINE_OUTPUT_DIR}/l3_output_adaptive/（含 s_end.npy + l3_metadata.json）
# 验证以链路B为准；链路A仅用于 V5 的 rate_max=20 对照（若参数确实一致）。
ENGINE_OUTPUT_DIR = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_output"
L2_CHECKPOINT_CANDIDATES = [
    os.path.join(ENGINE_OUTPUT_DIR, "l2_checkpoint_adaptive"),
    os.path.join(PROJECT_DIR, "l2_checkpoint (副本)"),
    os.path.join(PROJECT_DIR, "l2_checkpoint"),
]

# production L3 输出目录（u_end.npy/s_end.npy/v.npy/l3_metadata.json 所在目录）。
# 盘点已确认完整输出为 l3_output_adaptive；l3_output/ 与 l3_output (副本)/
# 均不完整（缺 s_end/v/l3_metadata），是旧固定 rate_max 链路的残留。
L3_OUTPUT_DIR = os.path.join(ENGINE_OUTPUT_DIR, "l3_output_adaptive")

# 原始数据（溯源用，验证脚本不直接读）
ADATA_PATH = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad"
NETWORK_PATH = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/network.tsv"
GENE_LIST_PATH = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/gene_universe_used.txt"

# 期望规模（inventory 核对用）
EXPECTED_N = 74984
EXPECTED_G = 17714
EXPECTED_EDGES = 194843

RANDOM_SEED = 20260914

# 子抽样输入/检查点存放处（02/03/05 共用，创建一次后复用）
SUBSET_DIR = os.path.join(RESULTS_DIR, "_subset")


# ================================================================ 共享工具
def find_l2_checkpoint() -> str:
    """在候选目录中找到真正的 production checkpoint（含 metadata.json，优先 N=EXPECTED_N）。"""
    valid = []
    for d in L2_CHECKPOINT_CANDIDATES:
        meta_p = os.path.join(d, "metadata.json")
        if os.path.isfile(meta_p):
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            valid.append((d, meta))
    if not valid:
        raise FileNotFoundError(
            f"未找到有效 L2 checkpoint，候选: {L2_CHECKPOINT_CANDIDATES}")
    for d, meta in valid:
        if meta.get("N") == EXPECTED_N and meta.get("G") == EXPECTED_G:
            return d
    return valid[0][0]


def find_l3_output() -> str:
    """定位 production L3 输出目录（含 u_end.npy 且形状为 (N,G)）。"""
    if L3_OUTPUT_DIR and os.path.isfile(os.path.join(L3_OUTPUT_DIR, "u_end.npy")):
        return L3_OUTPUT_DIR
    candidates = [os.path.join(PROJECT_DIR, "l3_output")]
    candidates += glob.glob(os.path.join(PROJECT_DIR, "*", ""))
    candidates += glob.glob(os.path.join(
        os.path.dirname(ENGINE_INPUT_DIR), "**", "u_end.npy"), recursive=True)
    for c in candidates:
        u_path = c if c.endswith("u_end.npy") else os.path.join(c, "u_end.npy")
        if os.path.isfile(u_path):
            try:
                arr = np.load(u_path, mmap_mode="r")
                if arr.shape == (EXPECTED_N, EXPECTED_G):
                    return os.path.dirname(u_path)
            except Exception:
                continue
    raise FileNotFoundError(
        "未找到 production L3 输出（u_end.npy 形状需为 "
        f"({EXPECTED_N},{EXPECTED_G})）。请在 config.py 中手动设置 L3_OUTPUT_DIR。")


def ensure_subset(n_cells: int, seed: int = RANDOM_SEED) -> dict:
    """创建（或复用）固定随机抽样的 cells 子集输入。

    输出（SUBSET_DIR 下）：
        cell_index_n{n}.npy  — 抽样细胞在 production 数据中的行号 (n,)
        Ms_sub_n{n}.npy      — (n, G) FP32
        Mu_sub_n{n}.npy      — (n, G) FP32
        X_sub_n{n}.npy       — (n, 10) FP32
    """
    os.makedirs(SUBSET_DIR, exist_ok=True)
    idx_path = os.path.join(SUBSET_DIR, f"cell_index_n{n_cells}.npy")
    ms_sub = os.path.join(SUBSET_DIR, f"Ms_sub_n{n_cells}.npy")
    mu_sub = os.path.join(SUBSET_DIR, f"Mu_sub_n{n_cells}.npy")
    x_sub = os.path.join(SUBSET_DIR, f"X_sub_n{n_cells}.npy")
    if all(os.path.isfile(p) for p in (idx_path, ms_sub, mu_sub, x_sub)):
        return {"idx": idx_path, "ms": ms_sub, "mu": mu_sub, "x": x_sub}

    Ms = np.load(MS_PATH, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(Ms.shape[0], size=n_cells, replace=False)).astype(np.int64)
    np.save(idx_path, idx)
    np.save(ms_sub, np.asarray(Ms[idx], dtype=np.float32))
    Mu = np.load(MU_PATH, mmap_mode="r")
    np.save(mu_sub, np.asarray(Mu[idx], dtype=np.float32))
    X = np.load(X_SCVI_PATH, mmap_mode="r")
    np.save(x_sub, np.asarray(X[idx], dtype=np.float32))
    print(f"[subset] n={n_cells} 子集已创建: {SUBSET_DIR}")
    return {"idx": idx_path, "ms": ms_sub, "mu": mu_sub, "x": x_sub}


def ensure_subset_checkpoint(n_cells: int) -> str:
    """基于 production checkpoint 创建子集版 checkpoint（仅改 N / t / alpha_fit 行）。

    run_l3_only.run_l3_streaming 会断言 Ms 形状 == meta(N,G)，
    因此子集验证必须配套子集 checkpoint。输出：
        {SUBSET_DIR}/l2_checkpoint_sub_n{n}/  （beta/gamma/gate/bias/W.npz 直接复制）
    """
    src = find_l2_checkpoint()
    dst = os.path.join(SUBSET_DIR, f"l2_checkpoint_sub_n{n_cells}")
    meta_dst = os.path.join(dst, "metadata.json")
    if os.path.isfile(meta_dst):
        return dst
    os.makedirs(dst, exist_ok=True)
    idx = np.load(ensure_subset(n_cells)["idx"])

    for name in ("beta.npy", "gamma.npy", "gate.npy", "bias.npy", "W.npz"):
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    t = np.load(os.path.join(src, "t.npy"), mmap_mode="r")
    np.save(os.path.join(dst, "t.npy"), np.asarray(t[idx], dtype=np.float32))

    alpha_src = os.path.join(src, "alpha_fit.npy")
    if os.path.isfile(alpha_src):
        a = np.load(alpha_src, mmap_mode="r")
        if a.ndim == 2:
            np.save(os.path.join(dst, "alpha_fit.npy"),
                    np.asarray(a[idx], dtype=np.float32))
        else:
            shutil.copy2(alpha_src, os.path.join(dst, "alpha_fit.npy"))

    with open(os.path.join(src, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["N"] = int(n_cells)
    meta["subset_of_N"] = meta.get("N") if meta.get("N") != n_cells else EXPECTED_N
    meta["subset_cell_index"] = os.path.join(SUBSET_DIR, f"cell_index_n{n_cells}.npy")
    with open(meta_dst, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[subset] 子集 checkpoint 已创建: {dst}")
    return dst


def rel_l2_per_cell(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """逐细胞 ||a-b||/max(||b||,eps)，a/b: (n, G)。返回 (n,) float64。"""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm(a64 - b64, axis=1)
    den = np.maximum(np.linalg.norm(b64, axis=1), eps)
    return num / den


def cosine_per_cell(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """逐细胞 cosine(a, b)，返回 (n,) float64。"""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    na = np.maximum(np.linalg.norm(a64, axis=1), eps)
    nb = np.maximum(np.linalg.norm(b64, axis=1), eps)
    return (a64 * b64).sum(axis=1) / (na * nb)


def summarize(arr: np.ndarray) -> dict:
    """median/P95/P99/max 汇总。"""
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
