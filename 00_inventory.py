#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""00_inventory.py — 验证第 0 步：路径盘点与健康检查（只读，不改任何数据）。

用法：
    cd /home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo
    python 00_inventory.py

输出：
    results/00_inventory.json   — 全部盘点结果
    终端打印人类可读汇总
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import config as C


def check_file(path: str, expect_shape=None) -> dict:
    info = {"path": path, "exists": os.path.isfile(path)}
    if not info["exists"]:
        return info
    info["size_mb"] = round(os.path.getsize(path) / 1024**2, 2)
    if path.endswith(".npy"):
        try:
            arr = np.load(path, mmap_mode="r")
            info["shape"] = list(arr.shape)
            info["dtype"] = str(arr.dtype)
            if expect_shape is not None:
                info["shape_ok"] = list(arr.shape) == list(expect_shape)
        except Exception as e:
            info["load_error"] = str(e)
    elif path.endswith(".npz") and "W" in os.path.basename(path):
        try:
            from scipy.sparse import load_npz
            W = load_npz(path)
            info["shape"] = list(W.shape)
            info["nnz"] = int(W.nnz)
            info["edges_ok"] = (W.nnz == C.EXPECTED_EDGES)
        except Exception as e:
            info["load_error"] = str(e)
    return info


def main() -> int:
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    rep = {"validation_dir": C.VALIDATION_DIR, "sections": {}}

    print("=" * 72)
    print("[00] 路径盘点与健康检查")
    print("=" * 72)

    # ---- 1. 引擎输入（production）
    print("\n[1] 引擎输入数据")
    sec = {}
    for name, p, shp in [
        ("Ms", C.MS_PATH, (C.EXPECTED_N, C.EXPECTED_G)),
        ("Mu", C.MU_PATH, (C.EXPECTED_N, C.EXPECTED_G)),
        ("X_scVI", C.X_SCVI_PATH, (C.EXPECTED_N, 10)),
        ("W", C.W_NPZ_PATH, None),
    ]:
        sec[name] = check_file(p, shp)
        print(f"    {name:8s}: {sec[name]}")
    rep["sections"]["engine_inputs"] = sec

    # ---- 2. L2 checkpoint 甄别
    print("\n[2] L2 checkpoint 候选甄别")
    cands = []
    for d in C.L2_CHECKPOINT_CANDIDATES:
        meta_p = os.path.join(d, "metadata.json")
        entry = {"dir": d, "has_metadata": os.path.isfile(meta_p)}
        if entry["has_metadata"]:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            entry["meta"] = meta
            entry["files"] = sorted(os.listdir(d))
            # P0 检查：rate_max provenance 是否完整
            entry["provenance_ok"] = all(
                k in meta for k in ("rate_max_actual", "rate_max_mode",
                                    "beta_p995", "gamma_p995"))
            entry["rate_max_recorded"] = meta.get("rate_max")
        print(f"    {entry}")
        cands.append(entry)
    rep["sections"]["l2_checkpoints"] = cands
    try:
        ckpt = C.find_l2_checkpoint()
        print(f"    -> 选定 production checkpoint: {ckpt}")
        rep["production_checkpoint"] = ckpt
    except FileNotFoundError as e:
        print(f"    !! {e}")
        rep["production_checkpoint"] = None

    # ---- 3. production L3 输出定位
    print("\n[3] production L3 输出定位")
    l3_meta = None
    try:
        l3_dir = C.find_l3_output()
        print(f"    -> L3 输出目录: {l3_dir}")
        entry = {"dir": l3_dir, "files": sorted(os.listdir(l3_dir))}
        for name, shp in [("u_end.npy", (C.EXPECTED_N, C.EXPECTED_G)),
                          ("s_end.npy", (C.EXPECTED_N, C.EXPECTED_G)),
                          ("v.npy", (C.EXPECTED_N, C.EXPECTED_G))]:
            entry[name] = check_file(os.path.join(l3_dir, name), shp)
        meta_p = os.path.join(l3_dir, "l3_metadata.json")
        if os.path.isfile(meta_p):
            with open(meta_p, "r", encoding="utf-8") as f:
                l3_meta = json.load(f)
            entry["l3_metadata"] = l3_meta
            print(f"    engine_route = {l3_meta.get('engine_route')}")
            print(f"    route_counts = {l3_meta.get('route_counts')}")
            print(f"    picard_residual_max = {l3_meta.get('picard_residual_max')}")
            print(f"    n_unconverged_batches = {l3_meta.get('n_unconverged_batches')}")
        rep["sections"]["l3_output"] = entry
    except FileNotFoundError as e:
        print(f"    !! {e}")
        rep["sections"]["l3_output"] = None

    # ---- 4. 引擎代码与依赖检查
    print("\n[4] 引擎关键文件存在性")
    code = {}
    for name in ["common.py", "closed_form_kernel.py", "decoupled_velocity_engine.py",
                 "sparse_regvelo_engine.py", "run_l3_only.py"]:
        p = os.path.join(C.PROJECT_DIR, name)
        code[name] = os.path.isfile(p)
        print(f"    {name:35s}: {'OK' if code[name] else 'MISSING'}")
    rep["sections"]["engine_code"] = code

    # ---- 5. 历史日志中的 route / 残差信息（快速扫一遍 JSON 日志）
    print("\n[5] 历史运行日志摘要（logs/*.json）")
    logs = sorted(glob.glob(os.path.join(C.PROJECT_DIR, "logs",
                                         "sparse_regvelo_engine_*.json")))
    log_summary = []
    for lp in logs[-10:]:  # 最近 10 次
        try:
            with open(lp, "r", encoding="utf-8") as f:
                j = json.load(f)
            recs = j.get("records", j)
            log_summary.append({
                "file": os.path.basename(lp),
                "engine_route": recs.get("engine_route"),
                "route_counts": recs.get("route_counts"),
                "picard_residual_max": recs.get("picard_residual_max"),
                "rate_max": recs.get("rate_max"),
                "rate_max_mode": recs.get("rate_max_mode"),
                "n_gated": recs.get("n_gated"),
                "peak_memory_gb": recs.get("peak_memory_gb"),
                "wall_time_sec": recs.get("wall_time_sec"),
            })
        except Exception as e:
            log_summary.append({"file": os.path.basename(lp), "error": str(e)})
    for s in log_summary:
        print(f"    {s}")
    rep["sections"]["recent_logs"] = log_summary

    out = os.path.join(C.RESULTS_DIR, "00_inventory.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[00] 盘点结果已写入: {out}")
    print("下一步：阅读 01_P0_修复说明.md，完成引擎代码修复后运行 02_batch_invariance.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
