#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_cpu_gpu_consistency.py — V3 验证：CPU 与 GPU 实现的数值一致性。

原理：同一算法（Picard/FOH + 路由）在 CPU FP32 与 GPU FP32 下应只相差
浮点累加顺序带来的噪声（~1e-6 量级）。显著偏差说明 GPU 路径有实现错误。

用法：
    python 03_cpu_gpu_consistency.py                 # 默认 n=500（CPU 较慢）
    python 03_cpu_gpu_consistency.py --n-cells 2000

输出：
    results/V3_cpu_gpu/cpu/    — CPU 版 L3 输出
    results/V3_cpu_gpu/gpu/    — GPU 版 L3 输出
    results/V3_cpu_gpu/V3_result.json

验收线（评价报告，FP32）：
    1e-5 ~ 1e-4  -> PASS（正常浮点噪声）
    1e-4 ~ 1e-2  -> WARN（需查累加顺序/共振支路）
    >= 1e-2      -> FAIL（必须调查）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import config as C

sys.path.insert(0, C.PROJECT_DIR)

THRESH_PASS = 1e-4
THRESH_FAIL = 1e-2


def main() -> int:
    ap = argparse.ArgumentParser(description="V3 CPU/GPU 一致性验证")
    ap.add_argument("--n-cells", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="两侧用相同 batch_size，隔离设备因素")
    # >>> 2026-09-15 新增（E1：faithful 复测，关闭 09-14 WARN）：
    # 覆盖输入，默认行为与原版逐字节一致（仍走 ensure_subset*）
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="L2 checkpoint 目录（默认 ensure_subset_checkpoint，"
                         "即 T0.5 拼接物；faithful 复测传 ckpt_adaptive）")
    ap.add_argument("--ms", type=str, default=None, help="Ms 路径（默认子集）")
    ap.add_argument("--mu", type=str, default=None, help="Mu 路径（默认子集）")
    ap.add_argument("--out", type=str, default=None,
                    help="输出根目录（默认 results/V3_cpu_gpu）")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("[V3] 无 CUDA 设备，无法做 CPU/GPU 对比。请在 RTX 机器上运行。")
        return 2

    out_root = args.out or os.path.join(C.RESULTS_DIR, "V3_cpu_gpu")
    os.makedirs(out_root, exist_ok=True)
    if args.ms:
        ms_path, mu_path = args.ms, args.mu
    else:
        subset = C.ensure_subset(args.n_cells)
        ms_path, mu_path = subset["ms"], subset["mu"]
    ckpt = args.checkpoint or C.ensure_subset_checkpoint(args.n_cells)
    print(f"[V3] checkpoint: {ckpt}")

    from run_l3_only import run_l3_streaming
    dirs = {}
    for dev in ("cpu", "cuda"):
        out_dir = os.path.join(out_root, dev)
        if os.path.isfile(os.path.join(out_dir, "l3_metadata.json")):
            print(f"[V3] {dev} 已存在，跳过: {out_dir}")
        else:
            print(f"[V3] 运行 device={dev} (n={args.n_cells}) ...")
            run_l3_streaming(
                ms_path=ms_path, mu_path=mu_path,
                checkpoint_dir=ckpt, out_dir=out_dir,
                batch_size=args.batch_size, device=dev,
            )
        dirs[dev] = out_dir

    # ---- 对比（以 CPU 为参考）
    result = {"n_cells": args.n_cells, "batch_size": args.batch_size,
              "checkpoint": ckpt, "ms": os.path.abspath(ms_path),
              "mu": os.path.abspath(mu_path),
              "scope": ("faithful-refit" if args.checkpoint else
                        "frozen-param(T0.5 composite)"),
              "arrays": {}, "route_check": {}}
    worst = 0.0
    for name in ("u_end", "s_end", "v"):
        ref = np.asarray(np.load(os.path.join(dirs["cpu"], f"{name}.npy"),
                                 mmap_mode="r"))
        tst = np.asarray(np.load(os.path.join(dirs["cuda"], f"{name}.npy"),
                                 mmap_mode="r"))
        rel = C.rel_l2_per_cell(tst, ref)
        stats = C.summarize(rel)
        stats["max_abs_diff"] = float(np.abs(tst - ref).max())
        result["arrays"][name] = stats
        worst = max(worst, stats["max"])
        print(f"[V3] {name}: rel_l2 median={stats['median']:.2e} "
              f"p99={stats['p99']:.2e} max={stats['max']:.2e}")
    cos = C.cosine_per_cell(
        np.asarray(np.load(os.path.join(dirs["cuda"], "v.npy"), mmap_mode="r")),
        np.asarray(np.load(os.path.join(dirs["cpu"], "v.npy"), mmap_mode="r")))
    result["v_cosine_min"] = float(cos.min())
    print(f"[V3] v cosine min={cos.min():.8f}")

    # ---- 路由一致性
    metas = {}
    for dev in ("cpu", "cuda"):
        with open(os.path.join(dirs[dev], "l3_metadata.json"),
                  "r", encoding="utf-8") as f:
            metas[dev] = json.load(f)
    same_route = metas["cpu"].get("route_counts") == metas["cuda"].get("route_counts")
    result["route_check"] = {"cpu": metas["cpu"].get("route_counts"),
                             "cuda": metas["cuda"].get("route_counts"),
                             "identical": bool(same_route)}
    if not same_route:
        print("[V3] 警告：CPU/GPU 路由分布不一致（Krylov 探针是设备相关的）")

    # ---- 判决
    if worst < THRESH_PASS:
        verdict = "PASS"
    elif worst < THRESH_FAIL:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    result["verdict"] = verdict
    result["worst_max_rel"] = worst

    out_json = os.path.join(out_root, "V3_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[V3] 判决: {verdict} (worst max rel = {worst:.2e})  ->  {out_json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
