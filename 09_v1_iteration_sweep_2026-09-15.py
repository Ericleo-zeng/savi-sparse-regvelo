#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09_v1_iteration_sweep.py — E2：V1 尾部的迭代预算归因（投稿前关闭实验）。

问题：V1 中 v 的 P99 残差擦线 FAIL（1.04–1.17e-2 vs 1e-2）。是动力学刚性
内禀，还是 max_iter=10 的预算限制？本探针在 faithful checkpoint 上抽样，
用 max_iter ∈ {10,20,30,50}（tol=1e-3, K=10）传播，以收紧参考
（K=20, tol=1e-6, max_iter=50）为基准，给出 v 残差 P99 随预算的曲线。

注意：本实验只证机制，不改已存生产输出。
用法：
    python 09_v1_iteration_sweep.py \
        --checkpoint results/V5_rate_max/ckpt_adaptive \
        --ms results/_subset/Ms_sub_n5000.npy \
        --mu results/_subset/Mu_sub_n5000.npy \
        --n-cells 500 --out results/V1_iteration_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import config as C

sys.path.insert(0, C.PROJECT_DIR)


def main() -> int:
    ap = argparse.ArgumentParser(description="V1 tail: Picard iteration sweep")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--mu", required=True)
    ap.add_argument("--n-cells", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[10, 20, 30, 50])
    ap.add_argument("--bs", type=int, default=256,
                    help="探针批大小（内存 ∝ K×bs×G，仅影响显存不影响数值）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from run_l3_only import load_l2_checkpoint
    from common import make_csr
    from closed_form_kernel import picard_propagate, segment_knots
    from sparse_regvelo_engine import PICARD_TOL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_c, gamma_c, t_c, gate_c, bias_c, alpha_c, W, meta = \
        load_l2_checkpoint(args.checkpoint)
    W_csr = make_csr(W, device=device)
    beta = torch.from_numpy(beta_c).to(device)
    gamma = torch.from_numpy(gamma_c).to(device)
    bias = torch.from_numpy(bias_c).to(device)
    t_max = float(t_c.max()) or 1.0

    rng = np.random.default_rng(args.seed)
    Ms = np.load(args.ms, mmap_mode="r")
    Mu = np.load(args.mu, mmap_mode="r")
    N = Ms.shape[0]
    n = min(args.n_cells, N)
    idx = np.sort(rng.choice(N, size=n, replace=False))
    s0 = np.asarray(Ms[idx], dtype=np.float32)
    u0 = np.asarray(Mu[idx], dtype=np.float32)

    def propagate(K: int, tol: float, max_iter: int):
        knots = segment_knots(t_max, K)
        u = np.empty_like(u0)
        s = np.empty_like(s0)
        n_unconv = 0
        for st in range(0, n, args.bs):
            en = min(st + args.bs, n)
            ub = torch.from_numpy(u0[st:en]).to(device)
            sb = torch.from_numpy(s0[st:en]).to(device)
            u_b, s_b, res, conv = picard_propagate(
                ub, sb, W_csr, bias, beta, gamma, knots,
                max_iter=max_iter, tol=tol)
            if not bool(conv):
                n_unconv += 1
            u[st:en] = u_b.cpu().numpy()
            s[st:en] = s_b.cpu().numpy()
            del ub, sb, u_b, s_b
            if device == "cuda":
                torch.cuda.empty_cache()
        v = beta_c[None, :] * u - gamma_c[None, :] * s
        return v, n_unconv

    print(f"[E2] reference: K=20, tol=1e-6, max_iter=50 ...")
    v_ref, unconv_ref = propagate(20, 1e-6, 50)
    print(f"[E2] reference done (unconverged batches: {unconv_ref})")

    result = {"checkpoint": args.checkpoint,
              "n_cells": int(n), "seed": args.seed,
              "reference": {"K": 20, "tol": 1e-6, "max_iter": 50,
                            "n_unconverged": unconv_ref},
              "budgets": {}}
    print(f"{'budget':>7s} | {'v rel median':>13s} | {'v rel P99':>12s} | "
          f"{'v rel max':>12s} | {'cosine med':>11s} | unconv")
    for m in args.budgets:
        v_m, unconv = propagate(10, PICARD_TOL, m)
        rel = C.rel_l2_per_cell(v_m, v_ref)
        cos = C.cosine_per_cell(v_m, v_ref)
        entry = {"v_rel": C.summarize(rel),
                 "v_cosine_median": float(np.median(cos)),
                 "v_cosine_min": float(cos.min()),
                 "v_cosine_p99": float(np.percentile(cos, 99)),
                 "n_unconverged": unconv}
        result["budgets"][str(m)] = entry
        print(f"{m:>7d} | {entry['v_rel']['median']:>13.3e} | "
              f"{entry['v_rel']['p99']:>12.3e} | {entry['v_rel']['max']:>12.3e} | "
              f"{entry['v_cosine_median']:>11.6f} | {unconv}")

    # >>> E2b：K × tol 口径差分解（max_iter=50；回答"尾部来自 K 还是 tol"）
    print()
    print("[E2b] K × tol decomposition (max_iter=50), vs reference (K=20, tol=1e-6):")
    print(f"{'arm':>16s} | {'v rel median':>13s} | {'v rel P99':>12s} | "
          f"{'v rel max':>12s} | {'cosine med':>11s}")
    result["decomposition"] = {}
    for K_arm, tol_arm in ((10, 1e-3), (10, 1e-6), (20, 1e-3)):
        v_a, _ = propagate(K_arm, tol_arm, 50)
        rel_a = C.rel_l2_per_cell(v_a, v_ref)
        cos_a = C.cosine_per_cell(v_a, v_ref)
        s_a = C.summarize(rel_a)
        result["decomposition"][f"K{K_arm}_tol{tol_arm:g}"] = {
            "v_rel": s_a, "v_cosine_median": float(np.median(cos_a)),
            "v_cosine_min": float(cos_a.min())}
        print(f"K{K_arm:<3d} tol={tol_arm:<6g} | {s_a['median']:>13.3e} | "
              f"{s_a['p99']:>12.3e} | {s_a['max']:>12.3e} | "
              f"{np.median(cos_a):>11.6f} (min {cos_a.min():.4f})")

    out_json = os.path.join(args.out, "V1_iter_sweep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[E2] -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
