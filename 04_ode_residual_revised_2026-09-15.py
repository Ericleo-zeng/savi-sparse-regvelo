#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""04_ode_residual.py — V1 验证：production 输出的动力学方程残差。

原理（评价报告 Validation 1）：已产出的 s_end/u_end 是否真的满足动力学方程？
做法：对抽样细胞，用显著收紧的容差（tol=PICARD_TOL/1000, max_iter=50, K×2 分段）
从相同初值 (Ms, Mu) 重新传播，得到参考解 s_ref/u_ref，计算

    r_i = ||x_stored_i - x_ref_i|| / (||x_ref_i|| + eps)

这就是 production 容差（tol=1e-3, max_iter=10, K=10）下的实际误差预算。

用法：
    python 04_ode_residual.py                    # 默认抽 2000 cells，GPU 重传播
    python 04_ode_residual.py --n-cells 5000 --device cuda

2026-09-15 修订（V1 FAIL 归因对照需要）：
    新增 --checkpoint / --l3-dir / --ms / --mu / --out 五个覆盖参数，
    默认全部保持 production 行为（旧命令逐字节兼容）。
    抽样总体由硬编码 EXPECTED_N 改为 Ms.shape[0]，子集输入自动正确。
    结果 JSON 新增 source 溯源块（ms/mu/l3_dir/checkpoint/rate_max_*）。

输入：
    L2 checkpoint（参数，默认 production）+ L3 输出（stored 解，默认 production）
    + Ms/Mu（初值，默认 production 全量，按 RANDOM_SEED+1 固定抽样）
输出：
    {--out}/V1_result.json
    {--out}/residual_hist.png

验收线：median < 2e-3（2×PICARD_TOL）且 P99 < 1e-2 -> PASS
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import config as C

sys.path.insert(0, C.PROJECT_DIR)

THRESH_MEDIAN = 2e-3
THRESH_P99 = 1e-2


def main() -> int:
    ap = argparse.ArgumentParser(description="V1 ODE 传播残差验证")
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tight-tol", type=float, default=1e-6,
                    help="参考传播的 Picard 容差（production 为 1e-3）")
    ap.add_argument("--tight-max-iter", type=int, default=50)
    ap.add_argument("--k-mult", type=int, default=2,
                    help="参考传播的分段数倍数（K_eff = K × k_mult）")
    ap.add_argument("--ref-bs", type=int, default=1024,
                    help="参考/重算传播的 batch size（仅影响显存；逐细胞独立传播，"
                         "不影响数值。K×bs 过大 OOM 时调小，如 512/256）")
    # >>> 2026-09-15 新增：输入可覆盖（默认保持 production 行为，旧命令完全兼容）
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="L2 checkpoint 目录（默认 config.find_l2_checkpoint()）")
    ap.add_argument("--l3-dir", type=str, default=None,
                    help="stored 解所在 L3 输出目录（默认 config.find_l3_output()）")
    ap.add_argument("--ms", type=str, default=None,
                    help="Ms .npy 路径（默认 production 全量）")
    ap.add_argument("--mu", type=str, default=None,
                    help="Mu .npy 路径（默认 production 全量）")
    ap.add_argument("--out", type=str, default=None,
                    help="结果输出目录（默认 results/V1_ode_residual；"
                         "使用覆盖输入时务必另设，避免覆盖基线记录）")
    args = ap.parse_args()

    out_root = args.out or os.path.join(C.RESULTS_DIR, "V1_ode_residual")
    os.makedirs(out_root, exist_ok=True)

    # ---- 加载 L2 checkpoint 参数
    import torch
    from run_l3_only import load_l2_checkpoint
    from common import make_csr
    from closed_form_kernel import picard_propagate, segment_knots
    from sparse_regvelo_engine import PICARD_TOL

    ckpt = args.checkpoint or C.find_l2_checkpoint()
    print(f"[V1] checkpoint: {ckpt}")

    beta_c, gamma_c, t_c, gate_c, bias_c, alpha_c, W, meta = \
        load_l2_checkpoint(ckpt)
    device = args.device
    W_csr = make_csr(W, device=device)
    beta = torch.from_numpy(beta_c).to(device)
    gamma = torch.from_numpy(gamma_c).to(device)
    bias = torch.from_numpy(bias_c).to(device)
    t_t = torch.from_numpy(t_c).to(device)

    K_prod = int(meta.get("K", 10))
    K_ref = K_prod * args.k_mult
    t_max = float(t_c.max())
    if t_max <= 0:
        t_max = 1.0
    t_knots_ref = segment_knots(t_max, K_ref)
    print(f"[V1] production: K={K_prod}, tol={PICARD_TOL}, max_iter="
          f"{meta.get('picard_max_iter', 10)} | 参考: K={K_ref}, "
          f"tol={args.tight_tol}, max_iter={args.tight_max_iter}")

    # ---- 抽样：初值来自 Ms/Mu（默认 production 全量；覆盖时按输入实际规模抽样）
    ms_path = args.ms or C.MS_PATH
    mu_path = args.mu or C.MU_PATH
    rng = np.random.default_rng(C.RANDOM_SEED + 1)
    Ms = np.load(ms_path, mmap_mode="r")
    Mu = np.load(mu_path, mmap_mode="r")
    N_total = int(Ms.shape[0])
    n_cells = min(args.n_cells, N_total)
    idx = np.sort(rng.choice(N_total, size=n_cells, replace=False))
    s0 = np.asarray(Ms[idx], dtype=np.float32)
    u0 = np.asarray(Mu[idx], dtype=np.float32)

    # ---- 尝试取 stored 解；缺失则用 checkpoint 参数现场重算
    #      （00_inventory 发现部分环境下 s_end.npy/v.npy 缺失，此时本脚本退化为
    #        "production 容差 vs 收紧容差" 的自足对比，不依赖 stored 全量输出）
    s_stored = u_stored = None
    stored_mode = "stored"
    l3_dir = args.l3_dir or C.find_l3_output()
    try:
        s_path = os.path.join(l3_dir, "s_end.npy")
        u_path = os.path.join(l3_dir, "u_end.npy")
        if os.path.isfile(s_path) and os.path.isfile(u_path):
            s_stored = np.asarray(np.load(s_path, mmap_mode="r")[idx])
            u_stored = np.asarray(np.load(u_path, mmap_mode="r")[idx])
            print(f"[V1] 使用 stored 解: {l3_dir}")
        else:
            raise FileNotFoundError(f"{l3_dir} 中 s_end.npy/u_end.npy 不全")
    except (FileNotFoundError, Exception) as e:  # noqa: BLE001
        stored_mode = "recomputed"
        print(f"[V1] stored 解不可用（{e}）")
        print(f"[V1] 退化模式：现场以 checkpoint 参数 (K={K_prod}, tol={PICARD_TOL}, "
              f"max_iter={meta.get('picard_max_iter', 10)}) 重算对照解")
        t_knots_prod = segment_knots(t_max, K_prod)
        s_stored = np.empty_like(s0)
        u_stored = np.empty_like(u0)
        for st in range(0, len(idx), args.ref_bs):
            en = min(st + args.ref_bs, len(idx))
            ub = torch.from_numpy(u0[st:en]).to(device)
            sb = torch.from_numpy(s0[st:en]).to(device)
            u_b, s_b, res, conv = picard_propagate(
                ub, sb, W_csr, bias, beta, gamma, t_knots_prod,
                max_iter=int(meta.get("picard_max_iter", 10)), tol=PICARD_TOL)
            u_stored[st:en] = u_b.cpu().numpy()
            s_stored[st:en] = s_b.cpu().numpy()
            del ub, sb, u_b, s_b
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- 收紧容差重传播（分 batch 防显存峰值）
    n = len(idx)
    bs = args.ref_bs
    s_ref = np.empty_like(s0)
    u_ref = np.empty_like(u0)
    n_unconverged = 0
    for st in range(0, n, bs):
        en = min(st + bs, n)
        ub = torch.from_numpy(u0[st:en]).to(device)
        sb = torch.from_numpy(s0[st:en]).to(device)
        u_b, s_b, res, conv = picard_propagate(
            ub, sb, W_csr, bias, beta, gamma, t_knots_ref,
            max_iter=args.tight_max_iter, tol=args.tight_tol)
        if not bool(conv):
            n_unconverged += 1
            print(f"[V1] 警告：参考传播在 batch {st}-{en} 未收敛 (res={float(res):.2e})")
        u_ref[st:en] = u_b.cpu().numpy()
        s_ref[st:en] = s_b.cpu().numpy()
        del ub, sb, u_b, s_b
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"[V1] 参考传播完成，未收敛 batch 数: {n_unconverged}")

    # ---- 残差统计
    res_s = C.rel_l2_per_cell(s_stored, s_ref)
    res_u = C.rel_l2_per_cell(u_stored, u_ref)
    v_stored = beta_c[None, :] * u_stored - gamma_c[None, :] * s_stored
    v_ref = beta_c[None, :] * u_ref - gamma_c[None, :] * s_ref
    res_v = C.rel_l2_per_cell(v_stored, v_ref)

    result = {
        "n_cells": int(n),
        "checkpoint": ckpt,
        "compare_mode": stored_mode,
        # >>> 2026-09-15 新增：溯源块（供 07 登记与交叉对照分辨输入来源）
        "source": {
            "ms": os.path.abspath(ms_path),
            "mu": os.path.abspath(mu_path),
            "l3_dir": os.path.abspath(l3_dir),
            "population": N_total,
            "sampled": int(n),
            "rate_max_mode": meta.get("rate_max_mode"),
            "rate_max_actual": meta.get("rate_max_actual"),
        },
        "production_config": {"K": K_prod, "picard_tol": PICARD_TOL,
                              "picard_max_iter": meta.get("picard_max_iter", 10)},
        "reference_config": {"K": K_ref, "tol": args.tight_tol,
                             "max_iter": args.tight_max_iter,
                             "n_unconverged_batches": n_unconverged},
        "residual_s_end": C.summarize(res_s),
        "residual_u_end": C.summarize(res_u),
        "residual_v": C.summarize(res_v),
    }
    for k, r in (("s_end", res_s), ("u_end", res_u), ("v", res_v)):
        print(f"[V1] residual {k}: median={result[f'residual_' + ('s_end' if k=='s_end' else ('u_end' if k=='u_end' else 'v'))]['median']:.2e} "
              f"p99={np.percentile(r, 99):.2e} max={r.max():.2e}")

    ok = (result["residual_s_end"]["median"] < THRESH_MEDIAN
          and result["residual_s_end"]["p99"] < THRESH_P99
          and result["residual_v"]["median"] < THRESH_MEDIAN
          and result["residual_v"]["p99"] < THRESH_P99)
    result["verdict"] = "PASS" if ok else "FAIL"

    # ---- stored 侧自身日志信息一并纳入（若存在）
    try:
        l3_meta_p = os.path.join(l3_dir, "l3_metadata.json")
        if os.path.isfile(l3_meta_p):
            with open(l3_meta_p, "r", encoding="utf-8") as f:
                result["production_l3_metadata"] = json.load(f)
    except FileNotFoundError:
        pass

    # ---- 残差分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arr, lab in ((res_s, "s_end"), (res_u, "u_end"), (res_v, "v")):
            ax.hist(np.log10(arr + 1e-16), bins=60, alpha=0.55, label=lab)
        ax.axvline(np.log10(THRESH_MEDIAN), color="k", ls="--",
                   label=f"median thresh {THRESH_MEDIAN:g}")
        ax.set_xlabel("log10(relative residual)")
        ax.set_ylabel("cells")
        ax.legend()
        ax.set_title("V1 ODE propagation residual (production vs tightened reference)")
        fig.tight_layout()
        png = os.path.join(out_root, "residual_hist.png")
        fig.savefig(png, dpi=150)
        print(f"[V1] 残差分布图: {png}")
    except Exception as e:
        print(f"[V1] 绘图跳过: {e}")

    out_json = os.path.join(out_root, "V1_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[V1] 判决: {result['verdict']}  ->  {out_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
