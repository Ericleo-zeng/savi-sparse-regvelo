#!/usr/bin/env python3
"""run_l3_only.py — 独立 L3 流式传播入口（解耦 L2 检查点）。

从 L2 检查点加载参数，流式读取 Ms/Mu（memmap），逐 batch 在 GPU 上执行
Picard 传播与降级路由，结果流式写入 .npy memmap。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import get_device, make_csr, peak_memory_gb, reset_peak_memory

# 依赖（与主脚本一致）
try:
    from closed_form_kernel import (
        arnoldi_expmv,
        coupled_backup_engine,
        picard_propagate,
        segment_knots,
    )
    from sparse_regvelo_engine import (
        ALPHA_CLAMP_MAX,
        KRYLOV_M,
        KRYLOV_TOL,
        PICARD_TOL,
        analytic_jacobian,
        select_engine,
        _krylov_residual_probe,
    )
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERR = str(_e)


# ================================================================ 检查点加载
def load_l2_checkpoint(checkpoint_dir: str):
    """加载 L2 检查点，全部保持 CPU numpy。"""
    beta = np.load(os.path.join(checkpoint_dir, "beta.npy"))
    gamma = np.load(os.path.join(checkpoint_dir, "gamma.npy"))
    t = np.load(os.path.join(checkpoint_dir, "t.npy"))
    gate = np.load(os.path.join(checkpoint_dir, "gate.npy"))
    bias = np.load(os.path.join(checkpoint_dir, "bias.npy"))

    alpha_fit = None
    alpha_path = os.path.join(checkpoint_dir, "alpha_fit.npy")
    if os.path.exists(alpha_path):
        alpha_fit = np.load(alpha_path)

    with open(os.path.join(checkpoint_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    w_path = os.path.join(checkpoint_dir, "W.npz")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"W.npz not found in checkpoint: {checkpoint_dir}")
    from scipy.sparse import load_npz
    W = load_npz(w_path)

    return beta, gamma, t, gate, bias, alpha_fit, W, meta


# ================================================================ 核心：流式 L3
def run_l3_streaming(ms_path: str,
                     mu_path: str,
                     checkpoint_dir: str,
                     out_dir: str,
                     batch_size: int = 2048,
                     device: str | None = None,
                     K: int | None = None) -> dict:
    """流式 L3：Ms/Mu 不整体进内存，逐 batch 读、算、写。"""
    if not _DEPS_OK:
        raise RuntimeError(f"依赖缺失: {_DEPS_ERR}")

    if device is None:
        device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 加载 L2 参数（CPU）
    beta_cpu, gamma_cpu, t_cpu, gate_cpu, bias_cpu, alpha_fit_cpu, W, meta = \
        load_l2_checkpoint(checkpoint_dir)


    rate_max = meta.get("rate_max", 20.0)
    rate_max_mode = meta.get("rate_max_mode", "fixed")
    beta_p995 = meta.get("beta_p995_probe", None)
    gamma_p995 = meta.get("gamma_p995_probe", None)
    print(f"[L3] L2 边界模式: {rate_max_mode}")
    if beta_p995 is not None:
        print(f"[L3] 探测轮 beta_p99.5={beta_p995:.2f}, gamma_p99.5={gamma_p995:.2f}")
    print(f"[L3] L2 配置: rate_max={rate_max}, N={meta['N']}, G={meta['G']}")
    N, G = meta["N"], meta["G"]
    K_src = "CLI 覆盖" if K is not None else "checkpoint meta"
    K = K if K is not None else meta.get("K", 10)
    print(f"[L3] 分段数 K={K} ({K_src})")
    picard_max_iter = meta.get("picard_max_iter", 10)
    hard = meta.get("hard", True)

    # ---------------- W 与标量参数上 GPU
    W_csr = make_csr(W, device=device)
    beta = torch.from_numpy(beta_cpu).to(device=device, dtype=torch.float32)
    gamma = torch.from_numpy(gamma_cpu).to(device=device, dtype=torch.float32)
    t = torch.from_numpy(t_cpu).to(device=device, dtype=torch.float32)
    gate = torch.from_numpy(gate_cpu).to(device=device).bool()
    bias = torch.from_numpy(bias_cpu).to(device=device, dtype=torch.float32)

    t_max = float(t.max().item())
    if t_max <= 0.0:
        t_max = 1.0
    t_knots = segment_knots(t_max, K)

    # ---------------- 流式输入（memmap，只读）
    Ms_mmap = np.load(ms_path, mmap_mode="r")
    Mu_mmap = np.load(mu_path, mmap_mode="r")
    assert Ms_mmap.shape == (N, G), f"Ms shape mismatch: {Ms_mmap.shape} vs ({N},{G})"
    assert Mu_mmap.shape == (N, G), f"Mu shape mismatch: {Mu_mmap.shape} vs ({N},{G})"

    # ---------------- 预创建输出 memmap（不占用内存）
    u_end_mmap = np.lib.format.open_memmap(
        os.path.join(out_dir, "u_end.npy"), mode="w+", dtype=np.float32, shape=(N, G))
    s_end_mmap = np.lib.format.open_memmap(
        os.path.join(out_dir, "s_end.npy"), mode="w+", dtype=np.float32, shape=(N, G))
    v_mmap = np.lib.format.open_memmap(
        os.path.join(out_dir, "v.npy"), mode="w+", dtype=np.float32, shape=(N, G))

    # alpha 维度判断
    alpha_is_per_cell = (alpha_fit_cpu is not None and alpha_fit_cpu.ndim == 2)
    alpha_mmap = None
    if alpha_is_per_cell:
        alpha_mmap = np.lib.format.open_memmap(
            os.path.join(out_dir, "alpha.npy"), mode="w+", dtype=np.float32, shape=(N, G))

    # ---------------- 统计与日志
    picard_residuals = []
    picard_residuals_eff = []
    route_counts = {
        "closed_form_foh": 0,
        "krylov_expmv": 0,
        "full_sparse_expmv": 0,
        "full_dynamics_path": 0,
    }
    krylov_residual_global = 0.0
    n_unconverged = 0

    reset_peak_memory()

    # ---------------- 逐 batch 流式处理
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        actual_bs = end - start

        # 从 memmap 读取当前 batch（显式复制，解除 mmap 引用）
        sb_cpu = np.array(Ms_mmap[start:end], dtype=np.float32)
        ub_cpu = np.array(Mu_mmap[start:end], dtype=np.float32)

        # 上 GPU（仅当前 batch）
        sb = torch.from_numpy(sb_cpu).to(device=device)
        ub = torch.from_numpy(ub_cpu).to(device=device)

        # ---- L3 传播
        u_b, s_b, res, converged = picard_propagate(
            ub, sb, W_csr, bias, beta, gamma, t_knots,
            max_iter=picard_max_iter, tol=PICARD_TOL,
        )
        res = float(res)
        converged = bool(converged)
        picard_residuals.append(res)
        if not converged:
            n_unconverged += 1
        res_eff = res if converged else max(res, 2.0 * PICARD_TOL)
        picard_residuals_eff.append(res_eff)

        # ---- 降级探测
        kry_b = krylov_residual_global
        if res_eff > PICARD_TOL:
            kry_b = _krylov_residual_probe(W_csr, sb, gamma, bias, t_max)
            krylov_residual_global = max(krylov_residual_global, kry_b)

        route = select_engine(False, res_eff, kry_b)
        route_counts[route] += 1

        if route in ("krylov_expmv", "full_sparse_expmv"):
            m = KRYLOV_M if route == "krylov_expmv" else 2 * KRYLOV_M
            u_b, s_b = coupled_backup_engine(
                ub, sb, W_csr, bias, beta, gamma, t_knots, m=m
            )

        # ---- velocity（直算，禁止重解 ODE）
        v_b = beta.unsqueeze(0) * u_b - gamma.unsqueeze(0) * s_b

        # ---- 写回 CPU memmap
        u_end_mmap[start:end] = u_b.cpu().numpy()
        s_end_mmap[start:end] = s_b.cpu().numpy()
        v_mmap[start:end] = v_b.cpu().numpy()

        # ---- 显式释放 + 清缓存
        del sb, ub, u_b, s_b, v_b, sb_cpu, ub_cpu
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"[L3] batch {start:>7d}-{end:>7d}/{N} | route={route:18s} | "
              f"res={res:.4f} | peak_mem={peak_memory_gb():.2f} GB")

    # ---------------- 全局 alpha（非 per-cell 时基于 s_end 均值计算）
    if alpha_is_per_cell:
        # CPU 端直接流式复制，无需 GPU
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            alpha_mmap[start:end] = alpha_fit_cpu[start:end]
        del alpha_mmap
        alpha_global = None
    else:
        if alpha_fit_cpu is not None and alpha_fit_cpu.ndim == 1:
            # per-gene alpha，直接落盘
            alpha_global = alpha_fit_cpu
            np.save(os.path.join(out_dir, "alpha.npy"), alpha_global)
        else:
            # 基于 s_end 均值重新计算
            s_mean = np.zeros(G, dtype=np.float64)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                s_mean += np.array(s_end_mmap[start:end], dtype=np.float64).sum(axis=0)
            s_mean = (s_mean / N).astype(np.float32)
            s_mean_t = torch.from_numpy(s_mean).to(device=device, dtype=torch.float32)
            z = torch.sparse.mm(W_csr, s_mean_t.unsqueeze(1)).squeeze(1) + bias
            alpha_out = torch.clamp(F.softplus(z), 0.0, ALPHA_CLAMP_MAX)
            alpha_global = alpha_out.cpu().numpy()
            np.save(os.path.join(out_dir, "alpha.npy"), alpha_global)

    # ---------------- 解析 Jacobian（s_end 均值作为代表点）
    s_mean = np.zeros(G, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        s_mean += np.array(s_end_mmap[start:end], dtype=np.float64).sum(axis=0)
    s_mean = (s_mean / N).astype(np.float32)
    s_mean_t = torch.from_numpy(s_mean).to(device=device, dtype=torch.float32)

    jac = analytic_jacobian(W_csr, s_mean_t.unsqueeze(0), beta, gamma, bias)
    J, J_diag = jac["J"], jac["diag"]

    np.savez(
        os.path.join(out_dir, "J.npz"),
        crow=J.crow_indices().cpu().numpy(),
        col=J.col_indices().cpu().numpy(),
        val=J.values().cpu().numpy(),
        diag=J_diag.cpu().numpy(),
        shape=np.array([G, G], dtype=np.int64),
    )

    # ---------------- 保存标量结果与元数据
    max_picard_res = max(picard_residuals) if picard_residuals else 0.0
    max_picard_res_eff = max(picard_residuals_eff) if picard_residuals_eff else 0.0
    engine_route = select_engine(False, max_picard_res_eff, krylov_residual_global)

    # 复制 t / beta / gamma / gate / bias 到输出目录，方便下游自包含
    np.save(os.path.join(out_dir, "t.npy"), t_cpu)
    np.save(os.path.join(out_dir, "beta.npy"), beta_cpu)
    np.save(os.path.join(out_dir, "gamma.npy"), gamma_cpu)
    np.save(os.path.join(out_dir, "gate.npy"), gate_cpu)
    np.save(os.path.join(out_dir, "bias.npy"), bias_cpu)

    meta_out = {
        "engine_route": engine_route,
        "picard_residual_max": max_picard_res,
        "picard_residual_max_eff": max_picard_res_eff,
        "krylov_residual_probe": krylov_residual_global,
        "n_unconverged_batches": n_unconverged,
        "route_counts": route_counts,
        "N": N,
        "G": G,
        "alpha_is_per_cell": alpha_is_per_cell,
    }
    with open(os.path.join(out_dir, "l3_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, default=str)

    # ---------------- 显存红线
    if device == "cuda":
        peak = peak_memory_gb()
        print(f"[L3] 峰值显存: {peak:.2f} GB")
        assert peak < 8.0, f"显存红线违反: {peak:.2f} GB >= 8 GB"

    # 关闭 memmap（确保刷盘）
    del u_end_mmap, s_end_mmap, v_mmap
    del Ms_mmap, Mu_mmap

    print(f"[L3] 完成。输出目录: {out_dir}")
    print(f"[L3] engine_route = {engine_route}")
    return meta_out


# ================================================================ CLI
def main():
    p = argparse.ArgumentParser(description="L3 流式传播（解耦 L2 检查点）")
    p.add_argument("--checkpoint", required=True, help="L2 检查点目录（由 --l2-checkpoint 生成）")
    p.add_argument("--ms", required=True, help="Ms (N,G) .npy 路径（memmap 读取，不整体加载）")
    p.add_argument("--mu", required=True, help="Mu (N,G) .npy 路径")
    p.add_argument("--out-dir", default="l3_output", help="输出目录")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="GPU batch size（默认 2048，12GB 显存安全）")
    p.add_argument("--K", type=int, default=None,
                   help="覆盖 checkpoint 中的 K（分段数），用于 K 敏感性验证")
    p.add_argument("--device", default=None, help="cuda / cpu")
    args = p.parse_args()

    run_l3_streaming(
        ms_path=args.ms,
        mu_path=args.mu,
        checkpoint_dir=args.checkpoint,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
        K=args.K,
    )


if __name__ == "__main__":
    sys.exit(main())
