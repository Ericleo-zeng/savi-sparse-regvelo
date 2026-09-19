#!/usr/bin/env python3
"""
sparse_regvelo_downstream.py — L3 结果验证与下游分析入口（三步合一）。

基于当前运行结果路径：
  L3 输出: /mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_output/l3_output/
  L2 checkpoint: ./l2_checkpoint
  原始输入: /mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input/

三步任务：
  1) velocity 流场可视化验证（采样检查分布，无全量加载）
  2) max=20 边界敏感性检验（定位被截断基因）
  3) 5 模块框架入口（Transition Field / Causal Gene Ranking / Intervention Simulation）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

# ================================================================ CONFIG
L3_OUTPUT_DIR = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_output/l3_output"
L2_CHECKPOINT_DIR = "./l2_checkpoint"
INPUT_DIR = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input"

# 基因名文件（可选）。如果你有 gene_names.npy / var_names.npy / features.npy，填路径：
GENE_NAMES_PATH = None  # 例如: os.path.join(INPUT_DIR, "gene_names.npy")

# 采样步长（velocity 分布验证用，越大内存占用越小）
VELOCITY_SAMPLE_STRIDE = 500

# 扰动模拟参数（Causal Gene Ranking / Intervention Simulation）
PERTURBATION_TOP_K = 50       # 因果排序取前 50 基因
PERTURBATION_DELTA = 0.5      # 干预幅度（过表达/敲低倍数）

# ================================================================ 工具函数
def load_mmap(path: str, dtype=np.float32):
    """安全 memmap 加载，只读。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return np.load(path, mmap_mode="r")


def load_gene_names(g_path: str | None, G: int):
    """加载基因名，若不存在则生成 index 占位符。"""
    if g_path and os.path.exists(g_path):
        names = np.load(g_path)
        if len(names) == G:
            return np.asarray(names, dtype=str)
    return np.array([f"gene_{i}" for i in range(G)], dtype=str)


def load_jacobian(j_path: str):
    """从 J.npz 重建 scipy CSR Jacobian。"""
    d = np.load(j_path)
    crow = d["crow"]
    col = d["col"]
    val = d["val"]
    diag = d["diag"]
    shape = tuple(d["shape"])
    J = csr_matrix((val, col, crow), shape=shape)
    return J, diag


# ================================================================ STEP 1: Velocity 分布验证
def step1_velocity_validation(v_path: str, stride: int = 500):
    """流式采样检查 velocity 分布，不加载全量。"""
    print("=" * 70)
    print("[STEP 1] Velocity 分布验证（memmap 采样）")
    print("=" * 70)

    v = load_mmap(v_path)
    N, G = v.shape
    print(f"  v shape: ({N}, {G})")
    print(f"  采样步长: 每 {stride} 个细胞取 1 个")

    # 采样（避免全量加载）
    idx = np.arange(0, N, stride)
    v_sample = np.array(v[idx], dtype=np.float64)  # 仅采样部分进内存

    p = [0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99]
    perc = np.percentile(v_sample, p)

    print()
    print(f"  Velocity 分位数（采样 {len(idx)} 个细胞）:")
    for pi, vi in zip(p, perc):
        print(f"    p{pi:>6.2f}: {vi:>12.6f}")

    # 异常值诊断
    extreme_pos = (v_sample > 1000).sum()
    extreme_neg = (v_sample < -1000).sum()
    nans = np.isnan(v_sample).sum()
    infs = np.isinf(v_sample).sum()

    print()
    print("  异常值诊断:")
    print(f"    v > 1000 的元素数: {extreme_pos}")
    print(f"    v < -1000 的元素数: {extreme_neg}")
    print(f"    NaN 数: {nans}")
    print(f"    Inf 数: {infs}")

    if extreme_pos + extreme_neg + nans + infs == 0:
        print("  ✅ Velocity 分布健康，无极端异常值")
    else:
        print("  ⚠️  发现异常值，建议检查被截断基因或放宽 L2 边界重跑")

    # 额外：按细胞计算 velocity norm 分布（流式，不加载全量）
    norms = []
    batch = 2048
    for start in range(0, N, batch):
        end = min(start + batch, N)
        vb = np.array(v[start:end], dtype=np.float32)
        norms.append(np.linalg.norm(vb, axis=1))
    norms = np.concatenate(norms)

    print()
    print("  逐细胞 velocity L2-norm:")
    print(f"    median: {np.median(norms):.4f}")
    print(f"    p99:    {np.percentile(norms, 99):.4f}")
    print(f"    max:    {np.max(norms):.4f}")

    return v_sample


# ================================================================ STEP 2: max=20 边界敏感性检验
def step2_boundary_sensitivity(beta_path: str, gamma_path: str, gene_names: np.ndarray):
    """检查 beta/gamma 被硬截断到 20 的基因。"""
    print("=" * 70)
    print("[STEP 2] max=20 边界敏感性检验")
    print("=" * 70)

    beta = np.load(beta_path)
    gamma = np.load(gamma_path)
    G = len(beta)

    print(f"  基因总数: {G}")
    print(f"  beta  range: [{beta.min():.4f}, {beta.max():.4f}] | median={np.median(beta):.4f}")
    print(f"  gamma range: [{gamma.min():.4f}, {gamma.max():.4f}] | median={np.median(gamma):.4f}")

    beta_trunc = beta >= 19.999   # 允许浮点误差
    gamma_trunc = gamma >= 19.999
    either_trunc = beta_trunc | gamma_trunc
    both_trunc = beta_trunc & gamma_trunc

    print()
    print("  截断统计:")
    print(f"    beta  触顶 (≈20):  {beta_trunc.sum()} 基因 ({beta_trunc.sum()/G*100:.2f}%)")
    print(f"    gamma 触顶 (≈20):  {gamma_trunc.sum()} 基因 ({gamma_trunc.sum()/G*100:.2f}%)")
    print(f"    任一触顶:          {either_trunc.sum()} 基因 ({either_trunc.sum()/G*100:.2f}%)")
    print(f"    同时触顶:          {both_trunc.sum()} 基因")

    if either_trunc.sum() > 0:
        print()
        print("  被截断基因列表（前 30 个）:")
        trunc_idx = np.where(either_trunc)[0]
        for i in trunc_idx[:30]:
            flag = ""
            if beta_trunc[i] and gamma_trunc[i]:
                flag = "[B+G]"
            elif beta_trunc[i]:
                flag = "[B]"
            else:
                flag = "[G]"
            print(f"    {i:>5d} {gene_names[i]:>20s} {flag}  beta={beta[i]:.4f} gamma={gamma[i]:.4f}")

        # 保存截断基因列表供人工审查
        out_path = os.path.join(L3_OUTPUT_DIR, "truncated_genes.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("idx\tgene_name\tbeta_trunc\tgamma_trunc\tbeta\tgamma\n")
            for i in trunc_idx:
                f.write(f"{i}\t{gene_names[i]}\t{beta_trunc[i]}\t{gamma_trunc[i]}\t{beta[i]:.6f}\t{gamma[i]:.6f}\n")
        print()
        print(f"  ✅ 截断基因列表已保存: {out_path}")
    else:
        print("  ✅ 无基因触顶，边界未对当前数据产生截断效应")

    return either_trunc


# ================================================================ STEP 3: 5 模块框架入口
def step3_downstream_entrance(
    l3_dir: str,
    beta: np.ndarray,
    gamma: np.ndarray,
    gene_names: np.ndarray,
    truncated_mask: np.ndarray,
):
    """
    提供 5 模块框架的入口代码：
      3a) Transition Field — velocity 流场基础统计
      3b) Causal Gene Ranking — Jacobian 扰动模拟
      3c) Intervention Simulation — 敲低/过表达命运偏移
    所有计算均支持流式/稀疏，不加载全量 (N,G) 到内存。
    """
    print("=" * 70)
    print("[STEP 3] 5 模块框架下游分析入口")
    print("=" * 70)

    v_path = os.path.join(l3_dir, "v.npy")
    s_path = os.path.join(l3_dir, "s_end.npy")
    t_path = os.path.join(l3_dir, "t.npy")
    j_path = os.path.join(l3_dir, "J.npz")

    # ---------------- 3a) Transition Field 基础统计
    print()
    print("[3a] Transition Field — velocity 流场基础统计")
    v = load_mmap(v_path)
    s = load_mmap(s_path)
    t = np.load(t_path)
    N, G = v.shape

    # 流式计算：velocity 与 s 的相关性（代表点：batch 均值）
    v_mean = np.zeros(G, dtype=np.float64)
    s_mean = np.zeros(G, dtype=np.float64)
    batch = 2048
    for start in range(0, N, batch):
        end = min(start + batch, N)
        v_mean += np.array(v[start:end], dtype=np.float64).sum(axis=0)
        s_mean += np.array(s[start:end], dtype=np.float64).sum(axis=0)
    v_mean /= N
    s_mean /= N

    # 速度-状态散度（简单估计：v · s / |s|²）
    divergence_proxy = np.dot(v_mean, s_mean) / (np.dot(s_mean, s_mean) + 1e-12)
    print(f"  全局 velocity·s / |s|² (散度代理): {divergence_proxy:.6f}")
    print(f"  v_mean 前 5 大基因: {gene_names[np.argsort(-np.abs(v_mean))[:5]]}")
    print(f"  s_mean 前 5 大基因: {gene_names[np.argsort(-np.abs(s_mean))[:5]]}")

    # 保存流场代表点供外部可视化
    np.savez(
        os.path.join(l3_dir, "transition_field_proxy.npz"),
        v_mean=v_mean.astype(np.float32),
        s_mean=s_mean.astype(np.float32),
        t=t,
    )
    print("  ✅ transition_field_proxy.npz 已保存")

    # ---------------- 3b) Causal Gene Ranking — Jacobian 扰动模拟
    print()
    print("[3b] Causal Gene Ranking — Jacobian 扰动模拟")
    J, J_diag = load_jacobian(j_path)
    print(f"  J shape: {J.shape}, nnz: {J.nnz}, density: {J.nnz / (G*G) * 100:.4f}%")

    # 策略：对每个基因 g，模拟将其 s_g 增加 1 个单位，看对全局 velocity 的影响
    # v_delta = J @ e_g = J 的第 g 列（稀疏取列）
    # 由于 J 是 (G,G) CSR，取列效率低，改为逐行扫描非零模式

    causal_scores = np.zeros(G, dtype=np.float64)
    for g in range(G):
        # J 的第 g 列 = 所有行 i 中 J[i,g] 非零的值
        # 用 CSC 转置 trick：J.T 的 CSR 就是 J 的 CSC
        col = J.T.tocsr()[g]  # 1 x G sparse row
        causal_scores[g] = np.abs(col.data).sum()  # L1 影响幅度

    top_causal_idx = np.argsort(-causal_scores)[:PERTURBATION_TOP_K]
    print()
    print(f"  因果排序 Top {PERTURBATION_TOP_K}（按 |J| L1 列和）:")
    for rank, idx in enumerate(top_causal_idx, 1):
        trunc_flag = "[TRUNCATED]" if truncated_mask[idx] else ""
        print(f"    {rank:>3d}. {gene_names[idx]:>20s}  score={causal_scores[idx]:.4f}  {trunc_flag}")

    np.savez(
        os.path.join(l3_dir, "causal_gene_ranking.npz"),
        gene_idx=top_causal_idx,
        scores=causal_scores[top_causal_idx].astype(np.float32),
        all_scores=causal_scores.astype(np.float32),
    )
    print("  ✅ causal_gene_ranking.npz 已保存")

    # ---------------- 3c) Intervention Simulation — 敲低/过表达命运偏移
    print()
    print("[3c] Intervention Simulation — 敲低/过表达命运偏移")

    # 选择因果排序 Top 10 做干预模拟
    top10 = top_causal_idx[:10]
    # 代表细胞：s_mean（可用，已算好）
    s0 = s_mean.astype(np.float32)
    v0 = v_mean.astype(np.float32)

    print(f"  基线 velocity norm: {np.linalg.norm(v0):.4f}")
    print(f"  干预 Top 10 基因，delta = ±{PERTURBATION_DELTA}")

    intervention_results = []
    for idx in top10:
        for direction, label in [(1, "OE"), (-1, "KD")]:
            s_pert = s0.copy()
            s_pert[idx] += direction * PERTURBATION_DELTA * s0[idx]  # 相对干预
            # v_pert = beta * u - gamma * s，但 u 未知。简化：用 Jacobian 线性近似
            # delta_v ≈ J @ delta_s
            delta_s = np.zeros(G, dtype=np.float32)
            delta_s[idx] = direction * PERTURBATION_DELTA * s0[idx]
            delta_v = J.dot(delta_s)  # 稀疏矩阵-向量乘法
            v_pert = v0 + delta_v
            delta_norm = np.linalg.norm(v_pert) - np.linalg.norm(v0)
            intervention_results.append({
                "gene": gene_names[idx],
                "idx": int(idx),
                "direction": label,
                "delta_v_norm": float(delta_norm),
                "v_pert_norm": float(np.linalg.norm(v_pert)),
            })
            print(f"    {gene_names[idx]:>20s} {label}  Δ|v|={delta_norm:+.4f}")

    # 保存
    with open(os.path.join(l3_dir, "intervention_simulation.json"), "w", encoding="utf-8") as f:
        json.dump(intervention_results, f, indent=2, default=str)
    print("  ✅ intervention_simulation.json 已保存")

    print()
    print("=" * 70)
    print("[DONE] 三步验证完成，所有结果已落盘到 L3 输出目录")
    print("=" * 70)


# ================================================================ MAIN
def main():
    print("sparse_regvelo_downstream.py — L3 结果验证与下游分析入口")
    print(f"L3 输出目录: {L3_OUTPUT_DIR}")
    print(f"L2 checkpoint: {L2_CHECKPOINT_DIR}")
    print(f"基因名文件: {GENE_NAMES_PATH or '(未提供，使用 gene_i 占位符)'}")

    # 加载基础参数
    beta = np.load(os.path.join(L3_OUTPUT_DIR, "beta.npy"))
    gamma = np.load(os.path.join(L3_OUTPUT_DIR, "gamma.npy"))
    G = len(beta)
    gene_names = load_gene_names(GENE_NAMES_PATH, G)

    # STEP 1: Velocity 验证
    v_sample = step1_velocity_validation(
        os.path.join(L3_OUTPUT_DIR, "v.npy"),
        stride=VELOCITY_SAMPLE_STRIDE,
    )

    # STEP 2: 边界敏感性
    truncated_mask = step2_boundary_sensitivity(
        os.path.join(L3_OUTPUT_DIR, "beta.npy"),
        os.path.join(L3_OUTPUT_DIR, "gamma.npy"),
        gene_names,
    )

    # STEP 3: 下游入口
    step3_downstream_entrance(
        L3_OUTPUT_DIR,
        beta,
        gamma,
        gene_names,
        truncated_mask,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
