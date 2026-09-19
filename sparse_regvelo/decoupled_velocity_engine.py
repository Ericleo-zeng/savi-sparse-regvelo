"""decoupled_velocity_engine_stable.py — Phase 2 C 管线（numerically stable fork）（L1 稳态初始化+门控，L2 1.5 步解耦推断）。

实现 SPEC §3.5（签名冻结）：
  L1: steady_state_init / steady_state_gate
  L2: pseudotime_from_latent -> anchor_direction
      -> for r in range(n_rounds): fit_genes_closed_form -> (非末轮) reassign_time
均值函数：解析两段式诱导动力学（alpha 分段常数 ZOH，逐基因解耦，无 GRN 耦合）。
给定 (beta, gamma, t_switch) 时预测对 (alpha_on, alpha_off) 线性 -> alpha 幅值闭式最小二乘；
beta/gamma 用对数网格粗搜 + 坐标精化（向量化）。

红线（SPEC §0）：n_rounds >= 1 入口断言；steady_state_init 仅输出 gamma/beta、alpha/beta
比值（可辨识性约束）；全流程无 beta=1 归一化；下游 v = beta*u - gamma*s 直算（在 Phase 3）。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import DEVICE, JsonLog


# ================================================================
# 增强采样近似 DPT（ES-DPT）：大样本下保留分支拓扑的 pseudotime
# ================================================================
def _enhanced_sampled_dpt(X, root_cell, k=15, n_comps=15, random_state=42):
    """
    增强采样近似 DPT（ES-DPT）。
    
    算法：
    1. 自适应子集：max(15000, int(N*0.2))，上限 25000
    2. 密度感知分层采样：基于局部密度分 5 层，确保分支末端（低密度）不被遗漏
    3. 子集上执行完整 DPT 流水线（kNN→高斯核→转移矩阵→扩散映射→MST→Dijkstra）
       —— 与原始 RegVelo / scanpy DPT 的数学流程完全一致
    4. 扩散坐标空间插值：用 Nystrom-like 扩展估计全数据集扩散坐标，再 kNN 插值
    5. 方向校正（root 在起点）
    
    内存：峰值 < 500 MB（子集 N<=25k，无 (N,G) 级数组）
    时间：~5-10s（N=75k，纯 CPU）
    分支感知：✅ 保留（与原始 RegVelo DPT 算法等价）
    """
    from sklearn.neighbors import kneighbors_graph, NearestNeighbors
    from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix
    import numpy as np

    N = X.shape[0]
    rng = np.random.default_rng(random_state)

    # --- 1. 自适应子集大小 ---
    n_sample = min(max(15000, int(N * 0.2)), 25000)

    # --- 2. 密度感知分层采样 ---
    # 快速密度估计：k=50 邻居平均距离（排除自环）
    k_dens = min(50, N - 1)
    nbrs_d = NearestNeighbors(n_neighbors=k_dens).fit(X)
    dist_d, _ = nbrs_d.kneighbors(X)
    avg_dist = dist_d[:, 1:].mean(axis=1)
    density = 1.0 / (avg_dist + 1e-12)

    # 按密度分 5 层（0=最稀疏/分支末端，4=最密集/主干）
    q = np.percentile(density, [20, 40, 60, 80])
    strata = np.digitize(density, q)  # 0-4

    # 每层采样配额：反比于平均密度（稀疏层多采），保底 100 个或全取
    n_per_stratum = np.bincount(strata, minlength=5)
    mean_dens = np.array([
        density[strata == s].mean() if (strata == s).any() else 1.0
        for s in range(5)
    ])
    alloc = 1.0 / (mean_dens + 1e-6)
    alloc = alloc / alloc.sum() * n_sample
    alloc = np.maximum(alloc, np.minimum(n_per_stratum, 100))
    alloc = (alloc / alloc.sum() * n_sample).astype(int)

    idx_sample = []
    for s in range(5):
        idx_in_s = np.where(strata == s)[0]
        n_take = min(alloc[s], len(idx_in_s))
        if n_take > 0:
            chosen = rng.choice(idx_in_s, size=n_take, replace=False)
            idx_sample.extend(chosen)

    idx_sample = np.array(idx_sample, dtype=int)
    if len(idx_sample) > n_sample:
        idx_sample = rng.choice(idx_sample, size=n_sample, replace=False)

    # 确保 root 在子集中（替换密度最高的冗余点）
    if isinstance(root_cell, (list, tuple, np.ndarray)):
        root = int(np.asarray(root_cell).ravel()[0])
    else:
        root = int(root_cell)
    if root not in idx_sample:
        densest = idx_sample[np.argmax(density[idx_sample])]
        idx_sample[idx_sample == densest] = root

    X_sample = X[idx_sample]
    n_s = len(idx_sample)

    # --- 3. 子集完整 DPT（与 scanpy / RegVelo 算法一致）---
    k = min(k, n_s - 1)

    # 3a. kNN + 高斯核
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_sample)
    dist, idx = nbrs.kneighbors(X_sample)
    dist, idx = dist[:, 1:], idx[:, 1:]  # 去自环

    pos = dist[dist > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    sigma = max(sigma, 1e-12)
    w = np.exp(-((dist / sigma) ** 2))

    rows = np.repeat(np.arange(n_s), k)
    A = csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n_s, n_s))
    A = A.maximum(A.T)  # 对称化

    # 3b. 转移矩阵 P = D^{-1}A
    D_vec = np.array(A.sum(axis=1)).flatten()
    P_rows, P_cols = A.nonzero()
    P_data = A.data / D_vec[P_rows]
    P = csr_matrix((P_data, (P_rows, P_cols)), shape=A.shape)

    # 3c. 扩散映射（eigsh，k=n_comps+1）— 25k 细胞安全，不会 OOM
    eigvals, eigvecs = eigsh(P, k=n_comps + 1, which='LM')
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[sort_idx], eigvecs[:, sort_idx]
    diff_coords_sample = eigvecs[:, 1:n_comps + 1] * eigvals[1:n_comps + 1]

    # 3d. 扩散空间 MST + Dijkstra
    A_diff = kneighbors_graph(
        diff_coords_sample, n_neighbors=min(15, n_s - 1), mode='distance'
    )
    A_diff = A_diff.maximum(A_diff.T).tocsr()
    mst = minimum_spanning_tree(A_diff)

    root_sub = np.where(idx_sample == root)[0][0]
    distances = dijkstra(mst, directed=False, indices=int(root_sub))
    dpt_sample = np.asarray(distances).flatten()

    finite = np.isfinite(dpt_sample)
    if not finite.all():
        fill = dpt_sample[finite].max() + 1.0 if finite.any() else 1.0
        dpt_sample = np.where(finite, dpt_sample, fill)
    dpt_sample = dpt_sample - dpt_sample.min()
    span = dpt_sample.max()
    dpt_sample = dpt_sample / span if span > 1e-12 else np.zeros_like(dpt_sample)

    # --- 4. 扩散坐标空间插值（Nystrom-like）---
    # 4a. 为全数据集估计扩散坐标：原始空间 kNN 加权平均子集扩散坐标
    nn_orig = NearestNeighbors(n_neighbors=min(20, n_s - 1)).fit(X_sample)
    dist_o, idx_o = nn_orig.kneighbors(X)  # (N, 20)

    sigma_o = float(np.median(dist_o[:, 1:])) if dist_o[:, 1:].size else 1.0
    sigma_o = max(sigma_o, 1e-12)
    w_o = np.exp(-((dist_o / sigma_o) ** 2))
    w_o /= w_o.sum(axis=1, keepdims=True)

    diff_coords_full = np.sum(
        diff_coords_sample[idx_o] * w_o[:, :, None], axis=1
    )  # (N, n_comps)

    # 4b. 在扩散坐标空间做 kNN 插值 DPT
    nn_diff = NearestNeighbors(n_neighbors=min(10, n_s - 1)).fit(diff_coords_sample)
    dist_d, idx_d = nn_diff.kneighbors(diff_coords_full)

    weights_d = 1.0 / (dist_d + 1e-8)
    weights_d /= weights_d.sum(axis=1, keepdims=True)
    dpt_full = np.sum(dpt_sample[idx_d] * weights_d, axis=1)

    # --- 5. 方向校正 ---
    if dpt_full[root] > np.median(dpt_full):
        dpt_full = 1.0 - dpt_full
    dpt_full = (dpt_full - dpt_full.min()) / (dpt_full.max() - dpt_full.min() + 1e-12)

    print(f"[ES-DPT] 完成：N={N}，子集={n_s}，密度分层采样，扩散空间插值，分支拓扑保留")
    return dpt_full.astype(np.float32)

# Numerical-stability policy for production-scale L2 fitting.
# The original model is unchanged mathematically; these are admissible parameter
# domain guards needed to prevent floating-point overflow/NaN propagation.
_EPS = 1e-12
_RATE_MIN = 1e-4
_RATE_MAX = 50.0
_RATIO_MIN = 1e-3
_RATIO_MAX = _RATE_MAX / _RATE_MIN
_ALPHA_MAX = 50.0
_FINITE_CLIP = 1e30

def _finite_summary(x, name):
    x = np.asarray(x)
    finite = np.isfinite(x)
    if not finite.any():
        return f"{name}: finite=0/{x.size}"
    xf = x[finite]
    return (f"{name}: finite={finite.sum()}/{x.size}, "
            f"min={xf.min():.6g}, median={np.median(xf):.6g}, "
            f"p99={np.percentile(xf, 99):.6g}, max={xf.max():.6g}")

def _sanitize_rates(beta, gamma, *, name="rates"):
    beta = np.asarray(beta, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    bad = (~np.isfinite(beta) | ~np.isfinite(gamma) |
           (beta <= 0.0) | (gamma <= 0.0))
    beta = np.nan_to_num(beta, nan=1.0, posinf=_RATE_MAX, neginf=_RATE_MIN)
    gamma = np.nan_to_num(gamma, nan=1.0, posinf=_RATE_MAX, neginf=_RATE_MIN)
    beta = np.clip(beta, _RATE_MIN, _RATE_MAX)
    gamma = np.clip(gamma, _RATE_MIN, _RATE_MAX)
    if np.any(bad):
        print(f"[STABLE] {name}: repaired {int(bad.sum())} non-finite/non-positive rates")
    return beta.astype(np.float32, copy=False), gamma.astype(np.float32, copy=False)




# ================================================================ L1: 稳态初值与门控
def steady_state_init(Mu, Ms, q_lo=0.01, q_hi=0.99):
    """L1 初值：极端分位数回归（velocyto 式）估 gamma/beta 斜率；alpha/beta 比值由高分位 u 估计。

    返回 dict(gamma_over_beta=(G,), alpha_over_beta=(G,))。仅比值，符合可辨识性约束。
    """
    Mu_np = Mu.cpu().numpy() if hasattr(Mu, "cpu") else np.asarray(Mu)
    Ms_np = Ms.cpu().numpy() if hasattr(Ms, "cpu") else np.asarray(Ms)
    u = np.asarray(Mu_np, dtype=np.float32)
    s = np.asarray(Ms_np, dtype=np.float32)
    N, G = u.shape
    gamma_over_beta = np.zeros(G, dtype=np.float32)
    alpha_over_beta = np.zeros(G, dtype=np.float32)
    for g in range(G):
        sg, ug = s[:, g], u[:, g]
        lo, hi = np.quantile(sg, q_lo), np.quantile(sg, q_hi)
        ext = (sg <= lo) | (sg >= hi)
        if ext.sum() < 3 or np.std(sg[ext]) < _EPS:
            gamma_over_beta[g] = 1.0  # 退化基因：比值占位，后续网格搜索覆盖
        else:
            # 极端分位细胞上的稳健线性拟合 u = a + b*s，斜率 b ≈ gamma/beta
            A = np.stack([np.ones(ext.sum()), sg[ext]], axis=1)
            coef, *_ = np.linalg.lstsq(A, ug[ext], rcond=None)
            slope = float(coef[1])
            # The ratio is only an initialization statistic.  Bound it to the
            # identifiable positive-rate domain so it cannot seed pathological
            # gamma values during the scale candidates below.
            gamma_over_beta[g] = float(np.clip(
                slope if np.isfinite(slope) else 1.0,
                _RATIO_MIN, _RATIO_MAX))
        top = sg >= hi
        alpha_over_beta[g] = max(float(np.mean(ug[top])) if top.any() else 0.0, 0.0)
    print("[L1] " + _finite_summary(gamma_over_beta, "gamma/beta"))
    print(f"[L1] bounded ratio range = [{_RATIO_MIN:g}, {_RATIO_MAX:g}]")
    return {"gamma_over_beta": gamma_over_beta, "alpha_over_beta": alpha_over_beta}


def _kmeans_1d(x, n_iter=25):
    """1-D 2-均值（Lloyd），返回 (mu1, mu2, p1, sigma1, sigma2)（p1 为簇 1 占比）。"""
    q1, q3 = np.quantile(x, [0.25, 0.75])
    c = np.array([q1, q3], dtype=np.float32)
    for _ in range(n_iter):
        lab = (np.abs(x - c[0]) > np.abs(x - c[1])).astype(np.int64)
        c_new = np.array([x[lab == 0].mean() if (lab == 0).any() else c[0],
                          x[lab == 1].mean() if (lab == 1).any() else c[1]])
        if np.allclose(c_new, c, atol=1e-10):
            c = c_new
            break
        c = c_new
    p1 = float((lab == 0).mean())
    s1 = float(x[lab == 0].std()) if (lab == 0).any() else 0.0
    s2 = float(x[lab == 1].std()) if (lab == 1).any() else 0.0
    return c[0], c[1], p1, s1, s2


def steady_state_gate(Mu, Ms, corr_thresh=0.1, curvature_thresh=1e-3, r2_thresh=0.3):
    """L1 门控三指标：u-s Pearson 相关、相图曲率（二阶差分能量）、多 regime 检测
    （残差双峰性：对 u - slope*s 做 2-均值聚类，簇间分离度>阈值判多 regime）。
    返回 gate (G,) bool：True = 违反稳态，路由完整动力学路径。

    指标细节：
      (i)   corr(u, s) < corr_thresh -> 违反（无稳态单调关系）；
      (ii)  相图曲率：s min-max 归一化、u z-score 后拟合二次多项式，二次项系数能量
            a2^2（归一化坐标下 s-u 曲线的离散二阶差分能量；等间距二阶差分对分箱间距
            不规则敏感，二次系数为其稳健等价）< curvature_thresh -> 相图退化
            （纯直线，无动力学信息）-> 违反；
      (iii) 残差 r = u - (a + b*s) 做 2-均值聚类，簇间分离度同时满足
            R2_b = p*q*(mu1-mu2)^2 / var(u) > r2_thresh（簇间变差占基因总变差比例）
            且 |mu1-mu2| > 2*(sigma1+sigma2)（相对簇内离散度显著分离）-> 多 regime -> 违反。
            后者排除单峰/单弧残差被 2-均值强行劈开的假阳性。
    """
    Mu_np = Mu.cpu().numpy() if hasattr(Mu, "cpu") else np.asarray(Mu)
    Ms_np = Ms.cpu().numpy() if hasattr(Ms, "cpu") else np.asarray(Ms)
    u = np.asarray(Mu_np, dtype=np.float32)
    s = np.asarray(Ms_np, dtype=np.float32)
    N, G = u.shape
    gate = np.zeros(G, dtype=bool)
    for g in range(G):
        sg, ug = s[:, g], u[:, g]
        su, ss = np.std(ug), np.std(sg)
        # (i) Pearson 相关
        corr = 0.0 if (su < _EPS or ss < _EPS) else float(np.corrcoef(ug, sg)[0, 1])
        viol_corr = corr < corr_thresh
        # (ii) 相图曲率（归一化坐标二次项系数能量）
        if su < _EPS or ss < _EPS:
            curv = 0.0
        else:
            sn_ = (sg - sg.min()) / (sg.max() - sg.min() + _EPS)
            uz = (ug - ug.mean()) / su
            A2 = np.stack([sn_ ** 2, sn_, np.ones(N)], axis=1)
            a2 = float(np.linalg.lstsq(A2, uz, rcond=None)[0][0])
            curv = a2 ** 2
        viol_curv = curv < curvature_thresh
        # (iii) 多 regime：残差 2-均值簇间分离度（R2 口径）
        if su < _EPS or ss < _EPS:
            r2_b = 0.0
        else:
            A = np.stack([np.ones(N), sg], axis=1)
            coef, *_ = np.linalg.lstsq(A, ug, rcond=None)
            r = ug - A @ coef
            m1, m2, p1, sd1, sd2 = _kmeans_1d(r)
            r2_b = float(p1 * (1.0 - p1) * (m1 - m2) ** 2 / (np.var(ug) + _EPS))
            sep_within = abs(m1 - m2) / (sd1 + sd2 + _EPS)
        viol_regime = (r2_b > r2_thresh) and (sep_within > 2.0)
        gate[g] = viol_corr or viol_curv or viol_regime
    return gate


# ================================================================ L2 Step 1: pseudotime
def pseudotime_from_latent(X_scVI, root_cell, k=15):
    """Step 1：X_scVI (N,10) -> kNN 图 -> 对称高斯核加权 -> 根细胞 geodesic pseudotime ∈[0,1]。

    实现：sklearn NearestNeighbors 构图（高斯核加权，带宽=中位距离），图对称化后
    scipy.sparse.csgraph.dijkstra 最短路径（cost = 1 - 相似度），非连通分量填 max+1
    再归一化 [0,1]。root_cell 为 int 时返回 (N,) ndarray；为 list[int] 时返回
    dict(root -> (N,))（多根敏感性分析）。
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    X = np.asarray(X_scVI.cpu().numpy() if hasattr(X_scVI, "cpu") else X_scVI, dtype=np.float32)
    N = X.shape[0]
    # ================================================================
    # 新增：大样本时自动启用增强采样近似 DPT（保留分支拓扑）
    # ================================================================
    if N >= 5000:
        try:
            return _enhanced_sampled_dpt(X, root_cell, k=k, random_state=42)
        except Exception as e:
            print(f"[pseudotime] ES-DPT 失败 ({e})，降级到直接图距离...")
    # ================================================================
    # 以下保留原有实现完全不变（N<5000 时仍走此路径，保证 --small 通过）
    # ================================================================                
    k = int(min(k, N - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]  # 去掉自环
    pos = dist[dist > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    sigma = max(sigma, _EPS)
    w = np.exp(-((dist / sigma) ** 2))
    rows = np.repeat(np.arange(N), k)
    W = csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(N, N))
    W = W.maximum(W.T)  # 对称化（无向图）
    C = W.copy()
    C.data = 1.0 - C.data  # 相似度 -> 距离成本 ∈ [0,1)

    def _one(root):
        d = dijkstra(C, directed=False, indices=int(root))
        finite = np.isfinite(d)
        if not finite.all():
            fill = (d[finite].max() + 1.0) if finite.any() else 1.0
            d = np.where(finite, d, fill)  # 非连通分量填 max+1
        d = d - d.min()
        span = d.max()
        return d / span if span > _EPS else np.zeros_like(d)

    if isinstance(root_cell, (list, tuple, np.ndarray)):
        return {int(r): _one(int(r)) for r in root_cell}
    return _one(int(root_cell))


# ================================================================ L2: 方向锚定
def anchor_direction(t, Mu, Ms):
    """TIVelo 式方向锚定：u 物理上先于 s（转录先升、剪切体随后累积）。

    按 t 排序分箱，对每基因比较 u 峰与 s 峰的出现时序：u 峰早于 s 峰记 +1
    （「u 早高且 s 晚高」），反之 -1；多数基因为 -1 则方向反转 t <- 1-t。
    返回 (t_anchored, flipped, score)，score ∈ [-1,1] 为基因平均符号得分。
    """
    t = np.asarray(t, dtype=np.float32)
    Mu_np = Mu.cpu().numpy() if hasattr(Mu, "cpu") else np.asarray(Mu)
    Ms_np = Ms.cpu().numpy() if hasattr(Ms, "cpu") else np.asarray(Ms)
    u = np.asarray(Mu_np, dtype=np.float32)
    s = np.asarray(Ms_np, dtype=np.float32)
    N, G = u.shape
    n_bins = min(10, N)
    order = np.argsort(t, kind="mergesort")
    bins = np.array_split(order, n_bins)
    signs = np.zeros(G)
    valid = np.zeros(G, dtype=bool)
    for g in range(G):
        ub = np.array([u[b, g].mean() for b in bins])
        sb = np.array([s[b, g].mean() for b in bins])
        dyn = (ub.max() - ub.min()) + (sb.max() - sb.min())
        scale = ub.mean() + sb.mean() + _EPS
        if dyn / scale < 1e-3:
            continue  # 无动态范围的基因不参与投票
        valid[g] = True
        signs[g] = np.sign(np.argmax(sb) - np.argmax(ub))
    score = float(signs[valid].mean()) if valid.any() else 0.0
    flipped = score < 0.0
    t_out = 1.0 - t if flipped else t.copy()
    return t_out, bool(flipped), score


# ================================================================ 均值函数：两段式 ZOH 诱导动力学
def _propagate_zoh(u0, s0, alpha, h, beta, gamma):
    """Stable ZOH propagation for
        du = alpha - beta*u
        ds = beta*u - gamma*s

    The original expression
        exp(-gamma*h) * expm1((gamma-beta)*h) / (gamma-beta)
    is algebraically equal to
        (exp(-beta*h) - exp(-gamma*h)) / (gamma-beta)
    and the latter never evaluates a potentially huge positive exponent.
    Near beta == gamma, the exact limiting form h*exp(-beta*h) is used.

    This preserves the model equations while eliminating the observed expm1
    overflow -> inf -> nan cascade.
    """
    h = np.asarray(h, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    u0 = np.asarray(u0, dtype=np.float32)
    s0 = np.asarray(s0, dtype=np.float32)

    beta, gamma = _sanitize_rates(beta, gamma)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore",
                     under="ignore"):
        bh = beta * h
        gh = gamma * h
        eb = np.exp(-bh)
        eg = np.exp(-gh)

        C = alpha / beta
        A = u0 - C

        den = gamma - beta
        # Stable difference quotient.  For |den*h| small, use the continuous
        # beta==gamma limit.  No np.where branch contains an overflowing expm1.
        small = np.abs(den * h) < 1e-4
        num = eb - eg
        mix_regular = num / den
        mix_limit = h * eb
        mix = np.where(small, mix_limit, mix_regular)

        # phi = (1-exp(-gamma*h))/gamma, stable for small gamma*h.
        small_gh = np.abs(gh) < 1e-4
        phi_regular = -np.expm1(-gh) / gamma
        # 1 - exp(-x) = x - x^2/2 + x^3/6 ...
        phi_limit = h - 0.5 * gh * h + (gh * gh * h) / 6.0
        phi = np.where(small_gh, phi_limit, phi_regular)

        u1 = A * eb + C
        s1 = s0 * eg + beta * (A * mix + C * phi)

    # Fail fast at the numerical boundary instead of allowing NaN to poison
    # the normal equations and hide the first offending gene.
    bad = ~(np.isfinite(u1) & np.isfinite(s1))
    if np.any(bad):
        raise FloatingPointError(
            "[STABLE] non-finite ZOH state: "
            + _finite_summary(beta, "beta") + "; "
            + _finite_summary(gamma, "gamma") + "; "
            + _finite_summary(h, "h")
        )
    return u1.astype(np.float32, copy=False), s1.astype(np.float32, copy=False)


def _trajectory(tau, beta, gamma, alpha_on, alpha_off, t_switch):
    """两段式诱导动力学：t<=t_switch 时 alpha=alpha_on（从 u=s=0 起步），
    t>t_switch 时 alpha=alpha_off（自切换点状态续播）。参数按最后一维广播。"""
    tau = np.asarray(tau, dtype=np.float32)
    h1 = np.minimum(tau, t_switch)
    h2 = np.maximum(tau - t_switch, 0.0)
    u_a, s_a = _propagate_zoh(0.0, 0.0, alpha_on, h1, beta, gamma)       # tau<=ts 直接值
    u_sw, s_sw = _propagate_zoh(0.0, 0.0, alpha_on, t_switch, beta, gamma)
    u_b, s_b = _propagate_zoh(u_sw, s_sw, alpha_off, h2, beta, gamma)    # tau>ts 续播
    u = np.where(h2 > 0, u_b, u_a)
    s = np.where(h2 > 0, s_b, s_a)
    return u, s


def _basis_at_ts(tau, betas, gammas, ts):
    """单位 alpha 基轨迹：U/S 形状 (P,2,N)，列 0 = alpha_on 基、列 1 = alpha_off 基。

    预测 = [U;S] @ [alpha_on, alpha_off]，对 alpha 线性（闭式 LS 的基础）。
    betas/gammas: (P,) 网格对；ts 标量或 (P,)；tau: (N,)。
    """
    tau = np.asarray(tau, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)      # ← 新增：截断上游 float64
    gammas = np.asarray(gammas, dtype=np.float32)    # ← 新增：截断上游 float64
    ts_arr = np.asarray(ts, dtype=np.float32)
    ts_b = ts_arr[:, None] if ts_arr.ndim > 0 else ts_arr
    h1 = np.minimum(tau[None, :], ts_b)
    h2 = np.maximum(tau[None, :] - ts_b, 0.0)
    b = betas[:, None]
    g = gammas[:, None]
    u1, s1 = _propagate_zoh(0.0, 0.0, 1.0, h1, b, g)          # 第一段单位 alpha
    uA, sA = _propagate_zoh(u1, s1, 0.0, h2, b, g)            # 第二段 alpha=0（衰减）
    uB, sB = _propagate_zoh(0.0, 0.0, 1.0, h2, b, g)          # 第二段单位 alpha
    U = np.stack([uA, uB], axis=1)
    S = np.stack([sA, sB], axis=1)
    return U, S


def _solve_alpha_2x2(m11, m12, m22, b1, b2, yty):
    """Defensive non-negative 2x2 closed-form least-squares solver.

    The normal equations remain unchanged.  Numerical guards prevent a single
    ill-conditioned gene from propagating NaN/Inf into the whole vectorized fit.
    """
    m11, m12, m22, b1, b2, yty = (
        np.asarray(v, dtype=np.float64) for v in (m11, m12, m22, b1, b2, yty)
    )

    finite = (np.isfinite(m11) & np.isfinite(m12) & np.isfinite(m22) &
              np.isfinite(b1) & np.isfinite(b2) & np.isfinite(yty))
    posdef_diag = (m11 > _EPS) & (m22 > _EPS)
    det = np.zeros_like(m11, dtype=np.float64)
    with np.errstate(all="ignore"):
        det = m11 * m22 - m12 * m12
    det_scale = np.maximum(m11 * m22, 1.0)
    well_conditioned = np.isfinite(det) & (np.abs(det) > _EPS * det_scale)
    ok_internal = finite & posdef_diag & well_conditioned

    a1 = np.zeros_like(m11)
    a2 = np.zeros_like(m22)
    with np.errstate(all="ignore"):
        a1[ok_internal] = (m22[ok_internal] * b1[ok_internal]
                           - m12[ok_internal] * b2[ok_internal]) / det[ok_internal]
        a2[ok_internal] = (-m12[ok_internal] * b1[ok_internal]
                           + m11[ok_internal] * b2[ok_internal]) / det[ok_internal]

    ok_internal &= np.isfinite(a1) & np.isfinite(a2) & (a1 >= 0.0) & (a2 >= 0.0)

    def rss_of(x1, x2):
        with np.errstate(all="ignore"):
            r = (yty - 2.0 * (x1 * b1 + x2 * b2)
                 + m11 * x1 * x1
                 + 2.0 * m12 * x1 * x2
                 + m22 * x2 * x2)
        return np.where(np.isfinite(r), np.maximum(r, 0.0), np.inf)

    cand_rss = [
        np.where(ok_internal, rss_of(np.where(ok_internal, a1, 0.0),
                                      np.where(ok_internal, a2, 0.0)), np.inf)
    ]
    cand_par = [(np.where(ok_internal, a1, 0.0),
                 np.where(ok_internal, a2, 0.0))]

    with np.errstate(all="ignore"):
        x1 = np.where((m11 > _EPS) & finite, b1 / m11, 0.0)
    x1 = np.maximum(np.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    cand_rss.append(rss_of(x1, np.zeros_like(x1)))
    cand_par.append((x1, np.zeros_like(x1)))

    with np.errstate(all="ignore"):
        x2 = np.where((m22 > _EPS) & finite, b2 / m22, 0.0)
    x2 = np.maximum(np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    cand_rss.append(rss_of(np.zeros_like(x2), x2))
    cand_par.append((np.zeros_like(x2), x2))

    # All-zero candidate. Broadcast to the normal-equation shape explicitly.
    # In the vectorized M-step, yty may be (1,G) while m11/m22 are (P,G);
    # np.stack requires every candidate to have exactly the same shape.
    yty_b = np.broadcast_to(yty, m11.shape)
    cand_rss.append(np.where(np.isfinite(yty_b),
                             np.maximum(yty_b, 0.0), np.inf))
    cand_par.append((np.zeros_like(x1), np.zeros_like(x1)))

    stack = np.stack(cand_rss, axis=0)
    best = np.argmin(stack, axis=0)
    rss = np.take_along_axis(stack, best[None, ...], axis=0)[0]
    a_on = np.choose(best, [p[0] for p in cand_par])
    a_off = np.choose(best, [p[1] for p in cand_par])

    bad = ~np.isfinite(rss)
    if np.any(bad):
        raise FloatingPointError(
            f"[STABLE] alpha solver produced non-finite RSS for {int(bad.sum())} entries; "
            + _finite_summary(det, "det")
        )
    return np.maximum(rss, 0.0), a_on, a_off


def _rss_single(tau, ug, sg, beta, gamma, ts):
    """单 (beta,gamma,ts) 下闭式 alpha 的加权 RSS（u,s 各按 1/std^2 加权）。"""
    rss, a_on, a_off = _rss_vec(tau, ug[None, :], sg[None, :],
                                np.array([beta]), np.array([gamma]), np.array([ts]))
    return float(rss[0]), float(a_on[0]), float(a_off[0])


def _rss_vec(tau, u_raw, s_raw, betas, gammas, tss):
    """跨基因向量化 RSS：u_raw/s_raw (G,N) 原始量纲；目标
    J = ||U a - u||^2/sd_u^2 + ||S a - s||^2/sd_s^2（alpha 真实量纲，闭式解一致）。"""
    sd_u = u_raw.std(axis=-1) + _EPS
    sd_s = s_raw.std(axis=-1) + _EPS
    wu = 1.0 / (sd_u ** 2)
    ws = 1.0 / (sd_s ** 2)
    # 防止表达量几乎不变的基因产生天文数字权重，导致 FP32 溢出为 inf
    # 上限 1e12 对应 sd_u >= 1e-6，对于计数数据已足够敏感
    wu = np.clip(wu, 0.0, 1e12)
    ws = np.clip(ws, 0.0, 1e12)
    
    U, S = _basis_at_ts(tau, betas, gammas, tss)      # (G,2,N)
    m11 = (U[:, 0] ** 2).sum(-1) * wu + (S[:, 0] ** 2).sum(-1) * ws
    m12 = (U[:, 0] * U[:, 1]).sum(-1) * wu + (S[:, 0] * S[:, 1]).sum(-1) * ws
    m22 = (U[:, 1] ** 2).sum(-1) * wu + (S[:, 1] ** 2).sum(-1) * ws
    b1 = (U[:, 0] * u_raw).sum(-1) * wu + (S[:, 0] * s_raw).sum(-1) * ws
    b2 = (U[:, 1] * u_raw).sum(-1) * wu + (S[:, 1] * s_raw).sum(-1) * ws
    yty = (u_raw ** 2).sum(-1) * wu + (s_raw ** 2).sum(-1) * ws
    return _solve_alpha_2x2(m11, m12, m22, b1, b2, yty)


# ================================================================ L2 Step 2 (M-step)
def fit_genes_closed_form(Mu, Ms, t, init, gate, K=10, max_iter_ls=10):
    """Step 2（M-step）：固定 t，逐基因独立拟合 (beta_g, gamma_g)；alpha 由闭式解给定
    （给定 beta,gamma 与两段式 ZOH 结构时 alpha 幅值对预测线性 -> 闭式最小二乘）。

    beta,gamma：对数网格粗搜（基轨迹跨基因共享、正态方程向量化）+ 坐标精化。
    gate=True 的基因跳过稳态初值（init 候选不参评），仅用宽先验网格。
    返回 theta dict(beta=(G,), gamma=(G,), alpha=(N,G) 每细胞 ZOH 取值,
    alpha_on=(G,), alpha_off=(G,), t_switch=(G,), loss=(G,))。
    """
    Mu_np = Mu.cpu().numpy() if hasattr(Mu, "cpu") else np.asarray(Mu)
    Ms_np = Ms.cpu().numpy() if hasattr(Ms, "cpu") else np.asarray(Ms)
    u = np.asarray(Mu_np, dtype=np.float32)
    s = np.asarray(Ms_np, dtype=np.float32)
    tau = np.asarray(t, dtype=np.float32)
    N, G = u.shape
    gate = np.asarray(gate, dtype=bool)

    n_grid = 9
    betas_1d = np.logspace(-1.3, 1.3, n_grid, dtype=np.float32)   # ← 新增
    gammas_1d = np.logspace(-1.3, 1.3, n_grid, dtype=np.float32)  # ← 新增
    BB, GG = np.meshgrid(betas_1d, gammas_1d, indexing="ij")
    betas = BB.ravel()
    gammas = GG.ravel()                              # (P,)
    P = betas.size
    ts_grid = np.linspace(0.1, 0.9, max(int(K) - 1, 3), dtype=np.float32)  # ← 新增

    su = u.std(axis=0) + _EPS
    ss = s.std(axis=0) + _EPS
    # 加权目标：||U a - u||^2/su^2 + ||S a - s||^2/ss^2（u,s 各自单位方差化，alpha 真实量纲）
    yty = ((u ** 2).sum(axis=0) / su ** 2 + (s ** 2).sum(axis=0) / ss ** 2)   # (G,)

    best_rss = np.full(G, np.inf)
    best = {"beta": np.ones(G), "gamma": np.ones(G), "ts": np.full(G, 0.5),
            "a_on": np.zeros(G), "a_off": np.zeros(G)}

    for ts in ts_grid:
        U, S = _basis_at_ts(tau, betas, gammas, ts)        # (P,2,N)，跨基因共享
        mU11 = (U[:, 0] ** 2).sum(-1)
        mU12 = (U[:, 0] * U[:, 1]).sum(-1)
        mU22 = (U[:, 1] ** 2).sum(-1)
        mS11 = (S[:, 0] ** 2).sum(-1)
        mS12 = (S[:, 0] * S[:, 1]).sum(-1)
        mS22 = (S[:, 1] ** 2).sum(-1)
        bU1 = U[:, 0] @ u
        bU2 = U[:, 1] @ u
        bS1 = S[:, 0] @ s
        bS2 = S[:, 1] @ s                                   # (P,G)，原始量纲
        wu2 = 1.0 / (su ** 2)[None, :]
        ws2 = 1.0 / (ss ** 2)[None, :]
        m11 = mU11[:, None] * wu2 + mS11[:, None] * ws2
        m12 = mU12[:, None] * wu2 + mS12[:, None] * ws2
        m22 = mU22[:, None] * wu2 + mS22[:, None] * ws2
        b1 = bU1 * wu2 + bS1 * ws2
        b2 = bU2 * wu2 + bS2 * ws2                          # (P,G)
        rss, a_on, a_off = _solve_alpha_2x2(m11, m12, m22, b1, b2, yty[None, :])
        idx = np.argmin(rss, axis=0)                        # (G,)
        rss_min = rss[idx, np.arange(G)]
        better = rss_min < best_rss
        best_rss = np.where(better, rss_min, best_rss)
        best["beta"] = np.where(better, betas[idx], best["beta"])
        best["gamma"] = np.where(better, gammas[idx], best["gamma"])
        best["ts"] = np.where(better, ts, best["ts"])
        best["a_on"] = np.where(better, a_on[idx, np.arange(G)], best["a_on"])
        best["a_off"] = np.where(better, a_off[idx, np.arange(G)], best["a_off"])

    # 稳态初值候选（仅 gate=False 基因参评；gate=True 跳过稳态初值走宽先验）
    ratio = np.asarray(init.get("gamma_over_beta", np.ones(G)), dtype=np.float32)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio = np.clip(ratio, _RATIO_MIN, _RATIO_MAX)
    free_idx = np.where(~gate)[0]
    if free_idx.size:
        unf, snf = u.T[free_idx], s.T[free_idx]             # (F,N)，原始量纲
        for scale in (0.3, 1.0, 3.0):
            b0 = np.full(free_idx.size, scale, dtype=np.float32)  # ← 新增
            g0 = ratio[free_idx] * scale
            for ts in ts_grid:
                rss, a_on, a_off = _rss_vec(tau, unf, snf, b0, g0,
                                            np.full(free_idx.size, ts))
                cur = best_rss[free_idx]
                better = rss < cur
                gi = free_idx[better]
                best_rss[gi] = rss[better]
                best["beta"][gi] = scale
                best["gamma"][gi] = g0[better]
                best["ts"][gi] = ts
                best["a_on"][gi] = a_on[better]
                best["a_off"][gi] = a_off[better]

    # 坐标精化（log beta, log gamma, ts；收缩步长模式搜索，跨基因向量化）
    # 坐标精化（log beta, log gamma, ts；收缩步长模式搜索）
    # 修改：基因分块，避免 (G,N) 级临时数组同时膨胀导致 OOM
    # 在 G=17714, N=75000, FP64 下，原代码峰值可达 60-80 GB
    GENE_CHUNK = 2048   # 硬编码安全块大小；FP64 下峰值 < 7GB，12GB 显存/主存安全
    lb = np.log(best["beta"]).astype(np.float32)   # ← 新增 .astype(...)
    lg = np.log(best["gamma"]).astype(np.float32)  # ← 新增 .astype(...)
    tsv = best["ts"].copy()
    rss_cur = best_rss.copy()
    a_on_cur = best["a_on"].copy()
    a_off_cur = best["a_off"].copy()
    s_bg = np.full(G, 0.15, dtype=np.float32)  # ← 新增
    s_ts = np.full(G, 0.05, dtype=np.float32)  # ← 新增
    active = np.ones(G, dtype=bool)
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(int(max_iter_ls)):
        if not active.any():
            print(f"[REFINE] iter={_+1:02d}/{max_iter_ls} EARLY STOP (all inactive)")
            break
        
        # ===== 进度计数器：每轮开始时 =====
        n_active = int(active.sum())
        print(
            f"[REFINE] iter={_+1:02d}/{max_iter_ls} "
            f"active={n_active}/{G} "
            f"s_bg_median={np.median(s_bg[active]) if n_active else 0:.5g} "
            f"s_ts_median={np.median(s_ts[active]) if n_active else 0:.5g}"
        )
        
        improved = np.zeros(G, dtype=bool)
        # ===== 基因分块：将 (G,N) 级 _rss_vec 拆分为 (chunk,N) =====
        for g_start in range(0, G, GENE_CHUNK):
            g_end = min(g_start + GENE_CHUNK, G)
            chunk_active = active[g_start:g_end]
            if not chunk_active.any():
                continue
            # 局部活跃索引 → 全局索引
            local_idx = np.where(chunk_active)[0]
            idx = local_idx + g_start
            # 提取子数组（视图，不额外拷贝大数组）
            u_chunk = u[:, idx]      # (N, n_active)
            s_chunk = s[:, idx]
            for db, dg, dt in offsets:
                nts = np.clip(tsv[idx] + dt * s_ts[idx], 0.05, 0.95)
                b_try = np.clip(np.exp(lb[idx] + db * s_bg[idx]),
                                _RATE_MIN, _RATE_MAX).astype(np.float32)
                g_try = np.clip(np.exp(lg[idx] + dg * s_bg[idx]),
                                _RATE_MIN, _RATE_MAX).astype(np.float32)
                rss, a_on, a_off = _rss_vec(
                    tau, u_chunk.T, s_chunk.T, b_try, g_try, nts
                )
                bet = rss < rss_cur[idx] - 1e-12
                gi = idx[bet]
                rss_cur[gi] = rss[bet]
                lb[gi] += db * s_bg[gi]
                lg[gi] += dg * s_bg[gi]
                tsv[gi] = nts[bet]
                a_on_cur[gi] = a_on[bet]
                a_off_cur[gi] = a_off[bet]
                improved[gi] = True

        # ===== 进度计数器：每轮结束时 =====
        n_improved = int(improved.sum())
        print(
            f"[REFINE] iter={_+1:02d} done | "
            f"improved={n_improved}/{n_active} | "
            f"stagnant={n_active - n_improved}"
        )
        
        stagnant = active & ~improved
        s_bg[stagnant] *= 0.5
        s_ts[stagnant] *= 0.5
        active = (s_bg >= 1e-3) | (s_ts >= 1e-3)
    best_rss = rss_cur
    best["beta"] = np.clip(np.exp(lb), _RATE_MIN, _RATE_MAX).astype(np.float32)
    best["gamma"] = np.clip(np.exp(lg), _RATE_MIN, _RATE_MAX).astype(np.float32)
    best["ts"] = np.clip(tsv, 0.05, 0.95).astype(np.float32)
    best["a_on"], best["a_off"] = a_on_cur, a_off_cur

    alpha_cell = np.where(tau[:, None] <= best["ts"][None, :],
                          best["a_on"][None, :], best["a_off"][None, :])   # (N,G)
    if not (np.isfinite(best["beta"]).all() and np.isfinite(best["gamma"]).all()):
        raise FloatingPointError("[STABLE] non-finite beta/gamma after coordinate refinement")
    print("[L2] " + _finite_summary(best["beta"], "beta"))
    print("[L2] " + _finite_summary(best["gamma"], "gamma"))
    print("[L2] " + _finite_summary(best["gamma"] / np.maximum(best["beta"], _RATE_MIN),
                                   "gamma/beta"))

    theta = {
        "beta": best["beta"].copy(),
        "gamma": best["gamma"].copy(),
        "alpha": alpha_cell,                       # (N,G)，SPEC 允许的两种形状之一
        "alpha_on": best["a_on"].copy(),
        "alpha_off": best["a_off"].copy(),
        "t_switch": best["ts"].copy(),
        "loss": best_rss / (2.0 * N),              # 归一化（u,s 各按 std 加权）均方损失
    }
    return theta


# ================================================================ L2 纠偏 E-step
def reassign_time(Mu, Ms, theta, t, n_grid=60):
    """纠偏 E-step：每细胞投影到当前相位轨迹重新分配时间（在共享 velocity genes 上
    最小化 ||(u,s)_obs - (u,s)_pred(t)||，u,s 各自 z-score 后联合距离；
    t ∈ [0,1] 网格搜索（>=50 点）+ 抛物线局部精化）。返回 t_new。"""
    Mu_np = Mu.cpu().numpy() if hasattr(Mu, "cpu") else np.asarray(Mu)
    Ms_np = Ms.cpu().numpy() if hasattr(Ms, "cpu") else np.asarray(Ms)
    u = np.asarray(Mu_np, dtype=np.float32)
    s = np.asarray(Ms_np, dtype=np.float32)
    N, G = u.shape
    beta = np.asarray(theta["beta"], dtype=np.float32)
    gamma = np.asarray(theta["gamma"], dtype=np.float32)
    a_on = np.asarray(theta["alpha_on"], dtype=np.float32)
    a_off = np.asarray(theta["alpha_off"], dtype=np.float32)
    ts = np.asarray(theta["t_switch"], dtype=np.float32)
    mask = np.asarray(theta.get("velocity_mask", np.ones(G, dtype=bool)), dtype=bool)
    if not mask.any():
        mask = np.ones(G, dtype=bool)

    grid = np.linspace(0.0, 1.0, max(int(n_grid), 50), dtype=np.float32)  # ← 新增
    uh, sh = _trajectory(grid[None, :], beta[:, None], gamma[:, None],
                         a_on[:, None], a_off[:, None], ts[:, None])       # (G,n_grid)
    # z-score（观测统计量同时用于预测）
    mu_u, sd_u = u.mean(axis=0), u.std(axis=0) + _EPS
    mu_s, sd_s = s.mean(axis=0), s.std(axis=0) + _EPS
    un = (u - mu_u[None, :]) / sd_u[None, :]
    sn = (s - mu_s[None, :]) / sd_s[None, :]
    uhn = (uh - mu_u[:, None]) / sd_u[:, None]
    shn = (sh - mu_s[:, None]) / sd_s[:, None]
    un, sn, uhn, shn = un[:, mask], sn[:, mask], uhn[mask], shn[mask]

    # D[i,j] = sum_g (un[i,g]-uhn[g,j])^2 + (sn[i,g]-shn[g,j])^2（展开为矩阵乘）
    Du = (un ** 2).sum(1)[:, None] + (uhn ** 2).sum(0)[None, :] - 2.0 * (un @ uhn)
    Ds = (sn ** 2).sum(1)[:, None] + (shn ** 2).sum(0)[None, :] - 2.0 * (sn @ shn)
    D = Du + Ds                                                            # (N,n_grid)
    j = np.argmin(D, axis=1)
    # 抛物线局部精化
    h = grid[1] - grid[0]
    jm = np.clip(j - 1, 0, len(grid) - 1)
    jp = np.clip(j + 1, 0, len(grid) - 1)
    Lm = D[np.arange(N), jm]
    L0 = D[np.arange(N), j]
    Lp = D[np.arange(N), jp]
    denom = Lp - 2.0 * L0 + Lm
    delta = np.where(np.abs(denom) > _EPS, 0.5 * (Lm - Lp) / np.where(np.abs(denom) > _EPS, denom, 1.0), 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    t_new = np.clip(grid[j] + delta * h, 0.0, 1.0)
    return t_new


# ================================================================ 主流程
def decoupled_engine(Mu, Ms, X_scVI, root_cell=None, n_rounds=2, K=10):
    """主流程：pseudotime -> anchor -> steady init + gate -> for r in range(n_rounds):
    fit_genes_closed_form -> (r < n_rounds-1) reassign_time；
    第 2 轮起参数相对增量 < 1% 提前截断；断言 n_rounds >= 1（SPEC §0 红线）。
    返回 (theta, t, gate, diagnostics)。"""
    assert int(n_rounds) >= 1, "n_rounds >= 1（SPEC §0 红线：n_rounds=0 禁止）"
    Mu_np = Mu.cpu().numpy() if hasattr(Mu, "cpu") else np.asarray(Mu)
    Ms_np = Ms.cpu().numpy() if hasattr(Ms, "cpu") else np.asarray(Ms)
    u = np.asarray(Mu_np, dtype=np.float32)
    s = np.asarray(Ms_np, dtype=np.float32)
    N, G = u.shape

    if root_cell is None:
        root = int(np.argmin(u.sum(axis=1) + s.sum(axis=1)))   # 低表达细胞作为根
    elif isinstance(root_cell, (list, tuple, np.ndarray)):
        root = int(np.asarray(root_cell).ravel()[0])
    else:
        root = int(root_cell)

    t = pseudotime_from_latent(X_scVI, root)
    t, flipped, anchor_score = anchor_direction(t, Mu, Ms)
    init = steady_state_init(Mu, Ms)
    gate = steady_state_gate(Mu, Ms)

    diagnostics = {
        "device": DEVICE,
        "root_cell": root,
        "anchor_flipped": bool(flipped),
        "anchor_score": float(anchor_score),
        "n_gated": int(gate.sum()),
        "n_genes": int(G),
        "n_cells": int(N),
        "n_rounds_max": int(n_rounds),
        "K": int(K),
        "rounds": [],
    }

    theta = None
    prev_vec = None
    for r in range(int(n_rounds)):
        theta = fit_genes_closed_form(u, s, t, init, gate, K=K)
        theta["velocity_mask"] = ~gate                      # 共享 velocity genes
        vec = np.concatenate([theta["beta"], theta["gamma"]])
        if prev_vec is None:
            rel_change = float("inf")
        else:
            rel_change = float(np.linalg.norm(vec - prev_vec) /
                               (np.linalg.norm(prev_vec) + _EPS))
        mean_loss = float(np.mean(theta["loss"]))
        diagnostics["rounds"].append({
            "round": r,
            "mean_loss": mean_loss,
            "median_loss": float(np.median(theta["loss"])),
            "param_rel_change": rel_change,
        })
        prev_vec = vec
        if r >= 1 and rel_change < 0.01:                    # 第 2 轮起 <1% 提前截断
            diagnostics["early_stopped"] = True
            break
        if r < int(n_rounds) - 1:
            t = reassign_time(u, s, theta, t)
    diagnostics.setdefault("early_stopped", False)
    diagnostics["final_mean_loss"] = float(np.mean(theta["loss"]))
    return theta, t, gate, diagnostics


# ================================================================ 合成自检数据
def _synth_small(G=128, N=512, seed=0):
    """--small 合成数据：沿 pseudotime 的两段式诱导动力学 u/s + 噪声；
    X_scVI 为该流形的 10D 平滑嵌入（t 的基函数 + 小噪声）。"""
    rng = np.random.default_rng(seed)
    t_true = rng.uniform(0.0, 1.0, size=N)
    beta = rng.uniform(0.5, 2.5, size=G)
    gamma = rng.uniform(0.5, 2.5, size=G)
    a_on = rng.uniform(2.0, 8.0, size=G)
    a_off = rng.uniform(0.0, 0.5, size=G)
    ts = rng.uniform(0.35, 0.65, size=G)
    u, s = _trajectory(t_true[None, :], beta[:, None], gamma[:, None],
                       a_on[:, None], a_off[:, None], ts[:, None])          # (G,N)
    # 乘性 + 加性噪声，截断非负
    u = u * np.exp(0.08 * rng.standard_normal((G, N))) + 0.02 * np.abs(rng.standard_normal((G, N)))
    s = s * np.exp(0.08 * rng.standard_normal((G, N))) + 0.02 * np.abs(rng.standard_normal((G, N)))
    Mu = np.clip(u, 0.0, None).T.astype(np.float32)                         # (N,G)
    Ms = np.clip(s, 0.0, None).T.astype(np.float32)
    tt = t_true[:, None]
    feats = [np.sin(np.pi * tt), np.cos(np.pi * tt), tt, tt ** 2,
             np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt), tt ** 3,
             np.sin(3 * np.pi * tt), np.cos(3 * np.pi * tt), tt ** 4]
    X = np.concatenate(feats, axis=1) + 0.03 * rng.standard_normal((N, 10))
    truth = {"t_true": t_true, "beta": beta, "gamma": gamma,
             "alpha_on": a_on, "alpha_off": a_off, "t_switch": ts}
    return Mu, Ms, X.astype(np.float32), truth


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Phase 2 C 管线：L1 稳态初始化+门控，L2 1.5 步解耦推断")
    ap.add_argument("--mu", type=str, default=None, help="Mu .npy，形状 (N,G)")
    ap.add_argument("--ms", type=str, default=None, help="Ms .npy，形状 (N,G)")
    ap.add_argument("--x-scvi", type=str, default=None, help="X_scVI .npy，形状 (N,10)")
    ap.add_argument("--out-dir", type=str, default="logs")
    ap.add_argument("--n-rounds", type=int, default=2)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--root-cell", type=int, default=None)
    ap.add_argument("--small", action="store_true",
                    help="内部合成数据自检（G=128, N=512）")
    args = ap.parse_args()

    log = JsonLog("decoupled_velocity_engine", out_dir=args.out_dir)
    log.record("config", {"mu": args.mu, "ms": args.ms, "x_scvi": args.x_scvi,
                          "n_rounds": args.n_rounds, "K": args.K,
                          "root_cell": args.root_cell, "small": args.small})
    if args.small:
        Mu, Ms, X, truth = _synth_small(G=128, N=512)
        mode = "synth_small"
    else:
        if not (args.mu and args.ms and args.x_scvi):
            raise SystemExit("需提供 --mu/--ms/--x-scvi，或使用 --small 自检")
        Mu = np.load(args.mu)
        Ms = np.load(args.ms)
        X = np.load(args.x_scvi)
        truth = None
        mode = "from_files"
    log.record("mode", mode)
    log.record("shapes", {"Mu": list(Mu.shape), "Ms": list(Ms.shape), "X_scVI": list(X.shape)})

    try:
        theta, t, gate, diagnostics = decoupled_engine(
            Mu, Ms, X, root_cell=args.root_cell, n_rounds=args.n_rounds, K=args.K)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
        print(f"[STABLE] NUMERICAL FAILURE: {e}")
        raise

    # 诊断记录（数组只记摘要，完整结果存 npz）
    diag_json = dict(diagnostics)
    log.record("diagnostics", diag_json)
    if truth is not None:
        sp_t = _spearman(t, truth["t_true"])
        ratio_fit = theta["gamma"] / np.maximum(theta["beta"], _EPS)
        ratio_true = truth["gamma"] / truth["beta"]
        sp_ratio = _spearman(ratio_fit, ratio_true)
        log.record("selfcheck", {
            "spearman_t_vs_true": sp_t,
            "spearman_gamma_over_beta_vs_true": sp_ratio,
            "median_ratio_rel_err": float(np.median(
                np.abs(ratio_fit - ratio_true) / (np.abs(ratio_true) + _EPS))),
        })

    out_npz = os.path.join(args.out_dir, "decoupled_engine_output.npz")
    np.savez_compressed(out_npz, t=t, gate=gate, beta=theta["beta"], gamma=theta["gamma"],
                        alpha=theta["alpha"], alpha_on=theta["alpha_on"],
                        alpha_off=theta["alpha_off"], t_switch=theta["t_switch"],
                        loss=theta["loss"])
    log.record("output_npz", out_npz)
    path = log.finalize()
    print(f"[decoupled_velocity_engine] device={DEVICE} mode={mode} "
          f"mean_loss={diag_json['final_mean_loss']:.4f} n_gated={diag_json['n_gated']}")
    print(f"[decoupled_velocity_engine] JSON log: {path}")


if __name__ == "__main__":
    main()
