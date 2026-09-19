"""closed_form_kernel.py — L3 分段 FOH 闭式传播核 + Picard + Krylov expmv（SPEC §3.1）。

数学规格（与 SPEC §3.1 逐字符一致，FP32，全程 torch，可微）：
段内 h = t[k+1]-t[k]，FOH 输入 α(τ) = a + b·τ，a = alpha_knots[...,k]，
b = (alpha_knots[...,k+1]-a)/h：

    C   = a/β - b/β²
    A   = u - C
    u_new = A·e^{-βh} + (b/β)·h + C
    s_new = s·e^{-γh} + β·(A·mix + C·φ_g) + b·ramp_g
    mix    = e^{-γh}·expm1((γ-β)h)/(γ-β)   # |γ-β|·h ≥ 1e-6
           = h·e^{-βh}                     # 共振支路（极限）
    φ_g    = -expm1(-γh)/γ                 # γh ≥ 1e-8，否则级数 h - γh²/2
    ramp_g = (γh + expm1(-γh))/γ²          # γh ≥ 1e-4，否则级数 h²/2 - γh³/6 + γ²h⁴/24

β 防御性 clamp_min(1e-8)。全程 FP32。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import DEVICE

# 数值支路阈值（SPEC §3.1 冻结）
_MIX_TOL = 1e-6    # |γ-β|·h
_PHI_TOL = 1e-8    # γh
_RAMP_TOL = 1e-4   # γh
_BETA_FLOOR = 1e-8


def _as_f32(x: torch.Tensor, device) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32)


def _segment_kernels(beta: torch.Tensor, gamma: torch.Tensor, h: torch.Tensor):
    """计算 e^{-βh}, e^{-γh}, mix, φ_g, ramp_g（全部 (G,)，三分支数值稳定）。

    注：标量核函数在 float64 下求值后 cast 回 FP32——ramp_g 的一般支路
    (γh + expm1(-γh))/γ² 在 γh~1e-3 时存在灾难性相消，纯 FP32 相对误差 ~2e-4，
    无法满足 SPEC §4 测试契约（与级数值 rel err < 1e-6）。公式与分支阈值不变，
    状态更新 (u,s,A,C) 仍全程 FP32。"""
    beta = beta.to(torch.float64)
    gamma = gamma.to(torch.float64)
    h = h.to(torch.float64)
    Eb = torch.exp(-beta * h)
    Eg = torch.exp(-gamma * h)

    # ---- mix: e^{-γh}·expm1((γ-β)h)/(γ-β) 或共振极限 h·e^{-βh}
    d = gamma - beta
    x = d * h
    mask_mix = x.abs() >= _MIX_TOL
    d_safe = torch.where(mask_mix, d, torch.ones_like(d))
    mix_gen = Eg * torch.expm1(d_safe * h) / d_safe
    mix_res = h * Eb
    mix = torch.where(mask_mix, mix_gen, mix_res)

    # ---- φ_g: -expm1(-γh)/γ 或级数 h - γh²/2
    gh = gamma * h
    mask_phi = gh >= _PHI_TOL
    g_safe = torch.where(mask_phi, gamma, torch.ones_like(gamma))
    phi_gen = -torch.expm1(-g_safe * h) / g_safe
    phi_ser = h - gamma * h * h / 2.0
    phi_g = torch.where(mask_phi, phi_gen, phi_ser)

    # ---- ramp_g: (γh + expm1(-γh))/γ² 或级数 h²/2 - γh³/6 + γ²h⁴/24
    mask_ramp = gh >= _RAMP_TOL
    g_safe2 = torch.where(mask_ramp, gamma, torch.ones_like(gamma))
    ramp_gen = (g_safe2 * h + torch.expm1(-g_safe2 * h)) / g_safe2**2
    ramp_ser = h * h / 2.0 - gamma * h**3 / 6.0 + gamma**2 * h**4 / 24.0
    ramp_g = torch.where(mask_ramp, ramp_gen, ramp_ser)

    f32 = torch.float32
    return (Eb.to(f32), Eg.to(f32), mix.to(f32), phi_g.to(f32), ramp_g.to(f32))


def propagate_closed_form(u0, s0, alpha_knots, t_knots, beta, gamma):
    """u0,s0: (N,G) FP32; alpha_knots: (N,G,K+1) 或 (G,K+1)（广播）; t_knots: (K+1,) 升序;
    beta,gamma: (G,) 正。返回 (u, s) 各 (N,G)，为 t_knots[-1] 时刻状态。可微。"""
    device = u0.device
    u = _as_f32(u0, device)
    s = _as_f32(s0, device)
    beta = _as_f32(beta, device).clamp_min(_BETA_FLOOR)
    gamma = _as_f32(gamma, device)
    t = _as_f32(t_knots, device).reshape(-1)
    alpha = _as_f32(alpha_knots, device)

    assert t.numel() >= 2, "t_knots 至少需要 2 个节点"
    assert bool((t[1:] > t[:-1]).all()), "t_knots 必须严格升序"
    K = t.numel() - 1
    assert alpha.shape[-1] == K + 1, (
        f"alpha_knots 末维须为 K+1={K+1}，得到 {alpha.shape[-1]}")
    assert alpha.dim() in (2, 3), "alpha_knots 须为 (G,K+1) 或 (N,G,K+1)"
    if alpha.dim() == 3:
        assert alpha.shape[0] == u.shape[0] and alpha.shape[1] == u.shape[1]

    for k in range(K):
        h = t[k + 1] - t[k]
        a = alpha[..., k]
        b = (alpha[..., k + 1] - a) / h
        Eb, Eg, mix, phi_g, ramp_g = _segment_kernels(beta, gamma, h)  # (G,) 预计算复用

        C = a / beta - b / beta**2
        A = u - C
        u = A * Eb + (b / beta) * h + C
        s = s * Eg + beta * (A * mix + C * phi_g) + b * ramp_g
    return u, s


def alpha_nodes(W_csr, s_knots, bias):
    """W_csr: (G,G) CSR; s_knots: (N,G,K+1); bias: (G,)。
    返回 alpha_knots (N,G,K+1) = clamp(softplus(W@s_k + b), 0, 50)。内部用 torch.sparse.mm。"""
    assert W_csr.layout == torch.sparse_csr, "W_csr 必须为 CSR 稀疏张量"
    s = _as_f32(s_knots, W_csr.device)
    bias = _as_f32(bias, W_csr.device)
    N, G, K1 = s.shape
    # z[i, (n,k)] = sum_j W[i,j] * s[n,j,k]（与 common._alpha_np 的 s @ W.T 约定一致）
    s_flat = s.permute(1, 0, 2).reshape(G, N * K1)
    z = torch.sparse.mm(W_csr, s_flat)  # (G, N*K1)
    alpha = torch.clamp(F.softplus(z + bias.unsqueeze(1)), 0.0, 50.0)
    return alpha.reshape(G, N, K1).permute(1, 0, 2).contiguous()


def picard_propagate(u0, s0, W_csr, bias, beta, gamma, t_knots, max_iter=3, tol=1e-3):
    """冻结-外生化 Picard 迭代：初值 s_knots 由 s0 线性外插；每轮 alpha_nodes -> propagate_closed_form
    并在节点处采样新 s_knots；residual = 节点 s 的最大相对变化。返回 (u, s, residual, converged)。"""
    assert max_iter >= 1, "max_iter >= 1"
    device = u0.device
    u0 = _as_f32(u0, device)
    s0 = _as_f32(s0, device)
    beta = _as_f32(beta, device)
    gamma = _as_f32(gamma, device)
    t = _as_f32(t_knots, device).reshape(-1)
    assert t.numel() >= 2 and bool((t[1:] > t[:-1]).all()), "t_knots 必须严格升序"
    K = t.numel() - 1

    # 初值：s0 沿 t0 时刻导数 (βu0 - γs0) 线性外插
    ds0 = beta * u0 - gamma * s0
    s_knots = s0.unsqueeze(-1) + ds0.unsqueeze(-1) * t.reshape(1, 1, K + 1)

    u, s = u0, s0
    residual = float("inf")
    converged = False
    for _ in range(max_iter):
        alpha_knots = alpha_nodes(W_csr, s_knots, bias)
        u_k, s_k = u0, s0
        s_new = [s0]
        for k in range(K):
            u_k, s_k = propagate_closed_form(
                u_k, s_k, alpha_knots[..., k:k + 2], t[k:k + 2], beta, gamma)
            s_new.append(s_k)
        s_new_knots = torch.stack(s_new, dim=-1)  # (N,G,K+1)
        num = (s_new_knots - s_knots).norm(dim=(0, 1))          # (K+1,)
        den = s_knots.norm(dim=(0, 1)).clamp_min(1e-8)
        residual = float((num / den).max())
        s_knots = s_new_knots
        u, s = u_k, s_k
        if residual < tol:
            converged = True
            break
    return u, s, residual, converged


def segment_knots(t_max, K, strategy="uniform", alpha_breaks=None):
    """段边界。uniform：等分；"adaptive"：在 alpha_breaks（softplus/clamp 折点时刻）附近加密。
    返回 (K+1,) tensor（严格升序，端点 0 与 t_max）。"""
    assert K >= 1, "K >= 1"
    t_max = float(t_max)
    if strategy == "uniform" or alpha_breaks is None or len(alpha_breaks) == 0:
        return torch.linspace(0.0, t_max, K + 1, dtype=torch.float32, device=DEVICE)
    assert strategy == "adaptive", f"未知段策略: {strategy}"

    breaks = np.asarray(sorted(set(float(b) for b in alpha_breaks)), dtype=np.float64)
    breaks = breaks[(breaks > 0.0) & (breaks < t_max)]
    if breaks.size == 0:
        return torch.linspace(0.0, t_max, K + 1, dtype=torch.float32, device=DEVICE)
    # 折点过多时均匀抽稀，为全局网格留出配额
    max_breaks = max(1, K // 2)
    if breaks.size > max_breaks:
        idx = np.linspace(0, breaks.size - 1, max_breaks).round().astype(int)
        breaks = breaks[np.unique(idx)]

    # 密度 ρ(t) = 1 + Σ_b 4·exp(-((t-b)/σ)²)，σ 取折点最小间距的一半（下限 t_max/200）
    if breaks.size > 1:
        sigma = max(0.5 * np.diff(breaks).min(), t_max / 200.0)
    else:
        sigma = t_max / 50.0
    grid = np.linspace(0.0, t_max, 4001)
    rho = np.ones_like(grid)
    for bpt in breaks:
        rho += 4.0 * np.exp(-((grid - bpt) / sigma) ** 2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    knots = np.interp(np.linspace(0.0, 1.0, K + 1), cdf, grid)
    knots[0], knots[-1] = 0.0, t_max
    # 严格升序防御
    min_step = t_max * 1e-7
    for i in range(1, K + 1):
        if knots[i] <= knots[i - 1] + min_step:
            knots[i] = knots[i - 1] + min_step
    knots = knots * (t_max / knots[-1])                  # 重新归一到 [0, t_max]
    knots[0], knots[-1] = 0.0, t_max
    return torch.from_numpy(knots.astype(np.float32)).to(DEVICE)


def arnoldi_expmv(A_csr, v, t, m=30):
    """Krylov 备用引擎：w ≈ exp(t·A)·v。A_csr (G,G) CSR（可为增广算子），v (G,) 或 (G,r)。
    Arnoldi 构造 V,H；scipy.linalg.expm 小矩阵；返回 (w, residual_est)。
    residual_est = |h_{m+1,m}·e_mᵀ·exp(tH)·e1|·‖v‖（相对；v 归一化时即相对残余）。
    多列输入时 residual_est 为各列最大值。"""
    from scipy.linalg import expm as scipy_expm

    assert A_csr.layout == torch.sparse_csr, "A_csr 必须为 CSR 稀疏张量"
    device = A_csr.device
    single = v.dim() == 1
    V_in = v.detach().to(device="cpu", dtype=torch.float64)
    if single:
        V_in = V_in.unsqueeze(1)
    G, r = V_in.shape
    V_np = V_in.numpy()

    def matvec(x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x.astype(np.float32)).to(device).unsqueeze(1)
        wt = torch.sparse.mm(A_csr, xt)
        return wt.squeeze(1).to("cpu", torch.float64).numpy()

    W_out = np.zeros((G, r), dtype=np.float64)
    resids = np.zeros(r, dtype=np.float64)
    for j in range(r):
        x = V_np[:, j]
        nrm = np.linalg.norm(x)
        if nrm == 0.0:
            continue
        Q = np.zeros((G, m + 1))
        H = np.zeros((m + 1, m))
        Q[:, 0] = x / nrm
        m_eff = m
        happy = False
        for i in range(m):
            w = matvec(Q[:, i])
            for l in range(i + 1):  # 经典 Gram-Schmidt
                H[l, i] = Q[:, l] @ w
                w = w - H[l, i] * Q[:, l]
            for l in range(i + 1):  # 一次再正交化
                c = Q[:, l] @ w
                H[l, i] += c
                w = w - c * Q[:, l]
            H[i + 1, i] = np.linalg.norm(w)
            if H[i + 1, i] < 1e-14:
                m_eff = i + 1
                happy = True
                break
            Q[:, i + 1] = w / H[i + 1, i]
        Hm = H[:m_eff, :m_eff]
        Qm = Q[:, :m_eff]
        E = scipy_expm(t * Hm)
        W_out[:, j] = nrm * (Qm @ E[:, 0])
        if happy:
            resids[j] = 0.0
        else:
            # h_{m+1,m} = H[m_eff, m_eff-1]; e_mᵀ·exp(tH)·e1 = E[m_eff-1, 0]
            resids[j] = abs(H[m_eff, m_eff - 1] * E[m_eff - 1, 0]) * nrm

    w_t = torch.from_numpy(W_out).to(device=device, dtype=torch.float32)
    if single:
        return w_t.squeeze(1), float(resids[0])
    return w_t, float(resids.max())


def _scipy_csr_to_torch(M) -> torch.Tensor:
    """scipy CSR -> torch CSR（FP32, DEVICE）。仅用于装配增广算子，无 dense (G,G)。"""
    M = M.tocsr()
    crow = torch.from_numpy(M.indptr.astype(np.int64))
    col = torch.from_numpy(M.indices.astype(np.int64))
    val = torch.from_numpy(M.data.astype(np.float32))
    return torch.sparse_csr_tensor(crow, col, val, size=M.shape, device=DEVICE)


def coupled_backup_engine(u0, s0, W_csr, bias, beta, gamma, t_eval, m=30):
    """备用路径：对增广线性化算子 L = [[-βI, D⊙W],[βI, -γI]]（D=diag(softplus'(z₀))，z₀=W s₀+b）
    用 arnoldi_expmv 逐细胞传播。返回 (u, s) at t_eval[-1]。"""
    import scipy.sparse as sp

    device = W_csr.device
    u0 = _as_f32(u0, device)
    s0 = _as_f32(s0, device)
    beta = _as_f32(beta, device).clamp_min(_BETA_FLOOR)
    gamma = _as_f32(gamma, device)
    bias = _as_f32(bias, device)
    N, G = u0.shape
    t_eval = np.asarray(t_eval, dtype=np.float64).reshape(-1)
    assert t_eval.size >= 2, "t_eval 至少需要 2 个时刻"
    t_span = float(t_eval[-1] - t_eval[0])

    # z0 = W s0 + b（torch.sparse.mm，无 dense matmul）
    z0 = torch.sparse.mm(W_csr, s0.t()).t() + bias      # (N,G)
    D = torch.sigmoid(z0)                                # softplus'(z0) = σ(z0)

    # W -> scipy CSR（仅稀疏装配）
    W_sp = sp.csr_matrix(
        (W_csr.values().cpu().numpy(),
         W_csr.col_indices().cpu().numpy(),
         W_csr.crow_indices().cpu().numpy()), shape=(G, G))
    beta_np = beta.cpu().numpy()
    gamma_np = gamma.cpu().numpy()
    D_np = D.cpu().numpy()
    u0_np = u0.cpu().numpy()
    s0_np = s0.cpu().numpy()

    B = sp.diags(beta_np, format="csr")
    Gm = sp.diags(-gamma_np, format="csr")
    mB = sp.diags(-beta_np, format="csr")

    u_out = np.zeros((N, G), dtype=np.float32)
    s_out = np.zeros((N, G), dtype=np.float32)
    for n in range(N):
        DW = sp.diags(D_np[n], format="csr") @ W_sp      # D⊙W（行缩放，保持稀疏）
        L = sp.bmat([[mB, DW], [B, Gm]], format="csr")   # (2G, 2G)
        L_torch = _scipy_csr_to_torch(L)
        v0 = torch.from_numpy(
            np.concatenate([u0_np[n], s0_np[n]]).astype(np.float32)).to(device)
        w, _ = arnoldi_expmv(L_torch, v0, t_span, m=m)
        w_np = w.cpu().numpy()
        u_out[n] = w_np[:G]
        s_out[n] = w_np[G:]
    return (torch.from_numpy(u_out).to(device),
            torch.from_numpy(s_out).to(device))
