"""sparse_regvelo_engine.py — Phase 3 整合主入口（SPEC §3.6，签名冻结）。

流程：L1 稳态初值+门控 -> L2 1.5 步解耦（decoupled_engine）-> L3 分段闭式传播
（picard_propagate，cell mini-batch 循环），Picard 残差超阈时按 select_engine
硬编码规则自动降级（krylov_expmv / full_sparse_expmv / full_dynamics_path）。

红线（SPEC §0）：
- 状态更新/导数计算全程 FP32；仅原始 Ms/Mu 允许 BF16 存储（本入口读入后即转 FP32）。
- 下游 velocity 直算 v = beta*u - gamma*s，禁止重解 ODE。
- Jacobian 必须解析（analytic_jacobian），禁止数值差分。
- n_rounds >= 1 入口断言。
- GRN 一律 CSR，无 dense (G,G) 可学习矩阵。

依赖说明：closed_form_kernel.py 与 decoupled_velocity_engine.py 由其他代理并行
实现（SPEC §3.1/§3.5 冻结签名）。本模块 import 失败时降级：analytic_jacobian /
select_engine 等纯函数仍可用，infer 抛出明确错误。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    DEVICE,
    JsonLog,
    get_device,
    make_csr,
    peak_memory_gb,
    reset_peak_memory,
    synth_grn,
    synth_params,
)

# ---------------------------------------------------------------- 依赖（并行开发，允许缺失降级）
try:
    from closed_form_kernel import (
        arnoldi_expmv,
        coupled_backup_engine,
        picard_propagate,
        segment_knots,
    )
    from decoupled_velocity_engine import (
        decoupled_engine,
        steady_state_gate,
        steady_state_init,
    )

    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as _e:  # 依赖未合并时本地自检只覆盖纯函数
    _DEPS_OK = False
    _DEPS_ERR = str(_e)

# ---------------------------------------------------------------- 常量（SPEC §3.6 / 引导语 §5）
ALPHA_CLAMP_MAX = 50.0          # softplus clamp 上界（clamp 边界外导数置 0）
PICARD_TOL = 1e-3               # Picard residual 阈值
KRYLOV_TOL = 0.1                # Krylov residual 阈值
KRYLOV_M = 30                   # Krylov 子空间维数（探测与备用路径）
GRAPH_REG_LAMBDA_PLACEHOLDER = 0.0  # hard=False 时的正则项接口占位


# ================================================================ 解析 Jacobian
def _csr_row_indices(W_csr: torch.Tensor) -> torch.Tensor:
    """由 crow_indices 展开每个非零元所属的行号 (nnz,)。"""
    crow = W_csr.crow_indices()
    nnz_per_row = crow[1:] - crow[:-1]
    return torch.repeat_interleave(
        torch.arange(W_csr.shape[0], device=W_csr.device), nnz_per_row
    )


def analytic_jacobian(W_csr: torch.Tensor, s_batch: torch.Tensor,
                      beta: torch.Tensor, gamma: torch.Tensor,
                      bias: torch.Tensor) -> dict:
    """动态 Jacobian（解析，仅 GRN 边上的非零元）：

        J_ij = softplus'(z_i) * W_ij - gamma_i * delta_ij,   z = W s + b

    准稳态闭合 u_i ≈ alpha_i(s)/beta_i 下 v = beta*u - gamma*s 的解析导数
    （beta_i 与 1/beta_i 相消，故 J 中只出现 softplus'(z_i)*W_ij 与 -gamma_i）。
    softplus'(z) = sigmoid(z)；clamp 区间 [0, 50] 外（alpha 触 0 或 50）导数置 0。

    W_csr: (G,G) CSR，动力学约定 z_i = sum_j W_ij s_j（与 alpha_nodes /
    simulate_reference 一致）。s_batch: (N,G) FP32，取 batch 均值为代表点
    （逐细胞代表可由调用方分批调用实现）。beta/gamma/bias: (G,)。

    返回 dict(J=与 W_csr 同稀疏模式的 CSR, diag=gamma, z=z)。
    对角元 -gamma_i 不写入 J 以保留稀疏模式，单独以 diag 返回。禁止数值差分。
    """
    assert W_csr.layout == torch.sparse_csr, "W_csr 必须为 torch.sparse_csr_tensor"
    device = W_csr.device
    G = W_csr.shape[0]
    s_batch = s_batch.to(device=device, dtype=torch.float32)
    gamma = gamma.to(device=device, dtype=torch.float32)
    bias = bias.to(device=device, dtype=torch.float32)

    # 代表点：batch 均值（FP32）
    s_rep = s_batch.mean(dim=0)                                     # (G,)
    z = torch.sparse.mm(W_csr, s_rep.unsqueeze(1)).squeeze(1) + bias  # (G,)
    alpha = torch.clamp(F.softplus(z), 0.0, ALPHA_CLAMP_MAX)
    sig = torch.sigmoid(z)
    # clamp 边界外（alpha 触 0 或 50）导数置 0
    active = (alpha > 0.0) & (alpha < ALPHA_CLAMP_MAX)
    d = torch.where(active, sig, torch.zeros_like(sig))             # (G,)

    # J 的非零模式 = W_csr 模式：CSR 逐 value 计算 d_i * W_ij（i = 行号）
    row_idx = _csr_row_indices(W_csr)
    j_vals = W_csr.values().to(torch.float32) * d[row_idx]
    J = torch.sparse_csr_tensor(
        W_csr.crow_indices(), W_csr.col_indices(), j_vals, size=(G, G)
    )
    return {"J": J, "diag": gamma, "z": z}


# ================================================================ 自动降级规则
def select_engine(gene_subset_gate, picard_residual: float,
                  krylov_residual: float) -> str:
    """自动降级规则（引导语 §5 硬编码）：

    gate 违反 -> "full_dynamics_path"；
    picard_residual > 1e-3 且 krylov_residual < 0.1 -> "krylov_expmv"；
    krylov_residual >= 0.1 -> "full_sparse_expmv"；否则 -> "closed_form_foh"。
    """
    if isinstance(gene_subset_gate, (bool, np.bool_)):
        gate_hit = bool(gene_subset_gate)
    else:
        gate_hit = bool(torch.as_tensor(gene_subset_gate).any().item())
    if gate_hit:
        return "full_dynamics_path"
    if picard_residual > PICARD_TOL:
        return "krylov_expmv" if krylov_residual < KRYLOV_TOL else "full_sparse_expmv"
    return "closed_form_foh"


# ================================================================ 内部工具
def _infer_bias(W_csr: torch.Tensor, s_ref: torch.Tensor,
                alpha_target: torch.Tensor) -> torch.Tensor:
    """由拟合的 alpha 水平反推 per-gene bias（闭式 softplus 逆）：

    选 b_g 使 clamp(softplus(z_g + b_g)) 在参考状态 s_ref 处复现 alpha_target：
        b_g = softplus^{-1}(clip(alpha_target, eps, 50-eps)) - z_g,  z = W s_ref
    softplus^{-1}(y) = log(expm1(y))。这是 L2 theta 不带 bias 时的闭式补齐。
    """
    eps = 1e-4
    z = torch.sparse.mm(W_csr, s_ref.unsqueeze(1)).squeeze(1)       # (G,)
    a = torch.clamp(alpha_target.to(torch.float32), eps, ALPHA_CLAMP_MAX - eps)
    return torch.log(torch.expm1(a)) - z


def _krylov_residual_probe(W_csr: torch.Tensor, s_batch: torch.Tensor,
                           gamma: torch.Tensor, bias: torch.Tensor,
                           t_max: float, m: int = KRYLOV_M) -> float:
    """Krylov 充分性探测：对线性化耦合块 D∘W（D = diag(softplus'(z))，
    z 取 batch 均值代表点）用 arnoldi_expmv 估计相对残差。

    对角位移 -γI 不构成耦合、对 Krylov 收敛性影响小，故探测算子取 D∘W；
    这是 infer 路由决策的启发式估计，完整增广算子的实际传播由
    coupled_backup_engine 内部完成。
    """
    G = W_csr.shape[0]
    s_rep = s_batch.mean(dim=0).to(torch.float32)
    z = torch.sparse.mm(W_csr, s_rep.unsqueeze(1)).squeeze(1) + bias
    alpha = torch.clamp(F.softplus(z), 0.0, ALPHA_CLAMP_MAX)
    active = (alpha > 0.0) & (alpha < ALPHA_CLAMP_MAX)
    d = torch.where(active, torch.sigmoid(z), torch.zeros_like(z))
    row_idx = _csr_row_indices(W_csr)
    a_vals = W_csr.values().to(torch.float32) * d[row_idx]
    A = torch.sparse_csr_tensor(W_csr.crow_indices(), W_csr.col_indices(),
                                a_vals, size=(G, G))
    g = torch.Generator(device="cpu").manual_seed(0)
    v_probe = torch.randn(G, generator=g).to(W_csr.device)
    _, residual_est = arnoldi_expmv(A, v_probe, float(t_max), m=m)
    return float(residual_est)


# ================================================================ 主入口 infer
def infer(Ms, Mu, W_csr, X_scVI, root_cell=None, n_rounds: int = 2, K: int = 10,
          batch_size: int = 2048, hard: bool = True, out_dir: str = "logs",
          picard_max_iter: int = 10,
          l2_checkpoint_dir: str = None,
          l2_only: bool = False,
          rate_max: float  | None = None) -> dict:
    """L1 稳态初值+门控 -> L2 1.5 步解耦 -> L3 分段闭式传播（cell mini-batch）。

    内存优化：L1/L2（pseudotime / 门控 / 闭式拟合）全程在 CPU 执行；
    Ms/Mu/W 仅在 L3（Picard 传播）前转入 GPU，避免 DPT 阶段 GPU 空转占显存。
    """
    assert n_rounds >= 1, "n_rounds=0 被禁止（SPEC §0 红线）"
    if not _DEPS_OK:
        raise RuntimeError(
            "infer 依赖 closed_form_kernel / decoupled_velocity_engine，"
            f"import 失败：{_DEPS_ERR}"
        )
    device = get_device()
    log = JsonLog("sparse_regvelo_engine", out_dir)
    reset_peak_memory()

    # ---------------- 输入规范化：先保留 CPU numpy 版本给 L1/L2
    Ms_cpu = np.asarray(Ms, dtype=np.float32)
    Mu_cpu = np.asarray(Mu, dtype=np.float32)
    X_scVI_cpu = np.asarray(X_scVI, dtype=np.float32) if isinstance(X_scVI, np.ndarray) else np.asarray(X_scVI)
    N, G = Ms_cpu.shape
    assert Mu_cpu.shape == (N, G), "Mu 形状与 Ms 不一致"
    log.record("n_cells", N)
    log.record("n_genes", G)
    log.record("n_rounds", n_rounds)
    log.record("K", K)
    log.record("batch_size", batch_size)
    log.record("picard_max_iter", picard_max_iter)
    log.record("mode", "hard" if hard else "soft")
    if not hard:
        log.record("graph_reg_lambda", GRAPH_REG_LAMBDA_PLACEHOLDER)

    # ---------------- 自适应速率边界（两轮 L2：探测轮 + 精化轮）
    import decoupled_velocity_engine as _dve
    beta_p995 = None      # 探测分位数，供 checkpoint 记录
    gamma_p995 = None
    # 如果用户显式指定了 rate_max（非默认值），尊重用户选择
    if rate_max is not None:
        _dve._RATE_MAX = float(rate_max)
        _dve._RATIO_MAX = float(rate_max) / _dve._RATE_MIN
        log.record("rate_max", float(rate_max))
        log.record("rate_max_mode", "manual")
        print(f"[CONFIG] 手动指定速率边界: rate_max={rate_max}")
    else:
        # 探测轮：超宽松边界，收集真实分布
        _dve._RATE_MAX = 1000.0
        _dve._RATIO_MAX = 1000.0 / _dve._RATE_MIN
        print("[ADAPTIVE] 探测轮开始: rate_max=1000.0 (n_rounds=1)")
        theta_probe, _, _, _ = decoupled_engine(
            Mu_cpu, Ms_cpu, X_scVI_cpu, root_cell=root_cell, n_rounds=1, K=K
        )
        beta_probe = np.asarray(theta_probe["beta"], dtype=np.float32)
        gamma_probe = np.asarray(theta_probe["gamma"], dtype=np.float32)
        # 计算 p99.5 分位数
        beta_p995 = np.percentile(beta_probe, 99.5)
        gamma_p995 = np.percentile(gamma_probe, 99.5)
        # 自适应边界 = max(20, p99.5 * 1.2)，保留尾部余量
        adaptive_max = max(20.0, max(beta_p995, gamma_p995) * 1.2)
        # 上限保护：不超过 500（极端异常值过滤）
        adaptive_max = min(adaptive_max, 500.0)
        log.record("rate_max", float(adaptive_max))
        log.record("rate_max_mode", "adaptive")
        log.record("beta_p995_probe", float(beta_p995))
        log.record("gamma_p995_probe", float(gamma_p995))
        print(f"[ADAPTIVE] 探测完成: beta_p99.5={beta_p995:.2f}, gamma_p99.5={gamma_p995:.2f}")
        print(f"[ADAPTIVE] 自适应边界: rate_max={adaptive_max:.2f} (p99.5*1.2, 下限20, 上限500)")
        # 设置真正 L2 的边界
        _dve._RATE_MAX = float(adaptive_max)
        _dve._RATIO_MAX = float(adaptive_max) / _dve._RATE_MIN
        # 释放探测轮内存
        del theta_probe, beta_probe, gamma_probe
        import gc; gc.collect()

    # ---------------- L1：稳态初值 + 门控（CPU）
    init = steady_state_init(Mu_cpu, Ms_cpu)
    gate_l1 = steady_state_gate(Mu_cpu, Ms_cpu)

    # ---------------- L2：1.5 步解耦（CPU）
    theta, t, gate, diagnostics = decoupled_engine(
        Mu_cpu, Ms_cpu, X_scVI_cpu, root_cell=root_cell, n_rounds=n_rounds, K=K
    )
    t_cpu = np.asarray(t, dtype=np.float32)
    gate_cpu = np.asarray(gate, dtype=bool)
    beta_cpu = np.asarray(theta["beta"], dtype=np.float32)
    gamma_cpu = np.asarray(theta["gamma"], dtype=np.float32)
    alpha_fit_cpu = np.asarray(theta["alpha"], dtype=np.float32) if theta.get("alpha") is not None else None

    n_gated = int(gate_cpu.sum())
    gated_genes_route = "full_dynamics_path" if n_gated > 0 else None
    log.record("gate_violations", n_gated)
    log.record("n_gated", n_gated)
    log.record("gated_genes_route", gated_genes_route)

    # ---------------- L2 完成：释放 CPU 大数组（可选，让 DPT 内存尽快回收）
    del X_scVI_cpu
    import gc; gc.collect()

    # ---------------- L2 补齐 bias（CPU 计算即可）
    alpha_gene_cpu = alpha_fit_cpu.mean(axis=0) if (alpha_fit_cpu is not None and alpha_fit_cpu.ndim == 2) else alpha_fit_cpu
    if alpha_gene_cpu is None:
        alpha_gene_cpu = np.ones(G, dtype=np.float32)
    # _infer_bias 需要 torch，但可以在 CPU 算
    bias_cpu = _infer_bias_cpu(W_csr, Ms_cpu.mean(axis=0), alpha_gene_cpu)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ---------------- L2 检查点保存（解耦 L3 前置步骤）
    if l2_checkpoint_dir is not None:
        _save_l2_checkpoint(
            checkpoint_dir=l2_checkpoint_dir,
            beta_cpu=beta_cpu,
            gamma_cpu=gamma_cpu,
            t_cpu=t_cpu,
            gate_cpu=gate_cpu,
            bias_cpu=bias_cpu,
            alpha_fit_cpu=alpha_fit_cpu,
            W=W_csr,                       # 当前 W（torch CSR / scipy CSR / numpy）
            meta=dict(
                N=N, G=G,
                n_rounds=n_rounds, K=K,
                batch_size=batch_size,
                hard=hard,
                picard_max_iter=picard_max_iter,
                rate_max_requested=(None if rate_max is None else float(rate_max)),  # 用户传入值；None=未指定(adaptive)    
                rate_max=float(_dve._RATE_MAX),            # 实际生效值（保留键名兼容旧读取）
                rate_max_actual=float(_dve._RATE_MAX),
                rate_max_mode=("manual" if rate_max is not None else "adaptive"),
                beta_p995=(None if beta_p995 is None else float(beta_p995)),
                gamma_p995=(None if gamma_p995 is None else float(gamma_p995)),
            ),
        )
        log.record("l2_checkpoint_dir", l2_checkpoint_dir)
        if l2_only:
            log_path = log.finalize({
                "init_gamma_over_beta_mean": float(
                    np.asarray(init["gamma_over_beta"]).astype(np.float32).mean().item()
                ),
                "l1_gate_violations": int(np.asarray(gate_l1).sum()),
            })
            return {
                "l2_complete": True,
                "checkpoint_dir": l2_checkpoint_dir,
                "beta": beta_cpu,
                "gamma": gamma_cpu,
                "t": t_cpu,
                "gate": gate_cpu,
                "bias": bias_cpu,
                "logs": log_path,
            }
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ---------------- 转入 GPU：仅 L3 需要的数据
    Ms = torch.from_numpy(Ms_cpu).to(device=device, dtype=torch.float32)
    Mu = torch.from_numpy(Mu_cpu).to(device=device, dtype=torch.float32)
    t = torch.from_numpy(t_cpu).to(device=device, dtype=torch.float32)
    gate = torch.from_numpy(gate_cpu).to(device=device).bool()
    beta = torch.from_numpy(beta_cpu).to(device=device, dtype=torch.float32)
    gamma = torch.from_numpy(gamma_cpu).to(device=device, dtype=torch.float32)
    alpha_fit = torch.from_numpy(alpha_fit_cpu).to(device=device, dtype=torch.float32) if alpha_fit_cpu is not None else None
    bias = torch.from_numpy(bias_cpu).to(device=device, dtype=torch.float32)
    W_csr = make_csr(W_csr, device=device)

    # ---------------- L3：分段闭式传播（cell mini-batch 循环，GPU）
    t_max = float(t.max().item())
    if t_max <= 0.0:
        t_max = 1.0
    t_knots = segment_knots(t_max, K)
    u_end = torch.empty_like(Mu)
    s_end = torch.empty_like(Ms)
    picard_residuals = []
    picard_residuals_eff = []
    route_counts = {"closed_form_foh": 0, "krylov_expmv": 0,
                    "full_sparse_expmv": 0, "full_dynamics_path": 0}
    krylov_residual_global = 0.0
    n_unconverged = 0

    for start in range(0, N, batch_size):
        ub = Mu[start:start + batch_size].contiguous()
        sb = Ms[start:start + batch_size].contiguous()
        u_b, s_b, res, converged = picard_propagate(
            ub, sb, W_csr, bias, beta, gamma, t_knots,
            max_iter=picard_max_iter, tol=PICARD_TOL
        )
        res = float(res)
        converged = bool(converged)
        picard_residuals.append(res)
        if not converged:
            n_unconverged += 1
        res_eff = res if converged else max(res, 2.0 * PICARD_TOL)
        picard_residuals_eff.append(res_eff)
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
        u_end[start:start + u_b.shape[0]] = u_b
        s_end[start:start + s_b.shape[0]] = s_b

    max_picard_res = max(picard_residuals) if picard_residuals else 0.0
    max_picard_res_eff = max(picard_residuals_eff) if picard_residuals_eff else 0.0
    engine_route = select_engine(False, max_picard_res_eff,
                                 krylov_residual_global)
    log.record("picard_residual_max", max_picard_res)
    log.record("krylov_residual_probe", krylov_residual_global)
    log.record("n_unconverged_batches", n_unconverged)
    log.record("engine_route", engine_route)
    log.record("route_counts", route_counts)

    # ---------------- 下游 velocity：直算 v = beta*u - gamma*s
    v = beta.unsqueeze(0) * u_end - gamma.unsqueeze(0) * s_end

    # ---------------- alpha 输出
    if alpha_fit is not None:
        alpha_out = alpha_fit
    else:
        z = torch.sparse.mm(W_csr, s_end.mean(dim=0).unsqueeze(1)).squeeze(1) + bias
        alpha_out = torch.clamp(F.softplus(z), 0.0, ALPHA_CLAMP_MAX)

    # ---------------- 解析 Jacobian
    jac = analytic_jacobian(W_csr, s_end, beta, gamma, bias)
    J, J_diag = jac["J"], jac["diag"]

    # ---------------- 显存红线（仅 CUDA）
    if device == "cuda":
        assert peak_memory_gb() < 8.0, (
            f"显存红线违反：peak={peak_memory_gb():.2f} GB >= 8 GB"
        )

    log_path = log.finalize({
        "init_gamma_over_beta_mean": float(
            np.asarray(init["gamma_over_beta"]).astype(np.float32).mean().item()
        ),
        "l1_gate_violations": int(np.asarray(gate_l1).sum()),
    })

    return {
        "t": t,
        "v": v,
        "alpha": alpha_out,
        "beta": beta,
        "gamma": gamma,
        "J": J,
        "J_diag": J_diag,
        "gate": gate,
        "engine_route": engine_route,
        "u": u_end,
        "s": s_end,
        "diagnostics": diagnostics,
        "logs": log_path,
    }


# ---------------- 新增：CPU 版 _infer_bias（避免提前创建设备 tensor）
def _infer_bias_cpu(W, s_ref_np: np.ndarray, alpha_target_np: np.ndarray) -> np.ndarray:
    """CPU numpy 版 _infer_bias，避免 L2 阶段占用 GPU 显存。
    全程保持 sparse，禁止任何 to_dense() / toarray() 操作。
    """
    import scipy.sparse
    eps = 1e-4
    s_ref = np.asarray(s_ref_np, dtype=np.float32)
    a = np.clip(np.asarray(alpha_target_np, dtype=np.float32), eps, ALPHA_CLAMP_MAX - eps)

    if isinstance(W, torch.Tensor) and W.layout == torch.sparse_csr:
        # torch sparse CSR → CPU 计算 z = W @ s_ref
        s_t = torch.from_numpy(s_ref).to(dtype=torch.float32)
        z = torch.sparse.mm(W.cpu(), s_t.unsqueeze(1)).squeeze(1).numpy()
    elif isinstance(W, scipy.sparse.spmatrix):
        # scipy sparse CSR/CSC/COO → 直接用 dot（保持 sparse）
        z = W.dot(s_ref)
    elif isinstance(W, np.ndarray):
        # numpy dense（fallback，理论上不应出现）
        z = s_ref @ W.T
    else:
        raise TypeError(f"Unsupported W type in _infer_bias_cpu: {type(W)}")

    return np.log(np.expm1(a)) - z
# ---------------- 新增：L2 检查点保存（解耦 L3）
def _save_l2_checkpoint(checkpoint_dir: str,
                        beta_cpu: np.ndarray,
                        gamma_cpu: np.ndarray,
                        t_cpu: np.ndarray,
                        gate_cpu: np.ndarray,
                        bias_cpu: np.ndarray,
                        alpha_fit_cpu: np.ndarray | None,
                        W,
                        meta: dict) -> None:
    """将 L2 推断参数统一落盘，供独立 L3 流式入口加载。"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    np.save(os.path.join(checkpoint_dir, "beta.npy"), beta_cpu)
    np.save(os.path.join(checkpoint_dir, "gamma.npy"), gamma_cpu)
    np.save(os.path.join(checkpoint_dir, "t.npy"), t_cpu)
    np.save(os.path.join(checkpoint_dir, "gate.npy"), gate_cpu)
    np.save(os.path.join(checkpoint_dir, "bias.npy"), bias_cpu)
    if alpha_fit_cpu is not None:
        np.save(os.path.join(checkpoint_dir, "alpha_fit.npy"), alpha_fit_cpu)

    # W 统一转 scipy CSR .npz（最通用、无 dense 峰值）
    if hasattr(W, 'toarray'):          # scipy sparse
        from scipy.sparse import save_npz
        save_npz(os.path.join(checkpoint_dir, "W.npz"), W)
    elif isinstance(W, torch.Tensor) and W.layout == torch.sparse_csr:
        from scipy.sparse import csr_matrix, save_npz
        W_scipy = csr_matrix(
            (W.values().cpu().numpy(),
             W.col_indices().cpu().numpy(),
             W.crow_indices().cpu().numpy()),
            shape=tuple(W.shape)
        )
        save_npz(os.path.join(checkpoint_dir, "W.npz"), W_scipy)
    elif isinstance(W, np.ndarray):
        from scipy.sparse import csr_matrix, save_npz
        save_npz(os.path.join(checkpoint_dir, "W.npz"), csr_matrix(W))
    else:
        raise TypeError(f"Unsupported W type for checkpoint: {type(W)}")

    import json
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


# ================================================================ 合成小规模数据
def _synth_small(G: int = 64, N: int = 128, seed: int = 0) -> dict:
    """--small 自检数据：G=64, N=128，u/s 沿伪时间合成 + 10D X_scVI +
    common.synth_grn(64) 先验 CSR。动力学一致的合成：
    s 取逐基因 sigmoid 伪时间轨迹，u = (ds/dt + γ s)/β（v=βu−γs=ds/dt 自洽）。
    """
    rng = np.random.default_rng(seed)
    beta, gamma, bias = (x.numpy() for x in synth_params(G, seed=seed + 1))
    t = np.sort(rng.uniform(0.0, 1.0, size=N)).astype(np.float32)
    # 逐基因 sigmoid 轨迹 s_g(t) = a_g / (1 + exp(-k_g (t - m_g))) + c_g
    a = rng.uniform(0.5, 2.0, size=G).astype(np.float32)
    k = rng.uniform(2.0, 8.0, size=G).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=G)
    m = rng.uniform(0.3, 0.7, size=G).astype(np.float32)
    c = rng.uniform(0.0, 0.5, size=G).astype(np.float32)
    tt = t[:, None]                                            # (N,1)
    sig = 1.0 / (1.0 + np.exp(-k[None, :] * (tt - m[None, :])))
    s = a[None, :] * sig + c[None, :]
    dsdt = a[None, :] * sig * (1.0 - sig) * k[None, :]         # sigmoid 导数
    u = (dsdt + gamma[None, :] * s) / beta[None, :]
    # 观测噪声（保持非负）
    Ms = np.clip(s + rng.normal(0, 0.02, size=s.shape), 0.0, None).astype(np.float32)
    Mu = np.clip(u + rng.normal(0, 0.02, size=u.shape), 0.0, None).astype(np.float32)
    # X_scVI：前两维承载伪时间，其余随机
    X = rng.normal(0, 1, size=(N, 10)).astype(np.float32)
    X[:, 0] = t * 4.0 + rng.normal(0, 0.1, size=N)
    X[:, 1] = t * 2.0 + rng.normal(0, 0.1, size=N)
    W = synth_grn(G, n_tf=11, n_feedback=6, seed=seed + 2)
    return {"Ms": Ms, "Mu": Mu, "W": W, "X_scVI": X, "t_true": t}


# ================================================================ CLI
def _load_array(path: str) -> np.ndarray:
    """加载 .npy / .npz（npz 取 'arr_0' 或首个键）。"""
    obj = np.load(path)
    if isinstance(obj, np.lib.npyio.NpzFile):
        key = "arr_0" if "arr_0" in obj.files else obj.files[0]
        return obj[key]
    return obj


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="后 RegVelo 稀疏闭式推断引擎 Phase 3 主入口")
    p.add_argument("--ms", type=str, default=None, help="Ms (N,G) .npy/.npz")
    p.add_argument("--mu", type=str, default=None, help="Mu (N,G) .npy/.npz")
    p.add_argument("--w", type=str, default=None, help="W dense (G,G) .npy/.npz")
    p.add_argument("--x-scvi", type=str, default=None, help="X_scVI (N,10)")
    p.add_argument("--out-dir", type=str, default="logs")
    p.add_argument("--n-rounds", type=int, default=2)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048,
               help="cell mini-batch size for L3 Picard propagation (default 2048 for 12GB VRAM safety)")
    p.add_argument("--picard-max-iter", type=int, default=10,
                   help="L3 Picard 迭代预算（kernel 默认 3 不收敛，本入口默认 10）")
    p.add_argument("--l2-checkpoint", type=str, default=None,
                   help="L2 完成后保存检查点到该目录（供独立 L3 脚本加载）")
    p.add_argument("--l2-only", action="store_true",
                   help="仅运行 L1+L2，保存检查点后立即退出（不进入 L3，避免 OOM）")
    p.add_argument("--rate-max", type=float, default=None,
                   help="beta/gamma 速率上限（默认 None=adaptive 自适应，可手动指定如 20.0/50.0/100.0）")
    p.add_argument("--hard", dest="hard", action="store_true", default=True)
    p.add_argument("--soft", dest="hard", action="store_false")
    p.add_argument("--small", action="store_true",
                   help="合成小规模自检（G=64, N=128）")
    args = p.parse_args(argv)

    if args.small:
        data = _synth_small(G=64, N=128)
        Ms, Mu, W, X = data["Ms"], data["Mu"], data["W"], data["X_scVI"]
        print(f"[--small] 合成数据 G={W.shape[0]}, N={Ms.shape[0]}")
    else:
        missing = [n for n, v in [("--ms", args.ms), ("--mu", args.mu),
                                  ("--w", args.w), ("--x-scvi", args.x_scvi)]
                   if v is None]
        if missing:
            p.error(f"非 --small 模式必须提供：{', '.join(missing)}")
        Ms = _load_array(args.ms)
        Mu = _load_array(args.mu)
        X = _load_array(args.x_scvi)
        
        # W 加载：区分 scipy CSR .npz 与 numpy dense .npy
        if args.w.endswith('.npz'):
            from scipy.sparse import load_npz
            W = load_npz(args.w)  # scipy CSR，直接传给 make_csr
        else:
            W = torch.from_numpy(np.asarray(_load_array(args.w), dtype=np.float32))

    W_csr = make_csr(W)  # make_csr 已扩展为支持 scipy CSR / torch / numpy
    try:
        out = infer(Ms, Mu, W_csr, X, root_cell=None, n_rounds=args.n_rounds,
                    K=args.K, batch_size=args.batch_size, hard=args.hard,
                    out_dir=args.out_dir, picard_max_iter=args.picard_max_iter,
                    l2_checkpoint_dir=args.l2_checkpoint,
                    l2_only=args.l2_only,
                    rate_max=args.rate_max)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if out.get("l2_complete"):
        print(f"[L2-only] 检查点已保存至: {out['checkpoint_dir']}")
        print(f"logs -> {out['logs']}")
        return 0

    print(f"engine_route = {out['engine_route']}")
    print(f"v shape = {tuple(out['v'].shape)}, J nnz = {out['J'].values().numel()}")
    print(f"logs -> {out['logs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
