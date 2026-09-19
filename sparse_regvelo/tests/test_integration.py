"""tests/test_integration.py — Phase 3 端到端小规模测试（SPEC §4）。

覆盖：
1. analytic_jacobian 数值正确性（解析式 vs 中心差分对照，rel err < 1e-3；
   差分仅允许出现在本测试中，引擎本体禁止数值差分）。
2. J 稀疏模式 ⊆ W 模式（实现上逐 value 复用 W 的 crow/col，模式恒等）。
3. select_engine 四分支全覆盖。
4. infer 小规模端到端（G=64, N=128）：依赖（closed_form_kernel /
   decoupled_velocity_engine）就绪时跑通，检查输出键齐全、
   v == beta*u - gamma*s 逐元素、peak_memory_gb 已记录；依赖缺失则 SKIP。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import DEVICE, make_csr, synth_grn, synth_params  # noqa: E402
from sparse_regvelo_engine import (  # noqa: E402
    _DEPS_OK,
    _DEPS_ERR,
    analytic_jacobian,
    select_engine,
)

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False


# ---------------------------------------------------------------- 工具
def _small_system(G: int = 32, N: int = 8, seed: int = 0):
    """构造小系统：W 稀疏、状态取小尺度使 z 远离 clamp 边界（导数非退化）。"""
    rng = np.random.default_rng(seed)
    W = synth_grn(G, n_tf=6, edges_per_tf=3, n_feedback=2, seed=seed)
    # 随机权重幅值（先验 0/1 上叠加符号与尺度），保持稀疏模式
    mask = W.numpy()
    Ww = mask * rng.uniform(-0.3, 0.6, size=mask.shape).astype(np.float32)
    beta, gamma, bias = (x.numpy() for x in synth_params(G, seed=seed + 1))
    bias = bias * 0.2  # 缩小 bias，z 远离 softplus 饱和区
    s = rng.uniform(0.0, 1.0, size=(N, G)).astype(np.float32)
    return Ww, s, beta, gamma, bias


def _v_qss(s: np.ndarray, W: np.ndarray, beta: np.ndarray,
           gamma: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """准稳态闭合速度（float64，数值对照用）：
    u ≈ clamp(softplus(W s + b),0,50)/β，v = βu − γs。
    动力学约定与引擎/参考积分器一致：z_i = sum_j W_ij s_j（z = W s）。"""
    z = (W @ s.T).T + bias[None, :]  # 动力学约定 z = W s + b，(N,G)
    sp = np.logaddexp(0.0, z)
    alpha = np.clip(sp, 0.0, 50.0)
    u = alpha / beta[None, :]
    return beta[None, :] * u - gamma[None, :] * s


# ---------------------------------------------------------------- 1) Jacobian 数值正确性
def test_analytic_jacobian_vs_central_diff():
    G, N = 32, 8
    Ww, s, beta, gamma, bias = _small_system(G=G, N=N)
    W_csr = make_csr(torch.from_numpy(Ww))
    out = analytic_jacobian(W_csr, torch.from_numpy(s),
                            torch.from_numpy(beta), torch.from_numpy(gamma),
                            torch.from_numpy(bias))
    J = out["J"].to_dense().cpu().numpy().astype(np.float64)
    J_diag = out["diag"].cpu().numpy().astype(np.float64)
    assert np.allclose(J_diag, gamma), "diag 应等于 gamma（-γ_i 对角单独返回）"

    # 代表点：batch 均值（与实现一致）
    s_rep = s.mean(axis=0).astype(np.float64)
    W64 = Ww.astype(np.float64)
    eps = 1e-4
    max_rel = 0.0
    rows, cols = np.nonzero(Ww)
    assert len(rows) > 0
    for i, j in zip(rows, cols):
        sp = s_rep.copy(); sp[j] += eps
        sm = s_rep.copy(); sm[j] -= eps
        dv_i = (_v_qss(sp[None], W64, beta, gamma, bias)[0, i]
                - _v_qss(sm[None], W64, beta, gamma, bias)[0, i]) / (2 * eps)
        ana = J[i, j]
        rel = abs(dv_i - ana) / max(1e-6, abs(dv_i), abs(ana))
        max_rel = max(max_rel, rel)
        assert rel < 1e-3, f"J[{i},{j}] 解析={ana:.6e} 差分={dv_i:.6e} rel={rel:.2e}"
    # 对角元数值对照：∂v_i/∂s_i 应 = J_ii（若 (i,i)∈W）− γ_i
    i = int(rows[0])
    sp = s_rep.copy(); sp[i] += eps
    sm = s_rep.copy(); sm[i] -= eps
    dv_ii = (_v_qss(sp[None], W64, beta, gamma, bias)[0, i]
             - _v_qss(sm[None], W64, beta, gamma, bias)[0, i]) / (2 * eps)
    ana_ii = (J[i, i] if Ww[i, i] != 0 else 0.0) - gamma[i]
    assert abs(dv_ii - ana_ii) / max(1e-6, abs(dv_ii)) < 1e-3
    print(f"[jacobian] max rel err over nnz = {max_rel:.2e}")


# ---------------------------------------------------------------- 2) J 稀疏模式 ⊆ W 模式
def test_jacobian_sparsity_pattern():
    G = 32
    Ww, s, beta, gamma, bias = _small_system(G=G, N=4, seed=7)
    # 构造极端 s 使部分导数置 0（clamp 边界外），验证模式仍 ⊆ W
    s_big = np.full((4, G), 1e4, dtype=np.float32)
    W_csr = make_csr(torch.from_numpy(Ww))
    out = analytic_jacobian(W_csr, torch.from_numpy(s_big),
                            torch.from_numpy(beta), torch.from_numpy(gamma),
                            torch.from_numpy(bias))
    J = out["J"]
    assert J.layout == torch.sparse_csr
    # 实现逐 value 复用 W 的 crow/col → 模式恒等（即使数值为 0）
    assert torch.equal(J.crow_indices(), W_csr.crow_indices())
    assert torch.equal(J.col_indices(), W_csr.col_indices())
    # 数值 ⊆ 模式：J 非零位置必在 W 非零位置（恒等模式的直接推论，显式断言）
    Jd = J.to_dense().cpu().numpy()
    assert np.all((Jd != 0) <= (Ww != 0))
    # clamp 饱和（alpha=50）时导数应置 0
    assert np.allclose(Jd, 0.0), "alpha 触 50 时 softplus' 应置 0"


# ---------------------------------------------------------------- 3) select_engine 四分支
def test_select_engine_four_branches():
    assert select_engine(True, 0.0, 0.0) == "full_dynamics_path"
    assert select_engine(np.array([False, True]), 0.0, 0.0) == "full_dynamics_path"
    assert select_engine(False, 2e-3, 0.05) == "krylov_expmv"
    assert select_engine(False, 2e-3, 0.1) == "full_sparse_expmv"
    assert select_engine(False, 2e-3, 0.2) == "full_sparse_expmv"
    assert select_engine(False, 1e-4, 0.5) == "closed_form_foh"
    assert select_engine(False, 1e-3, 0.5) == "closed_form_foh"  # 边界：> 1e-3 才降级
    assert select_engine(torch.zeros(4, dtype=torch.bool), 0.0, 0.0) \
        == "closed_form_foh"


# ---------------------------------------------------------------- 4) infer 端到端（依赖就绪时）
def test_infer_small_end_to_end(tmp_path=None):
    if not _DEPS_OK:
        msg = f"SKIP: 依赖未就绪（{_DEPS_ERR}），跳过 infer 端到端测试"
        print(msg)
        if _HAS_PYTEST:
            pytest.skip(msg)
        return
    from sparse_regvelo_engine import _synth_small, infer

    out_dir = str(tmp_path) if tmp_path is not None else "logs"
    data = _synth_small(G=64, N=128)
    W_csr = make_csr(data["W"])
    out = infer(data["Ms"], data["Mu"], W_csr, data["X_scVI"],
                root_cell=0, n_rounds=2, K=10, batch_size=64,
                hard=True, out_dir=out_dir)

    # 输出键齐全
    for key in ("t", "v", "alpha", "beta", "gamma", "J", "J_diag",
                "gate", "engine_route", "u", "s", "logs"):
        assert key in out, f"缺少输出键 {key}"
    N, G = data["Ms"].shape
    assert out["t"].shape == (N,)
    assert out["v"].shape == (N, G)
    assert out["beta"].shape == (G,) and out["gamma"].shape == (G,)
    assert out["J"].shape == (G, G) and out["J"].layout == torch.sparse_csr

    # v == beta*u - gamma*s 逐元素（直算红线，禁止重解 ODE）
    v_ref = out["beta"].unsqueeze(0) * out["u"] - out["gamma"].unsqueeze(0) * out["s"]
    assert torch.allclose(out["v"], v_ref, atol=1e-6), "v 必须逐元素等于 βu−γs"

    # J 稀疏模式 ⊆ W 模式
    assert torch.equal(out["J"].crow_indices(), W_csr.crow_indices())
    assert torch.equal(out["J"].col_indices(), W_csr.col_indices())

    # engine_route 合法取值（记录的是非 gated 补集路由结果）
    assert out["engine_route"] in ("closed_form_foh", "krylov_expmv",
                                   "full_sparse_expmv", "full_dynamics_path")

    # peak_memory_gb / wall_time_sec / device 已记录
    with open(out["logs"], encoding="utf-8") as f:
        log = json.load(f)
    for key in ("peak_memory_gb", "wall_time_sec", "device",
                "picard_residual_max", "krylov_residual_probe",
                "n_gated", "gated_genes_route", "picard_max_iter",
                "n_unconverged_batches"):
        assert key in log, f"日志缺少 {key}"
    assert log["device"] == DEVICE
    # 路由粒度：gated 子集固定 full_dynamics_path，不掩盖补集 picard/krylov 升级
    if log["n_gated"] > 0:
        assert log["gated_genes_route"] == "full_dynamics_path"
    n_batches = (N + 64 - 1) // 64  # batch_size=64
    assert sum(log["route_counts"].values()) == n_batches
    # 补集路由永不返回 full_dynamics_path（select_engine(False, ...)）
    assert out["engine_route"] != "full_dynamics_path"

    # n_rounds=0 红线断言
    try:
        infer(data["Ms"], data["Mu"], W_csr, data["X_scVI"], n_rounds=0,
              out_dir=out_dir)
        raise SystemExit("n_rounds=0 未触发断言")
    except AssertionError:
        pass
    print(f"[infer] end-to-end OK, engine_route={out['engine_route']}, "
          f"peak={log['peak_memory_gb']} GB")


# ---------------------------------------------------------------- 5) alpha 存在性分支（N1）
def test_infer_alpha_from_theta(tmp_path=None):
    """theta 提供 alpha（np.ndarray，(G,) 与 (N,G) 两种形状）时，
    infer 输出的 alpha 必须与之一致（N1：ndarray/tensor 存在性判断）。"""
    if not _DEPS_OK:
        msg = f"SKIP: 依赖未就绪（{_DEPS_ERR}），跳过 alpha 分支测试"
        print(msg)
        if _HAS_PYTEST:
            pytest.skip(msg)
        return
    import sparse_regvelo_engine as eng
    from sparse_regvelo_engine import _synth_small

    out_dir = str(tmp_path) if tmp_path is not None else "logs"
    data = _synth_small(G=64, N=128)
    W_csr = make_csr(data["W"])
    N, G = data["Ms"].shape
    real_decoupled = eng.decoupled_engine
    rng = np.random.default_rng(0)
    for alpha_shape in ((G,), (N, G)):
        alpha_known = rng.uniform(0.1, 2.0, size=alpha_shape).astype(np.float32)

        def fake_decoupled(Mu, Ms, X, root_cell=None, n_rounds=2, K=10,
                           _alpha=alpha_known):
            theta, t, gate, diag = real_decoupled(
                Mu, Ms, X, root_cell=root_cell, n_rounds=n_rounds, K=K)
            theta = dict(theta)
            theta["alpha"] = _alpha  # np.ndarray，验证 N1 修复后的分支
            return theta, t, gate, diag

        eng.decoupled_engine = fake_decoupled
        try:
            out = eng.infer(data["Ms"], data["Mu"], W_csr, data["X_scVI"],
                            n_rounds=1, K=4, batch_size=64, out_dir=out_dir)
        finally:
            eng.decoupled_engine = real_decoupled
        ref = torch.as_tensor(alpha_known)
        assert out["alpha"].shape == ref.shape, \
            f"alpha 形状 {tuple(out['alpha'].shape)} != {alpha_shape}"
        assert torch.allclose(out["alpha"], ref, atol=1e-6), \
            "theta 提供 alpha 时 infer 输出必须与之逐元素一致"
    print("[alpha] theta-provided alpha passthrough OK (G,) & (N,G)")


# ---------------------------------------------------------------- 6) Krylov 对拍（N3）
def test_arnoldi_expmv_vs_scipy():
    """G=32 非对称稀疏 A、随机 v：arnoldi_expmv(t=0.5, m=20) vs
    scipy.sparse.linalg.expm_multiply(dense)，rel err < 1e-6。"""
    if not _DEPS_OK:
        msg = f"SKIP: 依赖未就绪（{_DEPS_ERR}），跳过 arnoldi 对拍"
        print(msg)
        if _HAS_PYTEST:
            pytest.skip(msg)
        return
    from scipy.sparse.linalg import expm_multiply
    from closed_form_kernel import arnoldi_expmv

    G = 32
    rng = np.random.default_rng(42)
    A = rng.standard_normal((G, G)).astype(np.float32)
    A[rng.random((G, G)) > 0.25] = 0.0          # 稀疏化（~25% 密度）
    A *= 0.5 / np.linalg.norm(A, 2)             # 缩谱半径保证 m=20 收敛到机器精度
    v = rng.standard_normal(G).astype(np.float32)
    t = 0.5

    w, residual_est = arnoldi_expmv(make_csr(torch.from_numpy(A)),
                                    torch.from_numpy(v), t, m=20)
    # expm_multiply 计算 exp(M)·v：取 M = t·A 对齐 kernel 的 exp(t·A)·v
    w_ref = expm_multiply((t * A).astype(np.float64), v.astype(np.float64))
    rel = float(np.linalg.norm(w.cpu().numpy().astype(np.float64) - w_ref)
                / np.linalg.norm(w_ref))
    assert rel < 1e-6, f"arnoldi_expmv 对拍 rel err = {rel:.2e}"
    print(f"[arnoldi] rel err vs expm_multiply = {rel:.2e}, "
          f"residual_est = {residual_est:.2e}")


def test_coupled_backup_vs_dense_expm():
    """coupled_backup_engine vs 精确线性化 expm（dense 2G×2G，G=16，
    测试内允许 dense）：rel err < 1e-4。L = [[-βI, D⊙W],[βI, -γI]]，
    D = diag(σ(z0)), z0 = W s0 + b（逐细胞）。"""
    if not _DEPS_OK:
        msg = f"SKIP: 依赖未就绪（{_DEPS_ERR}），跳过 coupled 对拍"
        print(msg)
        if _HAS_PYTEST:
            pytest.skip(msg)
        return
    from scipy.linalg import expm as scipy_expm
    from closed_form_kernel import coupled_backup_engine

    G, N = 16, 3
    rng = np.random.default_rng(7)
    W = rng.standard_normal((G, G)).astype(np.float32)
    W[rng.random((G, G)) > 0.3] = 0.0
    W *= 0.3 / max(np.linalg.norm(W, 2), 1e-12)
    beta = rng.uniform(0.5, 1.5, size=G).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=G).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, size=G).astype(np.float32)
    u0 = rng.uniform(0.0, 0.5, size=(N, G)).astype(np.float32)
    s0 = rng.uniform(0.0, 0.5, size=(N, G)).astype(np.float32)
    t_eval = np.linspace(0.0, 0.5, 6).astype(np.float32)
    t_span = float(t_eval[-1] - t_eval[0])

    u_out, s_out = coupled_backup_engine(
        torch.from_numpy(u0), torch.from_numpy(s0),
        make_csr(torch.from_numpy(W)), torch.from_numpy(bias),
        torch.from_numpy(beta), torch.from_numpy(gamma),
        torch.from_numpy(t_eval), m=30)

    # 精确参考：逐细胞 dense expm（测试内允许）
    u_ref = np.zeros((N, G), dtype=np.float64)
    s_ref = np.zeros((N, G), dtype=np.float64)
    for n in range(N):
        z0 = W.astype(np.float64) @ s0[n].astype(np.float64) + bias
        D = 1.0 / (1.0 + np.exp(-z0))            # σ(z0)，值域小无 clamp 饱和
        L = np.zeros((2 * G, 2 * G), dtype=np.float64)
        L[:G, :G] = -np.diag(beta)
        L[:G, G:] = np.diag(D) @ W
        L[G:, :G] = np.diag(beta)
        L[G:, G:] = -np.diag(gamma)
        y0 = np.concatenate([u0[n], s0[n]]).astype(np.float64)
        y = scipy_expm(t_span * L) @ y0
        u_ref[n], s_ref[n] = y[:G], y[G:]

    err_u = float(np.linalg.norm(u_out.cpu().numpy() - u_ref)
                  / np.linalg.norm(u_ref))
    err_s = float(np.linalg.norm(s_out.cpu().numpy() - s_ref)
                  / np.linalg.norm(s_ref))
    assert err_u < 1e-4 and err_s < 1e-4, \
        f"coupled_backup_engine 对拍 rel err u={err_u:.2e} s={err_s:.2e}"
    print(f"[coupled] rel err vs dense expm: u={err_u:.2e}, s={err_s:.2e}")


if __name__ == "__main__":
    test_analytic_jacobian_vs_central_diff()
    test_jacobian_sparsity_pattern()
    test_select_engine_four_branches()
    test_infer_small_end_to_end()
    test_infer_alpha_from_theta()
    test_arnoldi_expmv_vs_scipy()
    test_coupled_backup_vs_dense_expm()
    print("ALL INTEGRATION TESTS DONE")
