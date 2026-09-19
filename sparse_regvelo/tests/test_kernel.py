"""tests/test_kernel.py — SPEC §4 kernel 测试契约。

1. 共振支路：β=γ±1e-8 与 β=γ 精确值的输出差 < 1e-4 相对；与极限解析式一致。
2. expm1 稳定性：γh ∈ {1e-9, 1e-6, 1e-3} 时 φ_g/ramp_g 与级数值一致（rel err < 1e-6），无 NaN。
3. CSR 梯度回传：alpha_nodes 前向后反传，W_csr.grad 为 CSR 且非零模式 = W 模式。
4. 正确性：K=40 对 DOP853 参考的 L2 rel err < 1e-3（小规模）。

pytest 风格，亦可直接 `python tests/test_kernel.py` 运行全部。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import make_csr, rel_l2, simulate_reference, synth_grn, synth_params
from closed_form_kernel import (_segment_kernels, alpha_nodes,
                                propagate_closed_form, segment_knots)


# ---------------------------------------------------------------- 1. 共振支路
def test_resonance_branch():
    torch.manual_seed(0)
    N, G, K = 4, 16, 5
    t = torch.linspace(0.0, 2.0, K + 1)
    u0 = torch.rand(N, G)
    s0 = torch.rand(N, G)
    alpha = torch.rand(N, G, K + 1) * 2.0
    gamma = torch.full((G,), 1.2345)

    u_eq, s_eq = propagate_closed_form(u0, s0, alpha, t, gamma, gamma)
    for eps in (1e-8, -1e-8):
        beta = gamma + eps
        u_eps, s_eps = propagate_closed_form(u0, s0, alpha, t, beta, gamma)
        assert rel_l2(u_eps.numpy(), u_eq.numpy()) < 1e-4, f"u 共振连续性失败 eps={eps}"
        assert rel_l2(s_eps.numpy(), s_eq.numpy()) < 1e-4, f"s 共振连续性失败 eps={eps}"

    # 与极限解析式一致（float64 手算 β=γ 的极限公式逐步对拍）
    u = u0.numpy().astype(np.float64)
    s = s0.numpy().astype(np.float64)
    g = gamma.numpy().astype(np.float64)
    al = alpha.numpy().astype(np.float64)
    tn = t.numpy().astype(np.float64)
    for k in range(K):
        h = tn[k + 1] - tn[k]
        a = al[..., k]
        b = (al[..., k + 1] - a) / h
        C = a / g - b / g**2
        A = u - C
        E = np.exp(-g * h)
        mix = h * E                       # 共振极限
        phi = -np.expm1(-g * h) / g
        ramp = (g * h + np.expm1(-g * h)) / g**2
        u = A * E + (b / g) * h + C
        s = s * E + g * (A * mix + C * phi) + b * ramp
    assert rel_l2(u_eq.numpy(), u) < 1e-4, "u 与共振极限解析式不一致"
    assert rel_l2(s_eq.numpy(), s) < 1e-4, "s 与共振极限解析式不一致"
    print("[PASS] test_resonance_branch")


# ---------------------------------------------------------------- 2. expm1 稳定性
def test_expm1_stability():
    h = 1.0
    for x in (1e-9, 1e-6, 1e-3):
        gamma = torch.tensor([x / h], dtype=torch.float32)
        beta = torch.tensor([1.0], dtype=torch.float32)
        _, _, mix, phi_g, ramp_g = _segment_kernels(beta, gamma, torch.tensor(h))
        assert torch.isfinite(phi_g).all() and torch.isfinite(ramp_g).all(), "NaN/Inf"
        assert torch.isfinite(mix).all()

        # float64 高精度参考（级数值 / 精确闭式，二者在该精度下等价）
        g = x / h
        phi_true = -np.expm1(-g * h) / g
        ramp_true = (g * h + np.expm1(-g * h)) / g**2
        phi_ser = h - g * h * h / 2.0
        ramp_ser = h * h / 2.0 - g * h**3 / 6.0 + g**2 * h**4 / 24.0
        # 与级数值一致（rel err < 1e-6）
        assert abs(float(phi_g) - phi_ser) / abs(phi_ser) < 1e-6, \
            f"φ_g 在 γh={x} 偏离级数: {float(phi_g)} vs {phi_ser}"
        assert abs(float(ramp_g) - ramp_ser) / abs(ramp_ser) < 1e-6, \
            f"ramp_g 在 γh={x} 偏离级数: {float(ramp_g)} vs {ramp_ser}"
        # 同时与精确闭式一致
        assert abs(float(phi_g) - phi_true) / abs(phi_true) < 1e-6
        assert abs(float(ramp_g) - ramp_true) / abs(ramp_true) < 1e-6
    print("[PASS] test_expm1_stability")


# ---------------------------------------------------------------- 3. CSR 梯度回传
def test_csr_gradient():
    torch.manual_seed(0)
    N, G, K1 = 8, 32, 4
    W = make_csr(synth_grn(G, n_tf=5, n_feedback=3, seed=0))
    bias = synth_params(G, seed=1)[2]
    s_knots = torch.rand(N, G, K1)

    W_req = W.detach().clone().requires_grad_(True)
    alpha = alpha_nodes(W_req, s_knots, bias)
    loss = alpha.square().mean()
    loss.backward()

    g = W_req.grad
    assert g is not None, "W_csr.grad 为空"
    assert g.layout == torch.sparse_csr, f"grad 非 CSR: {g.layout}"
    # 非零模式 = W 模式（crow/col 完全一致，且所有存储位置梯度非零）
    assert torch.equal(g.crow_indices(), W_req.crow_indices()), "crow 模式不一致"
    assert torch.equal(g.col_indices(), W_req.col_indices()), "col 模式不一致"
    assert bool((g.values() != 0).all()), "存在存储位置梯度恰为 0"
    print("[PASS] test_csr_gradient")


# ---------------------------------------------------------------- 4. K=40 vs DOP853
def test_k40_vs_dop853():
    G, N = 64, 16
    W = make_csr(synth_grn(G, n_tf=11, n_feedback=6, seed=0))
    beta, gamma, bias = synth_params(G, seed=1)
    u0 = torch.zeros(N, G)
    s0 = torch.zeros(N, G)

    ref = simulate_reference(u0, s0, W, bias, beta, gamma,
                             t_span=(0.0, 10.0), n_times=41)
    # K=40 uniform 节点与参考 t_eval (41 点) 重合，oracle-α 直接由参考 s 采样
    knots = segment_knots(10.0, 40, strategy="uniform")
    assert np.allclose(knots.numpy(), ref["t"], atol=1e-6)
    s_ref_knots = torch.from_numpy(ref["s"].astype(np.float32)).permute(1, 2, 0).contiguous()
    alpha = alpha_nodes(W, s_ref_knots, bias)
    u, s = propagate_closed_form(u0, s0, alpha, knots, beta, gamma)

    l2_u = rel_l2(u.numpy(), ref["u"][-1])
    l2_s = rel_l2(s.numpy(), ref["s"][-1])
    l2_all = rel_l2(np.concatenate([u.numpy().ravel(), s.numpy().ravel()]),
                    np.concatenate([ref["u"][-1].ravel(), ref["s"][-1].ravel()]))
    assert l2_all < 1e-3, f"K=40 vs DOP853 L2={l2_all:.3e} >= 1e-3 (u={l2_u:.2e}, s={l2_s:.2e})"
    print(f"[PASS] test_k40_vs_dop853 (L2={l2_all:.2e})")


if __name__ == "__main__":
    test_resonance_branch()
    test_expm1_stability()
    test_csr_gradient()
    test_k40_vs_dop853()
    print("ALL KERNEL TESTS PASSED")
