"""tests/test_pipeline.py — Phase 2 C 管线（L1/L2）组件测试（SPEC §4）。

覆盖：
  - steady_state_init：稳态直线基因上 gamma/beta 斜率比值恢复
  - steady_state_gate：三指标正反例（稳态直线 / 弯曲诱导 / 双 regime / 无相关云）
  - pseudotime_from_latent：线性流形上 Spearman > 0.95；多根返回 dict
  - anchor_direction：人工反转 t 后 flipped=True 翻回
  - fit_genes_closed_form / reassign_time：参数恢复与时间纠偏
  - decoupled_engine：n_rounds=0 触发 AssertionError；小规模端到端形状
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoupled_velocity_engine import (  # noqa: E402
    _trajectory,
    anchor_direction,
    decoupled_engine,
    fit_genes_closed_form,
    pseudotime_from_latent,
    reassign_time,
    steady_state_gate,
    steady_state_init,
    _synth_small,
)


# ---------------------------------------------------------------- L1 初值
def test_steady_state_init_recovers_ratio():
    rng = np.random.default_rng(0)
    N = 400
    r_true = 1.7
    s = rng.uniform(0.0, 5.0, N)
    u = r_true * s + 0.05 * rng.standard_normal(N)
    init = steady_state_init(u[:, None], s[:, None])
    assert init["gamma_over_beta"].shape == (1,)
    assert init["alpha_over_beta"].shape == (1,)
    # 仅比值输出；斜率比值应接近真实 gamma/beta
    assert abs(init["gamma_over_beta"][0] - r_true) / r_true < 0.1
    # 高分位细胞的 alpha/beta ≈ 高分位 u 均值（稳态关系），量级正确
    assert init["alpha_over_beta"][0] > 0.0


# ---------------------------------------------------------------- L1 门控三指标
def _three_indicator_genes(seed=42, N=200):
    rng = np.random.default_rng(seed)
    # 稳态直线基因（曲率退化 -> 违反 (ii)）
    s1 = rng.uniform(0.0, 5.0, N)
    u1 = 0.8 * s1 + 0.01 * rng.standard_normal(N)
    # 弯曲诱导基因（单 regime 健康动力学 -> 三指标均不违反）
    t_g = rng.uniform(0.0, 1.0, N)
    u2, s2 = _trajectory(t_g, 1.5, 1.2, 6.0, 0.1, 0.5)
    u2 = u2 + 0.05 * rng.standard_normal(N)
    s2 = s2 + 0.05 * rng.standard_normal(N)
    # 双 regime 基因（两条平行线，u 方向 offset -> 违反 (iii)）
    s3 = rng.uniform(0.0, 5.0, N)
    lab = rng.binomial(1, 0.5, N)
    u3 = 0.8 * s3 + 3.0 * lab + 0.1 * rng.standard_normal(N)
    Mu = np.stack([u1, u2, u3], axis=1)
    Ms = np.stack([s1, s2, s3], axis=1)
    return Mu, Ms


def test_gate_three_indicators():
    Mu, Ms = _three_indicator_genes()
    gate = steady_state_gate(Mu, Ms)
    assert gate.shape == (3,) and gate.dtype == bool
    # 稳态直线：相图退化 -> gate=True；弯曲诱导：gate=False；双 regime：gate=True
    assert gate[0] == True   # noqa: E712
    assert gate[1] == False  # noqa: E712
    assert gate[2] == True   # noqa: E712


def test_gate_corr_indicator():
    rng = np.random.default_rng(1)
    N = 200
    # 无相关云：corr≈0 < corr_thresh -> gate=True（违反 (i)）
    u = rng.standard_normal(N)
    s = rng.standard_normal(N)
    assert steady_state_gate(u[:, None], s[:, None])[0] == True  # noqa: E712
    # 强相关且带曲率的单 regime 基因 -> gate=False（正例对照见 test_gate_three_indicators）
    t_g = rng.uniform(0.0, 1.0, N)
    u2, s2 = _trajectory(t_g, 2.0, 0.8, 8.0, 0.2, 0.6)
    assert steady_state_gate(u2[:, None], s2[:, None])[0] == False  # noqa: E712


# ---------------------------------------------------------------- pseudotime
def test_pseudotime_linear_manifold_spearman():
    rng = np.random.default_rng(7)
    N = 200
    t_true = rng.uniform(0.0, 1.0, N)
    v = rng.standard_normal(10)
    X = t_true[:, None] * v[None, :] + 0.02 * rng.standard_normal((N, 10))
    root = int(np.argmin(t_true))
    t = pseudotime_from_latent(X, root)
    assert t.shape == (N,)
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert spearmanr(t, t_true).statistic > 0.95


def test_pseudotime_multi_root_dict():
    rng = np.random.default_rng(8)
    N = 100
    t_true = rng.uniform(0.0, 1.0, N)
    X = t_true[:, None] * rng.standard_normal(10)[None, :] + 0.02 * rng.standard_normal((N, 10))
    out = pseudotime_from_latent(X, [0, 50])
    assert isinstance(out, dict) and set(out.keys()) == {0, 50}
    for v in out.values():
        assert v.shape == (N,)


# ---------------------------------------------------------------- 方向锚定
def test_anchor_flips_reversed_time():
    rng = np.random.default_rng(11)
    N, G = 300, 8
    t_true = rng.uniform(0.0, 1.0, N)
    u, s = _trajectory(t_true[None, :], np.full((G, 1), 1.5), np.full((G, 1), 1.2),
                       np.full((G, 1), 6.0), np.full((G, 1), 0.1), np.full((G, 1), 0.5))
    Mu = u.T + 0.05 * np.abs(rng.standard_normal((N, G)))
    Ms = s.T + 0.05 * np.abs(rng.standard_normal((N, G)))
    t_rev = 1.0 - t_true  # 人工反转
    t_an, flipped, score = anchor_direction(t_rev, Mu, Ms)
    assert flipped == True  # noqa: E712（u 先于 s，应识别出反转）
    assert score < 0.0
    assert spearmanr(t_an, t_true).statistic > 0.99
    # 未反转输入不应翻转
    t_ok, flipped2, score2 = anchor_direction(t_true, Mu, Ms)
    assert flipped2 == False  # noqa: E712
    assert score2 > 0.0


# ---------------------------------------------------------------- n_rounds 红线
def test_n_rounds_zero_assertion():
    Mu, Ms, X, _ = _synth_small(G=8, N=32, seed=5)
    with pytest.raises(AssertionError):
        decoupled_engine(Mu, Ms, X, n_rounds=0)


# ---------------------------------------------------------------- M/E 步组件
def test_fit_genes_closed_form_recovers_params():
    Mu, Ms, X, truth = _synth_small(G=32, N=256, seed=9)
    init = steady_state_init(Mu, Ms)
    gate = steady_state_gate(Mu, Ms)
    theta = fit_genes_closed_form(Mu, Ms, truth["t_true"], init, gate, K=8)
    assert theta["beta"].shape == (32,) and theta["gamma"].shape == (32,)
    assert theta["alpha"].shape == (256, 32)
    assert theta["loss"].shape == (32,)
    assert np.all(theta["beta"] > 0) and np.all(theta["gamma"] > 0)
    ratio_fit = theta["gamma"] / theta["beta"]
    ratio_true = truth["gamma"] / truth["beta"]
    assert spearmanr(ratio_fit, ratio_true).statistic > 0.8
    assert np.median(theta["loss"]) < 0.2


def test_reassign_time_improves_perturbed_t():
    Mu, Ms, X, truth = _synth_small(G=32, N=256, seed=10)
    init = steady_state_init(Mu, Ms)
    gate = steady_state_gate(Mu, Ms)
    theta = fit_genes_closed_form(Mu, Ms, truth["t_true"], init, gate, K=8)
    theta["velocity_mask"] = ~gate
    rng = np.random.default_rng(0)
    t_pert = np.clip(truth["t_true"] + 0.1 * rng.standard_normal(256), 0.0, 1.0)
    t_new = reassign_time(Mu, Ms, theta, t_pert)
    assert t_new.shape == (256,)
    assert t_new.min() >= 0.0 and t_new.max() <= 1.0
    sp_before = spearmanr(t_pert, truth["t_true"]).statistic
    sp_after = spearmanr(t_new, truth["t_true"]).statistic
    assert sp_after > sp_before


# ---------------------------------------------------------------- 端到端
def test_decoupled_engine_end_to_end_small():
    G, N = 32, 128
    Mu, Ms, X, truth = _synth_small(G=G, N=N, seed=3)
    theta, t, gate, diagnostics = decoupled_engine(Mu, Ms, X, n_rounds=2, K=6)
    # 形状与类型
    assert theta["beta"].shape == (G,) and theta["gamma"].shape == (G,)
    assert theta["alpha"].shape == (N, G)
    assert theta["alpha_on"].shape == (G,) and theta["alpha_off"].shape == (G,)
    assert theta["t_switch"].shape == (G,) and theta["loss"].shape == (G,)
    assert t.shape == (N,) and gate.shape == (G,) and gate.dtype == bool
    # 诊断记录完整（n_rounds 纠偏）
    assert len(diagnostics["rounds"]) >= 1
    assert "param_rel_change" in diagnostics["rounds"][0]
    assert "n_gated" in diagnostics and "anchor_flipped" in diagnostics
    # 合成数据上时间恢复
    assert spearmanr(t, truth["t_true"]).statistic > 0.9
