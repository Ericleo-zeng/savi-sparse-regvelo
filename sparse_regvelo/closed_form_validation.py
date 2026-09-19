"""closed_form_validation.py — Phase 1 A0 验证 + E3 分段精度扫描（SPEC §3.3）。

合成场景：G=256, N=64, 11-TF 星形 GRN + 反馈边；u0=s0=0；t_span=(0,10)。
参考系：common.simulate_reference（DOP853, rtol=1e-9）。
扫描 K ∈ {1,2,4,8,10,20,40} × 段策略 {uniform, adaptive} × {oracle-α, picard 自洽}。
指标：trajectory L2 rel error、velocity cosine median（v=βu−γs at t_end）、
含反馈边基因子集单独统计（feedback_gene_metrics）。
验收：K=10 oracle cosine>0.85 且 L2<1e-3；否则 K=20；仍不达标 recommended_engine="krylov_expmv"。
输出：JSON + logs/closed_form_convergence.png（log-y 收敛曲线）+ E3_pass 布尔。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common import (DEVICE, JsonLog, make_csr, rel_l2, simulate_reference,
                    synth_grn, synth_params, velocity_cosine)
from closed_form_kernel import (alpha_nodes, picard_propagate,
                                propagate_closed_form, segment_knots)

T_SPAN = (0.0, 10.0)
N_TIMES = 41
ACCEPT_COSINE = 0.85
ACCEPT_L2 = 1e-3


# ---------------------------------------------------------------- 参考与工具
def reference_dense(u0, s0, W_csr, bias, beta, gamma, t_span):
    """与 common.simulate_reference 同一 rhs 的 dense-output DOP853（用于任意时刻节点采样）。
    common.py 接口冻结不可改，故在此复刻 rhs（公式逐行一致）。"""
    from scipy.integrate import solve_ivp

    u0n = np.asarray(u0, dtype=np.float64)
    s0n = np.asarray(s0, dtype=np.float64)
    N, G = u0n.shape
    Wn = W_csr.to_dense().cpu().numpy().astype(np.float64)
    bn = bias.cpu().numpy().astype(np.float64)
    betan = beta.cpu().numpy().astype(np.float64)
    gamman = gamma.cpu().numpy().astype(np.float64)

    def rhs(t, y):
        u = y[: N * G].reshape(N, G)
        s = y[N * G:].reshape(N, G)
        z = s @ Wn.T + bn
        a = np.clip(np.logaddexp(0.0, z), 0.0, 50.0)
        du = a - betan[None, :] * u
        ds = betan[None, :] * u - gamman[None, :] * s
        return np.concatenate([du.ravel(), ds.ravel()])

    y0 = np.concatenate([u0n.ravel(), s0n.ravel()])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", dense_output=True,
                    rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"DOP853 dense reference failed: {sol.message}")
    return sol


def sample_s_at(sol, times, N, G):
    """从 dense-output 参考解采样 s（-> (len(times), N, G) float32 torch）。"""
    y = sol.sol(np.asarray(times, dtype=np.float64))     # (2NG, T)
    s = y[N * G:].reshape(N, G, -1).transpose(2, 0, 1)   # (T,N,G)
    return torch.from_numpy(s.astype(np.float32)).to(DEVICE)


def sample_trajectory(u0, s0, alpha_knots, t_knots, t_eval, beta, gamma):
    """用 FOH 闭式核在任意 t_eval 时刻采样轨迹（段内用同一 α 线性插值精确求值）。
    返回 (u_traj, s_traj)，各 (len(t_eval), N, G)。"""
    t_knots = torch.as_tensor(t_knots, dtype=torch.float32, device=u0.device)
    t_eval = np.asarray(t_eval, dtype=np.float64)
    K = t_knots.numel() - 1
    u_k, s_k = u0, s0
    us, ss = [u0], [s0]
    k = 0
    for te in t_eval[1:]:
        while k < K - 1 and float(t_knots[k + 1]) < te - 1e-12:
            u_k, s_k = propagate_closed_form(
                u_k, s_k, alpha_knots[..., k:k + 2], t_knots[k:k + 2], beta, gamma)
            k += 1
        t_pair = torch.stack([t_knots[k], torch.tensor(float(te), dtype=torch.float32,
                                                       device=u0.device)])
        u_e, s_e = propagate_closed_form(
            u_k, s_k, alpha_knots[..., k:k + 2], t_pair, beta, gamma)
        if float(t_knots[k + 1]) <= te + 1e-9:  # 到达段末，推进 knot 状态
            u_k, s_k = u_e, s_e
            k = min(k + 1, K - 1)
        us.append(u_e)
        ss.append(s_e)
    return torch.stack(us), torch.stack(ss)


def picard_with_knots(u0, s0, W_csr, bias, beta, gamma, t_knots, max_iter=3, tol=1e-3):
    """与 closed_form_kernel.picard_propagate 完全相同的迭代（用于取回 alpha_knots 以采样
    全程轨迹）。返回 (u, s, residual, converged, alpha_knots)。"""
    t = torch.as_tensor(t_knots, dtype=torch.float32, device=u0.device).reshape(-1)
    K = t.numel() - 1
    ds0 = beta * u0 - gamma * s0
    s_knots = s0.unsqueeze(-1) + ds0.unsqueeze(-1) * t.reshape(1, 1, K + 1)
    u, s = u0, s0
    residual, converged = float("inf"), False
    alpha_knots = None
    for _ in range(max_iter):
        alpha_knots = alpha_nodes(W_csr, s_knots, bias)
        u_k, s_k = u0, s0
        s_new = [s0]
        for k in range(K):
            u_k, s_k = propagate_closed_form(
                u_k, s_k, alpha_knots[..., k:k + 2], t[k:k + 2], beta, gamma)
            s_new.append(s_k)
        s_new_knots = torch.stack(s_new, dim=-1)
        num = (s_new_knots - s_knots).norm(dim=(0, 1))
        den = s_knots.norm(dim=(0, 1)).clamp_min(1e-8)
        residual = float((num / den).max())
        s_knots = s_new_knots
        u, s = u_k, s_k
        if residual < tol:
            converged = True
            break
    return u, s, residual, converged, alpha_knots


def feedback_gene_indices(W_csr):
    """识别反馈边涉及基因：出度小（非 TF）的行 -> 出度大（TF）的列 的边，取其端点并集。"""
    Wd = W_csr.to_dense().cpu().numpy()
    out_deg = Wd.sum(axis=1)
    # synth_grn 出度二峰：反馈源=1、TF=edges_per_tf；取自适应阈值
    is_tf = out_deg >= max(2.0, 0.25 * out_deg.max())
    fb_edges = (Wd > 0) & (~is_tf)[:, None] & is_tf[None, :]
    genes = np.where(fb_edges.any(axis=0) | fb_edges.any(axis=1))[0]
    return genes, int(fb_edges.sum())


def compute_alpha_breaks(sol, W_csr, bias, N, G, t_span, n_grid=201):
    """从参考轨迹检测 softplus 膝点（z=0）与 clamp 折点（z=50）的穿越时刻。"""
    ts = np.linspace(t_span[0], t_span[1], n_grid)
    y = sol.sol(ts)
    s = y[N * G:].reshape(N, G, -1)[:, :, :]           # (N,G,T)
    Wn = W_csr.to_dense().cpu().numpy().astype(np.float64)
    bn = bias.cpu().numpy().astype(np.float64)
    z = np.einsum("ij,njt->nit", Wn, s) + bn[:, None]  # (N,G,T)，z_i = Σ_j W[i,j] s_j
    breaks = set()
    for level in (0.0, 50.0):
        f = z - level
        sign = np.sign(f)
        for t_idx in range(n_grid - 1):
            cross = (sign[:, :, t_idx] * sign[:, :, t_idx + 1]) < 0
            if not cross.any():
                continue
            f0 = f[:, :, t_idx][cross]
            f1 = f[:, :, t_idx + 1][cross]
            frac = f0 / (f0 - f1)
            for fr in frac:
                breaks.add(float(ts[t_idx] + fr * (ts[t_idx + 1] - ts[t_idx])))
    return sorted(b for b in breaks if t_span[0] < b < t_span[1])


def evaluate(u_end, s_end, u_traj, s_traj, ref, beta, gamma, fb_genes):
    """指标：trajectory L2 rel error、velocity cosine median、反馈边子集统计。"""
    ref_u = torch.from_numpy(np.asarray(ref["u"], dtype=np.float32)).to(u_end.device)
    ref_s = torch.from_numpy(np.asarray(ref["s"], dtype=np.float32)).to(u_end.device)
    pred = torch.cat([u_traj.reshape(-1), s_traj.reshape(-1)])
    truth = torch.cat([ref_u.reshape(-1), ref_s.reshape(-1)])
    traj_l2 = rel_l2(pred.cpu().numpy(), truth.cpu().numpy())

    v_pred = beta * u_end - gamma * s_end
    v_ref = beta * ref_u[-1] - gamma * ref_s[-1]
    cos = velocity_cosine(v_pred, v_ref)
    out = {
        "trajectory_L2_rel_error": traj_l2,
        "trajectory_L2_rel_error_u": rel_l2(u_traj.reshape(-1).cpu().numpy(),
                                            ref_u.reshape(-1).cpu().numpy()),
        "trajectory_L2_rel_error_s": rel_l2(s_traj.reshape(-1).cpu().numpy(),
                                            ref_s.reshape(-1).cpu().numpy()),
        "velocity_cosine_median": float(cos.median()),
        "final_state_L2_rel_error": rel_l2(
            torch.cat([u_end.reshape(-1), s_end.reshape(-1)]).cpu().numpy(),
            torch.cat([ref_u[-1].reshape(-1), ref_s[-1].reshape(-1)]).cpu().numpy()),
    }
    if len(fb_genes) > 0:
        idx = torch.as_tensor(fb_genes, device=u_end.device)
        cos_fb = velocity_cosine(v_pred[:, idx], v_ref[:, idx])
        pred_fb = torch.cat([u_traj[:, :, idx].reshape(-1), s_traj[:, :, idx].reshape(-1)])
        truth_fb = torch.cat([ref_u[:, :, idx].reshape(-1), ref_s[:, :, idx].reshape(-1)])
        out["feedback_gene_metrics"] = {
            "n_feedback_genes": int(len(fb_genes)),
            "velocity_cosine_median": float(cos_fb.median()),
            "trajectory_L2_rel_error": rel_l2(pred_fb.cpu().numpy(), truth_fb.cpu().numpy()),
        }
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="Phase 1 A0 闭式传播验证 + E3 分段精度扫描")
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--K", type=str, default="1,2,4,8,10,20,40")
    ap.add_argument("--small", action="store_true", help="快速自检档 (G=64,N=16)")
    ap.add_argument("--out-dir", type=str, default="logs")
    ap.add_argument("--regvelo-artifact", type=str, default=None,
                    help="预留接口：真实 RegVelo 输出 npz（当前版本未使用）")
    ap.add_argument("--picard-max-iter", type=int, default=30,
                    help="Picard 最大迭代（kernel 签名默认 3；验证档放宽以考察自洽极限，"
                         "实际迭代由 tol 提前截断并记录 residual/converged）")
    ap.add_argument("--picard-tol", type=float, default=1e-3,
                    help="与 select_engine 的 picard_residual 阈值一致")
    args = ap.parse_args()

    if args.small:
        args.G, args.N = 64, 16
    K_list = [int(x) for x in args.K.split(",")]

    log = JsonLog("closed_form_validation", out_dir=args.out_dir)
    log.record("config", {"G": args.G, "N": args.N, "K_list": K_list,
                          "t_span": T_SPAN, "n_times": N_TIMES,
                          "picard_max_iter": args.picard_max_iter,
                          "picard_tol": args.picard_tol,
                          "regvelo_artifact": args.regvelo_artifact})

    G, N = args.G, args.N
    W_csr = make_csr(synth_grn(G, n_tf=11, n_feedback=6, seed=0))
    beta, gamma, bias = synth_params(G, seed=1)
    beta, gamma, bias = beta.to(DEVICE), gamma.to(DEVICE), bias.to(DEVICE)
    u0 = torch.zeros(N, G, device=DEVICE)
    s0 = torch.zeros(N, G, device=DEVICE)

    # ---- 参考系：common.simulate_reference（DOP853, rtol=1e-9）
    ref = simulate_reference(u0, s0, W_csr, bias, beta, gamma,
                             t_span=T_SPAN, n_times=N_TIMES)
    t_eval = np.asarray(ref["t"], dtype=np.float64)
    # dense-output 参考（同一 rhs），用于节点处 oracle-α 与折点检测
    sol = reference_dense(u0, s0, W_csr, bias, beta, gamma, T_SPAN)
    # 一致性 sanity：dense 采样 vs simulate_reference
    s_chk = sample_s_at(sol, t_eval, N, G).cpu().numpy()
    log.record("reference_selfcheck_rel_l2",
               rel_l2(s_chk, np.asarray(ref["s"], dtype=np.float32)))

    fb_genes, n_fb_edges = feedback_gene_indices(W_csr)
    log.record("feedback_edges_detected", n_fb_edges)
    log.record("feedback_genes", fb_genes.tolist())

    alpha_breaks = compute_alpha_breaks(sol, W_csr, bias, N, G, T_SPAN)
    log.record("n_alpha_breaks", len(alpha_breaks))

    results = {}
    for K in K_list:
        for strategy in ("uniform", "adaptive"):
            knots = segment_knots(T_SPAN[1], K, strategy=strategy,
                                  alpha_breaks=alpha_breaks)
            key = f"K={K}/{strategy}"
            entry = {}
            # ---------- oracle-α（参考轨迹节点真值）
            s_ref_knots = sample_s_at(sol, knots.cpu().numpy(), N, G)  # (K+1,N,G)
            s_ref_knots = s_ref_knots.permute(1, 2, 0).contiguous()    # (N,G,K+1)
            alpha_oracle = alpha_nodes(W_csr, s_ref_knots, bias)
            u_o, s_o = propagate_closed_form(u0, s0, alpha_oracle, knots, beta, gamma)
            u_tr, s_tr = sample_trajectory(u0, s0, alpha_oracle, knots, t_eval,
                                           beta, gamma)
            entry["oracle"] = evaluate(u_o, s_o, u_tr, s_tr, ref, beta, gamma, fb_genes)

            # ---------- picard 自洽
            u_p, s_p, res_p, conv_p = picard_propagate(
                u0, s0, W_csr, bias, beta, gamma, knots,
                max_iter=args.picard_max_iter, tol=args.picard_tol)
            u_p2, s_p2, res_p2, conv_p2, alpha_pic = picard_with_knots(
                u0, s0, W_csr, bias, beta, gamma, knots,
                max_iter=args.picard_max_iter, tol=args.picard_tol)
            assert res_p == res_p2 and conv_p == conv_p2
            assert torch.allclose(u_p, u_p2, atol=1e-6) and torch.allclose(s_p, s_p2, atol=1e-6)
            u_trp, s_trp = sample_trajectory(u0, s0, alpha_pic, knots, t_eval,
                                             beta, gamma)
            entry["picard"] = evaluate(u_p, s_p, u_trp, s_trp, ref, beta, gamma, fb_genes)
            entry["picard"]["residual"] = res_p
            entry["picard"]["converged"] = bool(conv_p)
            entry["knots"] = knots.cpu().numpy().tolist()
            results[key] = entry
            print(f"[{key}] oracle L2={entry['oracle']['trajectory_L2_rel_error']:.3e} "
                  f"cos={entry['oracle']['velocity_cosine_median']:.4f} | "
                  f"picard L2={entry['picard']['trajectory_L2_rel_error']:.3e} "
                  f"cos={entry['picard']['velocity_cosine_median']:.4f} "
                  f"res={res_p:.2e} conv={conv_p}")

    log.record("results", results)

    # ---- 验收：K=10 oracle（任一策略达标即过），否则 K=20，否则降级建议
    def _accept(K):
        for strategy in ("uniform", "adaptive"):
            o = results.get(f"K={K}/{strategy}", {}).get("oracle")
            if o and o["velocity_cosine_median"] > ACCEPT_COSINE and \
                    o["trajectory_L2_rel_error"] < ACCEPT_L2:
                return True
        return False

    acceptance_pass = False
    K_used = None
    recommended_engine = "closed_form_foh"
    for K_try in (10, 20):
        if K_try in K_list and _accept(K_try):
            acceptance_pass, K_used = True, K_try
            break
    if not acceptance_pass:
        recommended_engine = "krylov_expmv"
    log.record("acceptance", {
        "criteria": f"oracle-α: cosine_median>{ACCEPT_COSINE} 且 trajectory_L2<{ACCEPT_L2}（K=10，否则 K=20）",
        "pass": acceptance_pass, "K_used": K_used,
        "K10_oracle": {s: results.get(f"K=10/{s}", {}).get("oracle") for s in ("uniform", "adaptive")},
        "K20_oracle": {s: results.get(f"K=20/{s}", {}).get("oracle") for s in ("uniform", "adaptive")},
    })
    log.record("recommended_engine", recommended_engine)

    # ---- K=40 与 DOP853 双重参照
    if "K=40/uniform" in results:
        log.record("double_reference", {
            "K40_oracle_uniform_vs_DOP853": results["K=40/uniform"]["oracle"],
        })

    # ---- E3：Picard 档 + 反馈边统计
    picard_all_converged = all(
        results[k]["picard"]["converged"] for k in results)
    p10 = results.get("K=10/uniform", {}).get("picard", {})
    fb10 = p10.get("feedback_gene_metrics", {})
    e3_pass = bool(
        picard_all_converged
        and p10.get("velocity_cosine_median", 0.0) > ACCEPT_COSINE
        and fb10.get("velocity_cosine_median", 0.0) > ACCEPT_COSINE)
    log.record("E3_pass", e3_pass)
    log.record("E3_criteria",
               "全部 K×策略的 Picard 收敛(residual<tol) 且 K=10/uniform Picard 档 "
               "全局与反馈边子集 velocity cosine median > 0.85"
               "（轨迹 L2 的 1e-3 验收由 acceptance/A0 判据独立给出）")
    log.record("E3_summary", {
        "picard_all_converged": picard_all_converged,
        "K10_uniform_picard": p10,
    })

    # ---- 收敛曲线图（log-y）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    styles = {("uniform", "oracle"): "o-", ("uniform", "picard"): "s--",
              ("adaptive", "oracle"): "^-", ("adaptive", "picard"): "d--"}
    for (strategy, mode), sty in styles.items():
        xs, l2s, coss = [], [], []
        for K in K_list:
            e = results.get(f"K={K}/{strategy}", {}).get(mode)
            if e:
                xs.append(K)
                l2s.append(e["trajectory_L2_rel_error"])
                coss.append(e["velocity_cosine_median"])
        axes[0].plot(xs, l2s, sty, label=f"{strategy}/{mode}")
        axes[1].plot(xs, coss, sty, label=f"{strategy}/{mode}")
    axes[0].axhline(ACCEPT_L2, color="r", ls=":", label="threshold 1e-3")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("K (segments)")
    axes[0].set_ylabel("trajectory L2 rel error (log)")
    axes[0].set_title("E3 segment accuracy sweep vs DOP853")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].axhline(ACCEPT_COSINE, color="r", ls=":", label="threshold 0.85")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("K (segments)")
    axes[1].set_ylabel("velocity cosine median")
    axes[1].set_title("velocity cosine at t_end")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "closed_form_convergence.png")
    fig.savefig(png_path, dpi=120)
    log.record("convergence_plot", png_path)

    path = log.finalize({"acceptance_pass": acceptance_pass})
    print(f"JSON log: {path}")
    print(f"acceptance_pass={acceptance_pass} (K_used={K_used}), "
          f"recommended_engine={recommended_engine}, E3_pass={e3_pass}")


if __name__ == "__main__":
    main()
