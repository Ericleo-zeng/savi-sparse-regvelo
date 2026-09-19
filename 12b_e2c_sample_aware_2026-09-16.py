#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12b_e2c_sample_aware.py — E2c 的 12-样本层级敏感性检查（CHECK C8 后半）。

重算 E2c 完全相同的 per-cell rel（faithful ckpt；prod K10/tol1e-3/iter10
vs ref K20/tol1e-6/iter50），然后：
  1) 校验 top-1% 逐状态计数与 E2c_result.json 一致（±1 容差，CUDA 归约漂移）；
  2) 输出 top-1% 的按样本分布（每样本 top 数、C1q top 数）；
  3) C1q 态内 permutation：top 成员在 942 个 C1q 细胞内随机化 B 次，
     统计 "19 个 top 槽位中单一样本最大占比" 的经验 p（浓度检验）；
  4) 复算 cell-level Fisher 与正文 p=1.0e-4/OR=3.3 交叉核对。

产物：resid_rel.npy（逐细胞残差，供后续复用）、E2c_sample_aware.json。
"""
from __future__ import annotations
import argparse, datetime, json, os, sys
import numpy as np
import config as C
sys.path.insert(0, C.PROJECT_DIR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/V5_rate_max/ckpt_adaptive")
    ap.add_argument("--ms", default="results/_subset/Ms_sub_n5000.npy")
    ap.add_argument("--mu", default="results/_subset/Mu_sub_n5000.npy")
    ap.add_argument("--labels", default="results/_subset/state_labels_n5000.npy")
    ap.add_argument("--samples", default="results/_subset/sample_ids_n5000.npy")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default="results/E2c_tail_grounding")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    import torch
    from run_l3_only import load_l2_checkpoint
    from common import make_csr
    from closed_form_kernel import picard_propagate, segment_knots
    from sparse_regvelo_engine import PICARD_TOL
    from scipy.stats import fisher_exact

    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_c, gamma_c, t_c, *_, W, meta = load_l2_checkpoint(args.checkpoint)
    W_csr = make_csr(W, device=device)
    beta = torch.from_numpy(beta_c).to(device)
    gamma = torch.from_numpy(gamma_c).to(device)
    bias = torch.from_numpy(np.load(os.path.join(args.checkpoint, "bias.npy"))).to(device)
    t_max = float(t_c.max()) or 1.0

    labels  = np.load(args.labels, allow_pickle=True)
    samples = np.load(args.samples, allow_pickle=True)
    s0 = np.asarray(np.load(args.ms, mmap_mode="r"), dtype=np.float32)
    u0 = np.asarray(np.load(args.mu, mmap_mode="r"), dtype=np.float32)
    N = s0.shape[0]
    assert labels.shape[0] == N and samples.shape[0] == N

    def propagate(K, tol, mi):
        knots = segment_knots(t_max, K)
        u = np.empty_like(u0); s = np.empty_like(s0)
        for st in range(0, N, args.bs):
            en = min(st + args.bs, N)
            ub = torch.from_numpy(u0[st:en]).to(device)
            sb = torch.from_numpy(s0[st:en]).to(device)
            u_b, s_b, _, _ = picard_propagate(ub, sb, W_csr, bias, beta, gamma,
                                              knots, max_iter=mi, tol=tol)
            u[st:en] = u_b.cpu().numpy(); s[st:en] = s_b.cpu().numpy()
            del ub, sb, u_b, s_b
            if device == "cuda": torch.cuda.empty_cache()
        return beta_c[None, :] * u - gamma_c[None, :] * s

    v_ref  = propagate(20, 1e-6, 50)
    v_prod = propagate(10, PICARD_TOL, 10)
    rel = C.rel_l2_per_cell(v_prod, v_ref)              # (N,)
    np.save(os.path.join(args.out, "resid_rel.npy"), rel)

    k_top = max(int(N * args.top_frac), 1)
    top_idx = np.argsort(rel)[-k_top:]

    # 1) 与 E2c_result.json 交叉核对
    with open(os.path.join(args.out, "E2c_result.json")) as f:
        e2c = json.load(f)
    print("== per-state top-1% counts: rerun vs E2c_result.json ==")
    state_check = {}
    for st in sorted(set(labels)):
        n_new = int(((labels == st)[top_idx]).sum())
        n_old = e2c["by_state"][st]["top1pct_count"]
        state_check[st] = {"rerun": n_new, "recorded": n_old}
        print(f"  {st:>22s}: rerun={n_new:3d} recorded={n_old:3d}")

    # 2) 按样本分布
    print("\n== top-1% by sample ==")
    smp_rows = []
    for smp in sorted(set(samples)):
        m_top = samples[top_idx] == smp
        n_c1q_top = int((labels[top_idx][m_top] == "C1q_inflammatory").sum())
        row = {"sample": str(smp), "n_cells": int((samples == smp).sum()),
               "top1pct_n": int(m_top.sum()), "c1q_top_n": n_c1q_top}
        smp_rows.append(row)
        print(f"  {str(smp):>16s} | cells={row['n_cells']:4d} | top1%={row['top1pct_n']:3d} | C1q_top={n_c1q_top:3d}")

    # 3) C1q 态内浓度 permutation
    c1q = np.where(labels == "C1q_inflammatory")[0]
    obs_top_c1q = top_idx[np.isin(top_idx, c1q)]
    n_top_c1q = len(obs_top_c1q)
    obs_max_share = float((samples[obs_top_c1q] == samples[obs_top_c1q][0]).mean()) \
        if n_top_c1q else 0.0
    # 更直接：单一样本最大占比
    if n_top_c1q:
        vals, cnts = np.unique(samples[obs_top_c1q], return_counts=True)
        obs_max_share = float(cnts.max() / n_top_c1q)
        n_distinct = int(len(vals))
    else:
        n_distinct = 0
    rng = np.random.default_rng(args.seed)
    perm_max = np.empty(args.perm)
    for b in range(args.perm):
        fake = rng.choice(c1q, size=n_top_c1q, replace=False)
        _, c = np.unique(samples[fake], return_counts=True)
        perm_max[b] = c.max() / n_top_c1q
    p_conc = float((perm_max >= obs_max_share).mean())
    print(f"\nC1q top cells: {n_top_c1q} | distinct samples={n_distinct}/12 | "
          f"max single-sample share={obs_max_share:.2f} | concentration p={p_conc:.4f} (B={args.perm})")

    # 4) cell-level Fisher 复算
    n_c1q = len(c1q)
    tab = [[n_top_c1q, k_top - n_top_c1q],
           [n_c1q - n_top_c1q, N - k_top - (n_c1q - n_top_c1q)]]
    OR, p_f = fisher_exact(tab)
    print(f"Fisher rerun: table={tab} OR={OR:.4f} p={p_f:.3e}")

    out = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "cmd": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
           "state_check": state_check,
           "by_sample": smp_rows,
           "c1q_top_n": int(n_top_c1q),
           "c1q_top_distinct_samples": n_distinct,
           "c1q_top_max_sample_share": obs_max_share,
           "concentration_perm_p": p_conc, "perm_B": args.perm,
           "fisher_table": tab, "fisher_OR": float(OR), "fisher_p": float(p_f)}
    with open(os.path.join(args.out, "E2c_sample_aware.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[12b] -> {os.path.join(args.out, 'E2c_sample_aware.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
