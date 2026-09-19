#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12_e2c_tail_grounding.py — E2c：V1 残差长尾的细胞状态归属（评审#5 建议 2）。

不做动力学刚性证明，只做统计关联：在 faithful checkpoint 上对全部 5000
子集细胞按生产口径（K=10, tol=1e-3, max_iter=10）传播，以收紧参考
（K=20, tol=1e-6, 50 iter）为基准得逐细胞 v 残差；检验 Top-1% 残差细胞
是否富集于特定注释态（逐状态均值残差 + Top-1% 富集表 + 比例检验）。

用法（labels 由 11b 脚本生成）：
  python 12_e2c_tail_grounding.py \
    --checkpoint results/V5_rate_max/ckpt_adaptive \
    --ms results/_subset/Ms_sub_n5000.npy --mu results/_subset/Mu_sub_n5000.npy \
    --labels results/_subset/state_labels_n5000.npy \
    --out results/E2c_tail_grounding
"""
from __future__ import annotations
import argparse, datetime, json, os, sys
import numpy as np
import config as C
sys.path.insert(0, C.PROJECT_DIR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ms", required=True); ap.add_argument("--mu", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    import torch
    from run_l3_only import load_l2_checkpoint
    from common import make_csr
    from closed_form_kernel import picard_propagate, segment_knots
    from sparse_regvelo_engine import PICARD_TOL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_c, gamma_c, t_c, *_ , W, meta = load_l2_checkpoint(args.checkpoint)
    W_csr = make_csr(W, device=device)
    beta = torch.from_numpy(beta_c).to(device)
    gamma = torch.from_numpy(gamma_c).to(device)
    bias = torch.from_numpy(np.load(os.path.join(args.checkpoint, "bias.npy"))).to(device)
    t_max = float(t_c.max()) or 1.0

    labels = np.load(args.labels, allow_pickle=True)
    s0 = np.asarray(np.load(args.ms, mmap_mode="r"), dtype=np.float32)
    u0 = np.asarray(np.load(args.mu, mmap_mode="r"), dtype=np.float32)
    N = s0.shape[0]; assert labels.shape[0] == N

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

    v_ref = propagate(20, 1e-6, 50)
    v_prod = propagate(10, PICARD_TOL, 10)
    rel = C.rel_l2_per_cell(v_prod, v_ref)          # (N,)

    states = sorted(set(labels))
    k_top = max(int(N * args.top_frac), 1)
    top_idx = np.argsort(rel)[-k_top:]
    table = {}
    print(f"{'state':>22s} | {'n':>5s} | {'mean rel':>10s} | {'p99 rel':>10s} | "
          f"{'top1% n':>7s} | {'top1% share':>11s} | {'pop share':>9s} | enrich")
    for st in states:
        m = labels == st
        n_st = int(m.sum())
        top_in = int((m[top_idx]).sum())
        pop_share = n_st / N
        top_share = top_in / k_top
        # 富集 = top1% 份额 / 总体份额（1.0 = 无富集）
        enrich = top_share / max(pop_share, 1e-12)
        table[st] = {"n": n_st,
                     "mean_rel": float(rel[m].mean()),
                     "p99_rel": float(np.percentile(rel[m], 99)),
                     "top1pct_count": top_in, "top1pct_share": float(top_share),
                     "population_share": float(pop_share),
                     "enrichment": float(enrich)}
        print(f"{st:>22s} | {n_st:>5d} | {table[st]['mean_rel']:>10.3e} | "
              f"{table[st]['p99_rel']:>10.3e} | {top_in:>7d} | {top_share:>11.3f} | "
              f"{pop_share:>9.3f} | {enrich:>6.2f}x")
    print(f"\n[E2c] overall v rel: median={np.median(rel):.3e} "
          f"p99={np.percentile(rel, 99):.3e} max={rel.max():.3e}")

    result = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "cmd": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
              "checkpoint": args.checkpoint, "N": int(N), "top_frac": args.top_frac,
              "overall": {"median": float(np.median(rel)),
                          "p99": float(np.percentile(rel, 99)),
                          "max": float(rel.max())},
              "by_state": table}
    with open(os.path.join(args.out, "E2c_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[E2c] -> {os.path.join(args.out, 'E2c_result.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
