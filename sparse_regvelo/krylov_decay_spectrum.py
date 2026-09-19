"""krylov_decay_spectrum.py — E1 Krylov 衰减谱预实验（SPEC §3.4）。

对原始非对称 W（合成星形+反馈，G=512 默认；--large 时 G=17714/nnz=194843 仅 CUDA）：
随机 v（seed 固定，r=8 个向量取中位），m ∈ {10,20,50,100,200}，
用 arnoldi_expmv 计算 exp(tA)v 的相对残余。
判据：m=200 时残余 < 10% → E1_verdict = "krylov_retained"，
否则 "krylov_retired_use_full_sparse_expmv"。
输出：JSON + logs/krylov_decay.png。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common import DEVICE, JsonLog, make_csr, synth_grn
from closed_form_kernel import arnoldi_expmv

M_LIST_DEFAULT = [10, 20, 50, 100, 200]


def main():
    ap = argparse.ArgumentParser(description="E1 Krylov 衰减谱预实验")
    ap.add_argument("--G", type=int, default=512)
    ap.add_argument("--r", type=int, default=8, help="随机向量个数（取中位）")
    ap.add_argument("--m", type=str, default=",".join(str(x) for x in M_LIST_DEFAULT))
    ap.add_argument("--t", type=float, default=1.0, help="exp(tA) 的时间尺度")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--large", action="store_true",
                    help="G=17714/nnz=194843 全量档（仅 CUDA；CPU 沙箱拒绝运行）")
    ap.add_argument("--small", action="store_true", help="快速自检档 (G=256)")
    ap.add_argument("--out-dir", type=str, default="logs")
    args = ap.parse_args()

    if args.large:
        args.G = 17714
        if DEVICE != "cuda":
            raise SystemExit("--large 仅 CUDA 可运行（CPU 沙箱 OOM 风险），退出")
    if args.small:
        args.G = 256
    m_list = [int(x) for x in args.m.split(",")]

    log = JsonLog("krylov_decay_spectrum", out_dir=args.out_dir)
    log.record("config", {"G": args.G, "r": args.r, "m_list": m_list,
                          "t": args.t, "seed": args.seed, "large": args.large})

    W_csr = make_csr(synth_grn(args.G, n_tf=11, n_feedback=6, seed=0))
    log.record("nnz", int(W_csr._nnz()))

    # 结构诊断：合成星形 W 的有效秩与随机向量的有效 Krylov 维数
    # （解释残余为 0 的 happy breakdown；仅诊断，不改判据）
    Wd = W_csr.to_dense().cpu().numpy().astype(np.float64)
    eff_rank = int(np.linalg.matrix_rank(Wd))
    log.record("effective_rank_W", eff_rank)

    # r 个随机向量（seed 固定），单位范数 → residual_est 即相对残余
    g = torch.Generator().manual_seed(args.seed)
    V = torch.randn(args.G, args.r, generator=g)
    V = V / V.norm(dim=0, keepdim=True)

    # 每个向量的有效 Krylov 维数（SVD 诊断，G 大时跳过）
    eff_dims = []
    if args.G <= 4096:
        Vn = V.numpy()
        for j in range(args.r):
            Kry = [Vn[:, j]]
            for _ in range(min(64, args.G - 1)):
                Kry.append(Wd @ Kry[-1])
            sv = np.linalg.svd(np.array(Kry), compute_uv=False)
            eff_dims.append(int((sv > 1e-12 * sv[0]).sum()))
        log.record("effective_krylov_dim_per_vector", eff_dims)

    per_m = {}
    for m in m_list:
        res = np.zeros(args.r)
        wnorm = np.zeros(args.r)
        for j in range(args.r):
            w, res_est = arnoldi_expmv(W_csr, V[:, j].to(W_csr.device), args.t, m=m)
            nw = float(w.norm())
            # 相对残余：相对解范数（v 已单位化，同时记录 raw）
            res[j] = res_est / max(nw, 1e-30)
            wnorm[j] = nw
        per_m[str(m)] = {
            "rel_residual_median": float(np.median(res)),
            "rel_residual_all": res.tolist(),
            "rel_residual_raw_median": float(np.median(res * wnorm)),
            "w_norm_median": float(np.median(wnorm)),
        }
        print(f"m={m:4d}  rel_residual median={np.median(res):.4e}  "
              f"(all={np.round(res, 4).tolist()})")
    log.record("results", per_m)

    m200 = per_m.get("200") or per_m[str(m_list[-1])]
    retained = m200["rel_residual_median"] < 0.10
    verdict = "krylov_retained" if retained else "krylov_retired_use_full_sparse_expmv"
    log.record("E1_verdict", verdict)
    log.record("E1_criteria", "m=200 时相对残余中位 < 10% → krylov_retained")
    log.record("krylov_retained", bool(retained))
    if eff_dims:
        log.record("note",
                   f"合成星形+反馈 W 有效秩={eff_rank}，随机向量有效 Krylov 维数"
                   f"中位={int(np.median(eff_dims))}；m≥该维数即 happy breakdown（残余精确为 0），"
                   "残余曲线退化为零是低秩结构的如实反映，verdict 仍按 m=200 判据给出")

    # ---- 衰减曲线图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    floor = 1e-16  # log 轴下 0 残余的下截显示
    meds = [max(per_m[str(m)]["rel_residual_median"], floor) for m in m_list]
    ax.plot(m_list, meds, "o-", label="median rel residual (0 clipped)")
    for j in range(args.r):
        ax.plot(m_list, [max(per_m[str(m)]["rel_residual_all"][j], floor) for m in m_list],
                ".", color="gray", alpha=0.4, markersize=4)
    ax.axhline(0.10, color="r", ls=":", label="10% threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Krylov dimension m")
    ax.set_ylabel("relative residual of exp(tA)v (log)")
    ax.set_title(f"E1 Krylov decay spectrum (G={args.G}, t={args.t}) — {verdict}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "krylov_decay.png")
    fig.savefig(png_path, dpi=120)
    log.record("decay_plot", png_path)

    path = log.finalize()
    print(f"JSON log: {path}")
    print(f"E1_verdict={verdict} (m=200 median rel residual="
          f"{m200['rel_residual_median']:.4e})")


if __name__ == "__main__":
    main()
