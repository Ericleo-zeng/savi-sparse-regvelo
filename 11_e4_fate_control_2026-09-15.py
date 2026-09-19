#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""11_e4_fate_control.py — E4：V7 的 fate 级对照（field-level → conclusion-level）。

问题：剔除 57 周期基因使速度场 cosine 中位跌至 0.68–0.80——这个 field 级
扰动是否传导为 fate 级失稳？两种结果都有论文价值：
  高一致 → "field-level 扰动不传导为 fate-level 失稳"；
  低一致 → "input curation 可定性改变 fate landscape"（警示卖点）。

设计（全部子集规模，固定种子，确定性）：
  每个 --l3 arm 用【自己的】v.npy / s_end.npy 构造 velocity-informed 核：
    kNN(k=30) 取自 10 维 latent；邻居位移取该 arm 的 s_end（基因空间）；
    c_ij = cos(v_i, s_j - s_i)；w_ij = softmax(c_ij / σ_i)，
    σ_i = median(|c_i·|)（CellRank VelocityKernel 口径）；
    T = 0.8·T_v + 0.2·T_c（T_c = latent kNN 高斯+max 对称化的扩散核）；
    target = 注释态的 latent 质心最近 r=30 代表（sfate 惯例）；
    (I−Q)X = R 用 scipy spsolve(f64) 精确解（5000 规模可承受）。
  对比：逐细胞 fate 向量 Spearman、top-state 一致率、类水平 L∞ 漂移、
        样本级 fate 矩阵（落 TSV，供符号分析）。

用法（先按下方 helper 提取 labels）：
  python 11_e4_fate_control.py \
    --l3-full  results/V5_rate_max/l3_adaptive \
    --l3-nocycle results/V7_no_cycling/l3_v7 \
    --l3-nocycle-norenorm results/V7_no_cycling/l3_v7a \
    --latent results/_subset/X_sub_n5000.npy \
    --labels results/_subset/state_labels_n5000.npy \
    --samples results/_subset/sample_ids_n5000.npy \
    --out results/E4_fate_control
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

import config as C

sys.path.insert(0, C.PROJECT_DIR)


def _require_complete_l3(d: str) -> None:
    if not os.path.exists(os.path.join(d, "l3_metadata.json")):
        raise SystemExit(f"[FATAL] {d} 缺少 l3_metadata.json（半成品，拒绝）")


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行 Spearman（对 6 维 fate 向量，N×6 输入，输出 (N,)）。"""
    def rank_rows(x):
        order = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)
        return order
    ra, rb = rank_rows(a), rank_rows(b)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    den[den == 0] = np.nan
    return num / den


def build_kernel(latent, s_end, v, k=30, w_v=0.8):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    _, nbr = nn.kneighbors(latent)          # (N, k+1) 含自身
    nbr = nbr[:, 1:]                        # 去自身

    N = latent.shape[0]
    rows, cols = np.repeat(np.arange(N), k), nbr.ravel()

    # 速度核：c_ij = cos(v_i, s_j - s_i)，σ_i = median(|c_i·|)
    diff = s_end[cols] - s_end[rows]        # (N*k, G)
    vv = np.repeat(v, k, axis=0)
    num = np.einsum("ij,ij->i", vv, diff)
    den = np.linalg.norm(vv, axis=1) * np.linalg.norm(diff, axis=1)
    c = num / np.maximum(den, 1e-12)        # (N*k,)
    sig = np.abs(c.reshape(N, k))
    sig = np.maximum(np.median(sig, axis=1), 1e-8)   # (N,)
    expo = c / np.repeat(sig, k)
    expo -= expo.max()                      # softmax 数值稳定（全局平移不改比例关系）
    w = np.exp(expo)
    w = w / np.repeat(w.reshape(N, k).sum(axis=1), k)
    Tv = __import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(
        (w, (rows, cols)), shape=(N, N))

    # 扩散核：高斯 + max 对称化 + 行随机
    import scipy.sparse as sp
    d2 = ((latent[rows] - latent[cols]) ** 2).sum(axis=1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    wc = np.exp(-d2 / max(med, 1e-12))
    Tc = sp.csr_matrix((wc, (rows, cols)), shape=(N, N))
    Tc = Tc.maximum(Tc.T)
    Tc = sp.diags(1.0 / np.maximum(Tc.sum(axis=1).A.ravel(), 1e-12)) @ Tc

    T = w_v * Tv + (1.0 - w_v) * Tc
    T = sp.diags(1.0 / np.maximum(T.sum(axis=1).A.ravel(), 1e-12)) @ T
    return T.tocsr()


def absorbing_reps(latent, labels, r=30):
    reps, state_names = [], []
    for st in sorted(set(labels)):
        idx = np.where(labels == st)[0]
        if len(idx) < r:
            raise SystemExit(f"[FATAL] 状态 {st} 仅 {len(idx)} 细胞 < r={r}")
        cen = latent[idx].mean(axis=0)
        order = np.argsort(((latent[idx] - cen) ** 2).sum(axis=1))[:r]
        reps.append(idx[order])
        state_names.append(st)
    return np.concatenate(reps), state_names


def solve_fate(T, reps, state_names, labels):
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    N = T.shape[0]
    rep_set = set(reps.tolist())
    trans = np.array([i for i in range(N) if i not in rep_set])
    Rfull = np.zeros((N, len(state_names)))
    for si, st in enumerate(state_names):
        Rfull[reps[labels[reps] == st], si] = 1.0
    Q = T[trans][:, trans].astype(np.float64)
    Rm = T[trans][:, reps].astype(np.float64) @ Rfull[reps]
    A = sp.eye(len(trans), format="csc") - Q.tocsc()
    X = np.zeros((len(trans), len(state_names)))
    for j in range(len(state_names)):
        X[:, j] = spsolve(A, Rm[:, j])
    fate = np.zeros((N, len(state_names)))
    fate[trans] = X
    fate[reps] = Rfull[reps]
    return fate / np.maximum(fate.sum(axis=1, keepdims=True), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 fate-level control")
    ap.add_argument("--l3-full", required=True)
    ap.add_argument("--l3-nocycle", required=True)
    ap.add_argument("--l3-nocycle-norenorm", default=None)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--labels", required=True, help="(N,) 字符串数组 .npy")
    ap.add_argument("--samples", default=None, help="(N,) 样本ID .npy（可选）")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--w-v", type=float, default=0.8)
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--with-diffusion-only", action="store_true",
                    help="附加纯扩散核基线（用 full arm 的 latent）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    latent = np.load(args.latent).astype(np.float32)
    labels = np.load(args.labels, allow_pickle=True)
    samples = np.load(args.samples, allow_pickle=True) if args.samples else None
    N = latent.shape[0]
    assert labels.shape[0] == N

    arms = [("full", args.l3_full), ("no_cycle", args.l3_nocycle)]
    if args.l3_nocycle_norenorm:
        arms.append(("no_cycle_norenorm", args.l3_nocycle_norenorm))

    result = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "cmd": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
              "k": args.k, "w_v": args.w_v, "r": args.r,
              "N": int(N), "arms": {}}
    fates, state_names = {}, None

    for name, d in arms:
        _require_complete_l3(d)
        s_end = np.asarray(np.load(os.path.join(d, "s_end.npy"), mmap_mode="r"))
        v = np.asarray(np.load(os.path.join(d, "v.npy"), mmap_mode="r"))
        T = build_kernel(latent, s_end, v, k=args.k, w_v=args.w_v)
        reps, sn = absorbing_reps(latent, labels, r=args.r)
        state_names = sn
        fates[name] = solve_fate(T, reps, sn, labels)
        result["arms"][name] = {"l3_dir": os.path.abspath(d)}
        print(f"[E4] arm {name}: kernel + solve done ({d})")

    if args.with_diffusion_only:
        import scipy.sparse as sp
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=args.k + 1).fit(latent)
        _, nbr = nn.kneighbors(latent)
        nbr = nbr[:, 1:]
        rows = np.repeat(np.arange(N), args.k)
        cols = nbr.ravel()
        d2 = ((latent[rows] - latent[cols]) ** 2).sum(axis=1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        w = np.exp(-d2 / max(med, 1e-12))
        Tc = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
        Tc = Tc.maximum(Tc.T)
        Tc = sp.diags(1.0 / np.maximum(Tc.sum(axis=1).A.ravel(), 1e-12)) @ Tc
        reps, sn = absorbing_reps(latent, labels, r=args.r)
        fates["diffusion_only"] = solve_fate(Tc.tocsr(), reps, sn, labels)
        result["arms"]["diffusion_only"] = {"baseline": True}
        print("[E4] arm diffusion_only: done")

    ref = fates["full"]
    print(f"\n[E4] states: {state_names}")
    for name, f in fates.items():
        if name == "full":
            continue
        sp_ = rowwise_spearman(ref, f)
        top_ref, top_new = ref.argmax(1), f.argmax(1)
        agree = float((top_ref == top_new).mean())
        shift = float(np.abs(ref - f).max())
        print(f"[E4] full vs {name}: fate Spearman median={np.nanmedian(sp_):.4f} "
              f"min={np.nanmin(sp_):.4f} | top-state agree={agree:.4f} | "
              f"class L∞ shift={shift:.4f}")
        result["arms"][name].update(
            {"fate_spearman_median": float(np.nanmedian(sp_)),
             "fate_spearman_min": float(np.nanmin(sp_)),
             "top_state_agreement": agree,
             "class_shift_Linf": shift})

    for name, f in fates.items():
        np.save(os.path.join(args.out, f"fate_{name}.npy"), f)
        if samples is not None:
            with open(os.path.join(args.out, f"fate_by_sample_{name}.tsv"),
                      "w", encoding="utf-8") as fh:
                fh.write("sample\t" + "\t".join(state_names) + "\n")
                for smp in sorted(set(samples)):
                    m = f[samples == smp].mean(axis=0)
                    fh.write(f"{smp}\t" + "\t".join(f"{x:.6f}" for x in m) + "\n")

    with open(os.path.join(args.out, "E4_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[E4] -> {os.path.join(args.out, 'E4_result.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
