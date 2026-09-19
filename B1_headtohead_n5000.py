#!/usr/bin/env python3
"""B1_headtohead_n5000.py — 500→5,000 细胞升级重跑判定实验。

1) 复用 results/_subset/cell_index_n5000.npy（seed=20260914）。
2) 基因空间沿用 results/B1_regvelo_reference/_inputs_hvg2k_n500/hvg_genes.txt。
3) 修复 scVelo 锚点：filter_and_normalize → moments(n_pcs=30) → velocity(mode='dynamical')，
   确认 velocity 非全 NaN 且范数非零后才计入。
4) 同场各跑一次：SAVI L2+L3（生产协议）、RegVelo GPU 200 epochs、scVelo dynamical。
5) 两两 per-cell cosine / rel-L2（median/P99/min）。
6) 写 results/B1_regvelo_reference/B1_headtohead_n5000.json，
   追加结论到 results/B1_regvelo_reference/B1_headtohead_diagnosis.md 第 5 节。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import anndata
import numpy as np
import psutil
import scipy.sparse as sp
import scvelo as scv
import torch

SPARSE_DIR = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/sparse_regvelo"
sys.path.insert(0, SPARSE_DIR)
sys.path.insert(0, "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo")

from config import cosine_per_cell, rel_l2_per_cell
from run_l3_only import run_l3_streaming

BASE = Path("/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo")
OUT_DIR = BASE / "results" / "B1_regvelo_reference"
INPUT_DIR = OUT_DIR / "_inputs_hvg2k_n5000"
RESULT_PATH = OUT_DIR / "B1_headtohead_n5000.json"
DIAG_MD = OUT_DIR / "B1_headtohead_diagnosis.md"
SUBSET_DIR = BASE / "results" / "_subset"
MAIN_W = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input/W.npz"
MAIN_GENE_LIST = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/gene_universe_used.txt"
ADATA_PATH = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad"
SEED = 20260914
N_CELLS = 5000
HVG_SOURCE = OUT_DIR / "_inputs_hvg2k_n500" / "hvg_genes.txt"

log_lines: list[str] = []


def log(msg: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)


def gpu_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return float("nan")


def process_ram_gb() -> float:
    try:
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return float("nan")


def load_hvg_genes() -> list[str]:
    genes = [x.strip() for x in HVG_SOURCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    log(f"Loaded {len(genes)} HVG genes from {HVG_SOURCE}")
    return genes


def build_inputs(hvg_genes: list[str]):
    """为 5k 子集构建 HVG2k 输入（Ms/Mu/X/W/hvg_genes.txt）。"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_genes = [x.strip() for x in Path(MAIN_GENE_LIST).read_text(encoding="utf-8").splitlines() if x.strip()]
    main_genes = list(dict.fromkeys(main_genes))
    gene_to_idx = {g: i for i, g in enumerate(main_genes)}

    missing = [g for g in hvg_genes if g not in gene_to_idx]
    if missing:
        log(f"WARNING: {len(missing)} HVG genes missing in main gene list: {missing[:10]}")
    present_genes = [g for g in hvg_genes if g in gene_to_idx]
    top_idx = np.array([gene_to_idx[g] for g in present_genes], dtype=np.int64)

    Ms = np.load(SUBSET_DIR / f"Ms_sub_n{N_CELLS}.npy")
    Mu = np.load(SUBSET_DIR / f"Mu_sub_n{N_CELLS}.npy")
    X = np.load(SUBSET_DIR / f"X_sub_n{N_CELLS}.npy")

    Ms_r = np.ascontiguousarray(Ms[:, top_idx], dtype=np.float32)
    Mu_r = np.ascontiguousarray(Mu[:, top_idx], dtype=np.float32)
    np.save(INPUT_DIR / "Ms.npy", Ms_r)
    np.save(INPUT_DIR / "Mu.npy", Mu_r)
    np.save(INPUT_DIR / "X.npy", X)

    W = sp.load_npz(MAIN_W)
    W_r = W[top_idx, :][:, top_idx].tocsr()
    W_r.eliminate_zeros()
    sp.save_npz(INPUT_DIR / "W.npz", W_r)

    gene_hash = hashlib.sha256("\n".join(present_genes).encode("utf-8")).hexdigest()[:16]
    (INPUT_DIR / "hvg_genes.txt").write_text("\n".join(present_genes), encoding="utf-8")
    (INPUT_DIR / "hvg_genes.sha256").write_text(
        hashlib.sha256("\n".join(present_genes).encode("utf-8")).hexdigest(), encoding="utf-8"
    )

    log(f"Built inputs: N={Ms_r.shape[0]}, G={Ms_r.shape[1]}, W nnz={W_r.nnz}, gene_hash={gene_hash}")
    return Ms_r.shape[1], W_r.nnz, gene_hash, present_genes


def run_savi_l2():
    log("Running SAVI L2 on HVG2k/5000...")
    l2_dir = INPUT_DIR / "l2_checkpoint"
    if l2_dir.exists():
        shutil.rmtree(l2_dir)
    cmd = [
        sys.executable,
        os.path.join(SPARSE_DIR, "sparse_regvelo_engine.py"),
        "--ms", str(INPUT_DIR / "Ms.npy"),
        "--mu", str(INPUT_DIR / "Mu.npy"),
        "--w", str(INPUT_DIR / "W.npz"),
        "--x-scvi", str(INPUT_DIR / "X.npy"),
        "--l2-only",
        "--l2-checkpoint", str(l2_dir),
        "--out-dir", str(INPUT_DIR / "l2_logs"),
        "--batch-size", "1024",
    ]
    t0 = time.time()
    mem_before = gpu_memory_mb()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rt = time.time() - t0
    mem_after = gpu_memory_mb()
    if proc.returncode != 0:
        log("L2 stdout:\n" + proc.stdout[-2000:])
        log("L2 stderr:\n" + proc.stderr[-2000:])
        raise RuntimeError(f"SAVI L2 failed: {proc.returncode}")
    log(f"SAVI L2 completed in {rt:.1f}s")
    return rt, (mem_after - mem_before) / 1024


def run_savi_l3():
    l2_dir = INPUT_DIR / "l2_checkpoint"
    out_dir = INPUT_DIR / "l3_production"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.time()
    mem_before = gpu_memory_mb()
    run_l3_streaming(
        ms_path=str(INPUT_DIR / "Ms.npy"),
        mu_path=str(INPUT_DIR / "Mu.npy"),
        checkpoint_dir=str(l2_dir),
        out_dir=str(out_dir),
        batch_size=1024,
        device="cuda",
        K=10,
    )
    rt = time.time() - t0
    mem_after = gpu_memory_mb()
    v = np.asarray(np.load(out_dir / "v.npy"), dtype=np.float32)
    log(f"SAVI L3 completed in {rt:.1f}s; v shape={v.shape}")
    return v, rt, (mem_after - mem_before) / 1024


def run_regvelo_gpu() -> tuple[np.ndarray | None, dict]:
    from regvelo import REGVELOVI

    Ms = np.load(INPUT_DIR / "Ms.npy")
    Mu = np.load(INPUT_DIR / "Mu.npy")
    adata = anndata.AnnData(X=Ms)
    adata.layers["Ms"] = Ms
    adata.layers["Mu"] = Mu
    adata.var_names = [f"g{i}" for i in range(Ms.shape[1])]
    adata.obs_names = [f"c{i}" for i in range(Ms.shape[0])]

    W_sparse = sp.load_npz(INPUT_DIR / "W.npz")
    W_tensor = torch.from_numpy(W_sparse.toarray()).to(dtype=torch.float32)
    log(f"[W conversion] dense shape={tuple(W_tensor.shape)}, nnz={W_tensor.count_nonzero().item()}")

    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    model = REGVELOVI(adata, W=W_tensor, soft_constraint=True)

    loss_log_path = INPUT_DIR / "regvelo_gpu_loss.log"

    from lightning.pytorch.callbacks import Callback

    class EpochLogger(Callback):
        def __init__(self, path: Path):
            self.path = path
            self.start = time.time()
            self.records = []

        def on_train_epoch_end(self, trainer, pl_module):
            ep = trainer.current_epoch
            if (ep + 1) % 10 == 0 or ep == 0:
                loss = None
                for k in ("train_loss", "loss", "train_loss_epoch"):
                    if k in trainer.callback_metrics:
                        loss = float(trainer.callback_metrics[k])
                        break
                rec = {"epoch": ep + 1, "wall_sec": time.time() - self.start, "loss": loss}
                self.records.append(rec)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

    torch.cuda.reset_peak_memory_stats()
    log("[RegVelo GPU] start training")
    t0 = time.time()
    info: dict = {"status": "failed", "error": None, "trace": None, "peak_gpu_gb": None, "runtime_sec": None}
    try:
        model.train(
            max_epochs=200,
            batch_size=512,
            train_size=0.9,
            early_stopping=True,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
            callbacks=[EpochLogger(loss_log_path)],
        )
    except Exception as e:
        info["error"] = str(e)
        info["trace"] = traceback.format_exc()
        info["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        log(f"[RegVelo GPU] failed: {e}; peak GPU {info['peak_gpu_gb']:.2f} GB")
        return None, info
    train_rt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    log(f"[RegVelo GPU] training done in {train_rt:.1f}s; peak GPU {peak:.2f} GB")

    t1 = time.time()
    v = model.get_velocity(adata, return_numpy=True)
    vel_rt = time.time() - t1
    v_path = INPUT_DIR / "v_regvelo.npy"
    np.save(v_path, np.asarray(v, dtype=np.float32))

    info.update({
        "status": "ran",
        "train_runtime_sec": round(train_rt, 2),
        "velocity_runtime_sec": round(vel_rt, 2),
        "total_runtime_sec": round(train_rt + vel_rt, 2),
        "peak_gpu_gb": round(peak, 3),
        "loss_log_path": str(loss_log_path),
        "velocity_path": str(v_path),
    })
    return np.asarray(v, dtype=np.float32), info


def run_scvelo_dynamical(hvg_genes: list[str]) -> tuple[np.ndarray | None, dict]:
    """运行修复后的 scVelo dynamical 锚点，返回 (velocity_array, info_dict)。"""
    log("Running scVelo dynamical anchor (fixed pipeline)...")
    cell_idx = np.load(SUBSET_DIR / "cell_index_n5000.npy")

    t0 = time.time()
    peak_ram_before = process_ram_gb()
    try:
        adata_full = anndata.read_h5ad(ADATA_PATH, backed="r")
        present_genes = [g for g in hvg_genes if g in adata_full.var_names]
        missing = [g for g in hvg_genes if g not in adata_full.var_names]
        if missing:
            log(f"scVelo anchor: {len(missing)} HVG genes missing in raw adata: {missing[:10]}")
        ad = adata_full[cell_idx, present_genes].to_memory().copy()
        ad.obs_names_make_unique()
        ad.var_names_make_unique()
    except Exception as e:
        return None, {"status": f"load_failed: {e}"}

    try:
        scv.settings.verbosity = 2
        scv.pp.filter_and_normalize(ad)
        scv.pp.moments(ad, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(ad, mode="dynamical")
        v_sc = np.asarray(ad.layers["velocity"], dtype=np.float32)
    except Exception as e:
        return None, {"status": f"scvelo_failed: {e}"}

    elapsed = time.time() - t0
    peak_ram_after = process_ram_gb()

    # 合格性检查：非全 NaN 且范数非零
    if not np.isfinite(v_sc).any():
        return None, {"status": "all_nan", "runtime_sec": round(elapsed, 2)}
    norms = np.linalg.norm(v_sc, axis=1)
    if np.max(norms) < 1e-12:
        return None, {"status": "zero_norm", "runtime_sec": round(elapsed, 2)}

    retained_genes = list(ad.var_names)
    retained_cells = int(ad.n_obs)
    log(f"scVelo completed in {elapsed:.1f}s; retained {retained_cells} cells, {len(retained_genes)} genes")

    info = {
        "status": "ran",
        "runtime_sec": round(elapsed, 2),
        "peak_ram_gb": round(max(peak_ram_before, peak_ram_after), 3),
        "retained_cells": retained_cells,
        "retained_genes": len(retained_genes),
        "retained_gene_names": retained_genes,
    }
    return v_sc, info


def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n_finite": 0, "median": float("nan"), "p99": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n_finite": int(finite.size),
        "median": float(np.median(finite)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def interpret_rubric(cosine_median: float) -> str:
    if cosine_median >= 0.9:
        return "strong agreement"
    if cosine_median >= 0.7:
        return "partial directional agreement"
    return "reformulation changes field (limitation)"


def append_diagnosis_section(
    result: dict,
    savi_v: np.ndarray,
    reg_v: np.ndarray,
    sc_v: np.ndarray | None,
    sc_info: dict,
):
    """追加第 5 节到 B1_headtohead_diagnosis.md。"""
    pairwise = result["pairwise"]
    rubric = result["interpretation"]

    lines = ["\n\n## 5. N5000 upgrade rerun (this execution)\n\n"]
    lines.append("### 5.1 Design\n\n")
    lines.append(f"- cells: {result['design']['n_cells']} (cell_index_n5000, seed={result['design']['seed']})\n")
    lines.append(f"- genes: {result['design']['n_genes']} (HVG source `{HVG_SOURCE}`)\n")
    lines.append(f"- HVG gene hash (sha256 first 16): {result['design']['gene_hash']}\n")
    lines.append(f"- status: **{result.get('status', 'unknown')}**\n\n")

    lines.append("### 5.2 Wall-clock / peak resources\n\n")
    lines.append("| method | wall-clock (s) | peak GPU (GB) | notes |\n")
    lines.append("|---|---|---|---|\n")
    savi = result["savi"]
    reg = result["regvelo_gpu"]
    sc = result["scvelo"]
    lines.append(f"| SAVI L2+L3 | {savi['l2_runtime_sec'] + savi['l3_runtime_sec']:.1f} | L2={savi['l2_peak_gpu_gb']:.2f}, L3={savi['l3_peak_gpu_gb']:.2f} | production protocol K=10, bs=1024 |\n")
    lines.append(f"| RegVelo GPU | {reg['total_runtime_sec']:.1f} | {reg['peak_gpu_gb']:.2f} | max_epochs=200, early_stopping |\n")
    if sc.get("status") == "ran":
        lines.append(f"| scVelo dynamical | {sc['runtime_sec']:.1f} | CPU RAM {sc.get('peak_ram_gb', 'N/A')} GB | filter_and_normalize → moments(n_pcs=30) → velocity(mode='dynamical') |\n")
    else:
        lines.append(f"| scVelo dynamical | — | — | status={sc.get('status')} |\n")
    lines.append("\n")

    lines.append("### 5.3 Pairwise per-cell metrics\n\n")
    lines.append("Cosine similarity (median / P99 / min):\n\n")
    lines.append("| pair | median | P99 | min |\n")
    lines.append("|---|---|---|---|\n")
    for pair in ("savi_regvelo", "savi_scvelo", "regvelo_scvelo"):
        c = pairwise[pair]["cosine"]
        lines.append(f"| {pair} | {c['median']:.4f} | {c['p99']:.4f} | {c['min']:.4f} |\n")
    lines.append("\nRelative L2 (median / P99 / min):\n\n")
    lines.append("| pair | median | P99 | min |\n")
    lines.append("|---|---|---|---|\n")
    for pair in ("savi_regvelo", "savi_scvelo", "regvelo_scvelo"):
        r = pairwise[pair]["relative_l2"]
        lines.append(f"| {pair} | {r['median']:.4f} | {r['p99']:.4f} | {r['min']:.4f} |\n")
    lines.append("\n")

    lines.append("### 5.4 Interpretation (frozen, status=open)\n\n")
    lines.append(f"- SAVI–RegVelo: median cosine = {pairwise['savi_regvelo']['cosine']['median']:.4f} → `{rubric['savi_regvelo']}`\n")
    if sc.get("status") == "ran":
        lines.append(f"- SAVI–scVelo: median cosine = {pairwise['savi_scvelo']['cosine']['median']:.4f} → `{rubric['savi_scvelo']}`\n")
        lines.append(f"- RegVelo–scVelo: median cosine = {pairwise['regvelo_scvelo']['cosine']['median']:.4f} → `{rubric['regvelo_scvelo']}`\n")
    else:
        lines.append("- scVelo anchor failed or excluded; no external anchor available.\n")
    lines.append(f"\nPre-registered rubric: median cosine ≥0.9 strong agreement, 0.7–0.9 partial, <0.7 reformulation changes field.\n")
    lines.append("**All numbers above are frozen pending human review.**\n")

    with open(DIAG_MD, "a", encoding="utf-8") as f:
        f.write("".join(lines))
    log(f"Appended section 5 to {DIAG_MD}")


def main() -> int:
    result: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "open",
        "design": {},
        "savi": {},
        "regvelo_gpu": {},
        "scvelo": {},
        "pairwise": {},
        "interpretation": {},
        "interpretation_rubric": {
            "cosine_median_ge_0.9": "strong agreement",
            "cosine_median_0.7_to_0.9": "partial directional agreement (broad directional agreement in text)",
            "cosine_median_lt_0.7": "reformulation changes field (limitation)",
        },
        "versions": {
            "regvelo": __import__("regvelo").__version__,
            "scvelo": __import__("scvelo").__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    try:
        hvg_genes = load_hvg_genes()
        G, W_nnz, gene_hash, present_genes = build_inputs(hvg_genes)
        result["design"] = {
            "n_cells": N_CELLS,
            "n_genes": G,
            "seed": SEED,
            "hvg_source": str(HVG_SOURCE),
            "gene_hash": gene_hash,
            "W_nnz": W_nnz,
        }

        # SAVI
        l2_rt, l2_mem = run_savi_l2()
        v_savi, l3_rt, l3_mem = run_savi_l3()
        result["savi"] = {
            "l2_runtime_sec": round(l2_rt, 2),
            "l2_peak_gpu_gb": round(l2_mem, 3),
            "l3_runtime_sec": round(l3_rt, 2),
            "l3_peak_gpu_gb": round(l3_mem, 3),
            "velocity_path": str(INPUT_DIR / "l3_production" / "v.npy"),
        }

        # RegVelo
        reg_v, reg_info = run_regvelo_gpu()
        result["regvelo_gpu"] = reg_info
        if reg_v is None:
            result["status"] = "regvelo_failed"
            RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            log("RegVelo failed; wrote partial result.")
            (OUT_DIR / "B1_headtohead_n5000_console.log").write_text("\n".join(log_lines), encoding="utf-8")
            return 1

        # scVelo
        sc_v, sc_info = run_scvelo_dynamical(present_genes)
        result["scvelo"] = sc_info

        # Pairwise comparison
        pairwise = {}
        if sc_v is not None and sc_info.get("status") == "ran":
            retained_genes = sc_info["retained_gene_names"]
            common = [g for g in present_genes if g in retained_genes]
            idx_savi = [present_genes.index(g) for g in common]
            idx_sc = [retained_genes.index(g) for g in common]

            v_savi_c = v_savi[:, idx_savi]
            reg_v_c = reg_v[:, idx_savi]
            sc_v_c = sc_v[:, idx_sc]

            # 仅保留 scVelo 过滤后仍存在的细胞（前 retained_cells 行）
            n_sc = sc_info["retained_cells"]
            v_savi_c = v_savi_c[:n_sc, :]
            reg_v_c = reg_v_c[:n_sc, :]
            sc_v_c = sc_v_c[:n_sc, :]

            cos_sr = cosine_per_cell(v_savi_c, reg_v_c)
            rel_sr = rel_l2_per_cell(v_savi_c, reg_v_c)
            cos_ssc = cosine_per_cell(v_savi_c, sc_v_c)
            rel_ssc = rel_l2_per_cell(v_savi_c, sc_v_c)
            cos_rsc = cosine_per_cell(reg_v_c, sc_v_c)
            rel_rsc = rel_l2_per_cell(reg_v_c, sc_v_c)

            pairwise = {
                "savi_regvelo": {"cosine": summarize(cos_sr), "relative_l2": summarize(rel_sr)},
                "savi_scvelo": {"cosine": summarize(cos_ssc), "relative_l2": summarize(rel_ssc)},
                "regvelo_scvelo": {"cosine": summarize(cos_rsc), "relative_l2": summarize(rel_rsc)},
                "common_genes": len(common),
            }
        else:
            # 仅 SAVI–RegVelo
            cos_sr = cosine_per_cell(v_savi, reg_v)
            rel_sr = rel_l2_per_cell(v_savi, reg_v)
            pairwise = {
                "savi_regvelo": {"cosine": summarize(cos_sr), "relative_l2": summarize(rel_sr)},
                "savi_scvelo": {"cosine": summarize([]), "relative_l2": summarize([])},
                "regvelo_scvelo": {"cosine": summarize([]), "relative_l2": summarize([])},
                "common_genes": G,
                "scvelo_note": sc_info.get("status"),
            }

        result["pairwise"] = pairwise
        result["interpretation"] = {
            "savi_regvelo": interpret_rubric(pairwise["savi_regvelo"]["cosine"]["median"]),
            "savi_scvelo": interpret_rubric(pairwise["savi_scvelo"]["cosine"]["median"]) if pairwise["savi_scvelo"]["cosine"]["n_finite"] > 0 else "unavailable",
            "regvelo_scvelo": interpret_rubric(pairwise["regvelo_scvelo"]["cosine"]["median"]) if pairwise["regvelo_scvelo"]["cosine"]["n_finite"] > 0 else "unavailable",
        }

        RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        log(f"Wrote {RESULT_PATH}")

        # Append diagnosis
        append_diagnosis_section(result, v_savi, reg_v, sc_v, sc_info)

        # Console output
        (OUT_DIR / "B1_headtohead_n5000_console.log").write_text("\n".join(log_lines), encoding="utf-8")

        # CHECKPOINT lines
        p = result["pairwise"]
        print(f"N5000_STATUS=ran; cells={N_CELLS}; genes={G}; scvelo={result['scvelo'].get('status', 'unknown')}; status=open")
        sr = p["savi_regvelo"]["cosine"]
        ss = p["savi_scvelo"]["cosine"]
        rs = p["regvelo_scvelo"]["cosine"]
        print(f"PAIRWISE=SAVI-RegVelo:{sr['median']:.4f}/{sr['p99']:.4f}/{sr['min']:.4f}; SAVI-scVelo:{ss['median']:.4f}/{ss['p99']:.4f}/{ss['min']:.4f}; RegVelo-scVelo:{rs['median']:.4f}/{rs['p99']:.4f}/{rs['min']:.4f}")
        interp = result["interpretation"]
        print(f"INTERP=SAVI-RegVelo={interp['savi_regvelo']}; SAVI-scVelo={interp['savi_scvelo']}; RegVelo-scVelo={interp['regvelo_scvelo']}")
        return 0

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["trace"] = traceback.format_exc()
        RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        log(f"ERROR: {e}")
        (OUT_DIR / "B1_headtohead_n5000_console.log").write_text("\n".join(log_lines), encoding="utf-8")
        print(f"N5000_STATUS=failed; error={e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
