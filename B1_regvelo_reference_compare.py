#!/usr/bin/env python3
"""B1_regvelo_reference_compare.py — SAVI vs RegVelo reference on a 5,000-cell subset.

Phase 2 of Plan B. Assumes config.py paths and sparse_regvelo/ modules are available.
Outputs: results/B1_regvelo_reference/B1_result.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Add sparse_regvelo to path
SPARSE_REGVELO_DIR = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/sparse_regvelo"
sys.path.insert(0, SPARSE_REGVELO_DIR)

from config import (
    EXPECTED_G,
    cosine_per_cell,
    ensure_subset,
    ensure_subset_checkpoint,
    rel_l2_per_cell,
)
from run_l3_only import run_l3_streaming

BASE = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
OUT_DIR = os.path.join(BASE, "results", "B1_regvelo_reference")
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_JSON = os.path.join(OUT_DIR, "B1_result.json")

import argparse

N_CELLS_DEFAULT = 5000
SEED = 20260914
BATCH_SIZE_SAVI = 1024


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def gpu_memory_mb() -> float:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return float("nan")


def convert_w_npz_to_torch(npz_path: str) -> torch.Tensor:
    """独立可复现函数：将 SAVI 的 W.npz 转换为 RegVelo 需要的 dense torch.Tensor。

    打印转换前后的 shape 与 sparsity。
    """
    from scipy.sparse import load_npz

    W_sparse = load_npz(npz_path)
    print(f"[W conversion] input sparse shape={W_sparse.shape}, nnz={W_sparse.nnz}, "
          f"sparsity={1 - W_sparse.nnz / (W_sparse.shape[0] * W_sparse.shape[1]):.6f}")
    W_dense = torch.from_numpy(W_sparse.toarray()).to(dtype=torch.float32)
    print(f"[W conversion] output dense shape={tuple(W_dense.shape)}, "
          f"sparsity={1 - W_dense.count_nonzero().item() / W_dense.numel():.6f}")
    return W_dense


def run_savi_subset() -> tuple[str, np.ndarray, float, float]:
    """Run SAVI L3 on 5k subset; return (out_dir, v_array, runtime_sec, peak_mem_gb)."""
    subset_paths = ensure_subset(N_CELLS)
    ckpt_dir = ensure_subset_checkpoint(N_CELLS)
    out_dir = os.path.join(OUT_DIR, "savi_l3_subset")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    mem_before = gpu_memory_mb()
    run_l3_streaming(
        ms_path=subset_paths["ms"],
        mu_path=subset_paths["mu"],
        checkpoint_dir=ckpt_dir,
        out_dir=out_dir,
        batch_size=BATCH_SIZE_SAVI,
        device="cuda",
        K=None,  # from checkpoint meta
    )
    runtime = time.time() - t0
    mem_after = gpu_memory_mb()
    v = np.load(os.path.join(out_dir, "v.npy"), mmap_mode="r")
    return out_dir, np.asarray(v, dtype=np.float32), runtime, (mem_after - mem_before) / 1024


def build_anndata():
    import anndata

    subset_paths = ensure_subset(N_CELLS)
    Ms = np.load(subset_paths["ms"])
    Mu = np.load(subset_paths["mu"])

    gene_list_path = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/gene_universe_used.txt"
    if os.path.isfile(gene_list_path):
        with open(gene_list_path, "r", encoding="utf-8") as f:
            var_names = [line.strip() for line in f if line.strip()]
    else:
        var_names = [f"gene_{i}" for i in range(Ms.shape[1])]
    if len(var_names) != Ms.shape[1]:
        print(f"[warn] gene list length {len(var_names)} != {Ms.shape[1]}; using generic names")
        var_names = [f"gene_{i}" for i in range(Ms.shape[1])]

    adata = anndata.AnnData(X=Ms)
    adata.layers["Ms"] = Ms
    adata.layers["Mu"] = Mu
    adata.var_names = var_names
    adata.obs_names = [f"cell_{i}" for i in range(Ms.shape[0])]
    return adata


def run_regvelo_reference(max_epochs: int) -> tuple[np.ndarray | None, dict]:
    """Train RegVelo reference on the same subset and return velocity + timing info."""
    from regvelo import REGVELOVI

    adata = build_anndata()
    W_npz_path = os.path.join(ensure_subset_checkpoint(N_CELLS), "W.npz")
    W_tensor = convert_w_npz_to_torch(W_npz_path)

    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    model = REGVELOVI(
        adata,
        W=W_tensor,
        soft_constraint=True,
    )

    print("[RegVelo] start training")
    t0 = time.time()
    mem_before = gpu_memory_mb()
    try:
        model.train(
            max_epochs=max_epochs,
            batch_size=512,
            train_size=0.9,
            early_stopping=True,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
        )
    except Exception as e:
        return None, {"error": str(e), "trace": traceback.format_exc()}
    train_runtime = time.time() - t0
    mem_after = gpu_memory_mb()

    print("[RegVelo] get_velocity")
    t0 = time.time()
    v = model.get_velocity(adata, return_numpy=True)
    vel_runtime = time.time() - t0

    info = {
        "train_runtime_sec": round(train_runtime, 2),
        "velocity_runtime_sec": round(vel_runtime, 2),
        "total_runtime_sec": round(train_runtime + vel_runtime, 2),
        "peak_gpu_memory_delta_gb": round((mem_after - mem_before) / 1024, 3),
    }
    return np.asarray(v, dtype=np.float32), info


def main() -> int:
    ap = argparse.ArgumentParser(description="B1 RegVelo reference comparison")
    ap.add_argument("--n-cells", type=int, default=N_CELLS_DEFAULT)
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()
    global N_CELLS
    N_CELLS = args.n_cells

    result: dict = {
        "timestamp": now(),
        "n_cells": N_CELLS,
        "n_genes": EXPECTED_G,
        "seed": SEED,
        "savi": {},
        "regvelo": {"status": "pending"},
        "comparison": {},
        "versions": {},
    }

    try:
        result["versions"] = {
            "regvelo": __import__("regvelo").__version__,
            "scvelo": __import__("scvelo").__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
    except Exception as e:
        result["versions"] = {"error": str(e)}

    # SAVI L3
    try:
        savi_out, savi_v, savi_rt, savi_mem = run_savi_subset()
        result["savi"] = {
            "status": "ran",
            "output_dir": savi_out,
            "runtime_sec": round(savi_rt, 2),
            "peak_gpu_memory_delta_gb": round(savi_mem, 3),
        }
        print(f"[SAVI] runtime={savi_rt:.1f}s, v shape={savi_v.shape}")
    except Exception as e:
        result["savi"] = {"status": "failed", "error": str(e), "trace": traceback.format_exc()}
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print("[B1] SAVI subset failed:", e)
        return 1

    # RegVelo
    reg_v, reg_info = run_regvelo_reference(args.max_epochs)
    if reg_v is None:
        result["regvelo"] = {"status": "failed", **reg_info}
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print("[B1] RegVelo failed:", reg_info.get("error"))
        return 1

    result["regvelo"].update({"status": "ran", **reg_info})
    print(f"[RegVelo] runtime={reg_info['total_runtime_sec']:.1f}s, v shape={reg_v.shape}")

    # Compare
    if savi_v.shape != reg_v.shape:
        result["comparison"] = {"error": f"shape mismatch: SAVI {savi_v.shape} vs RegVelo {reg_v.shape}"}
    else:
        cos = cosine_per_cell(reg_v, savi_v)
        rel = rel_l2_per_cell(reg_v, savi_v)
        result["comparison"] = {
            "cosine": {
                "median": round(float(np.median(cos)), 6),
                "p99": round(float(np.percentile(cos, 99)), 6),
                "min": round(float(np.min(cos)), 6),
            },
            "relative_l2": {
                "median": round(float(np.median(rel)), 6),
                "p99": round(float(np.percentile(rel, 99)), 6),
                "min": round(float(np.min(rel)), 6),
            },
        }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print("[B1] result saved to", RESULT_JSON)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
