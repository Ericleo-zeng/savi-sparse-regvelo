#!/usr/bin/env python3
# export_for_sparse_engine.py
# 修正版：适配 30d X_scVI → 取前 10d；W → CSR .npz

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz

# -----------------------------------------------------------------------------
# 你的实际数据路径
# -----------------------------------------------------------------------------
ADATA_PATH = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
NETWORK_PATH = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/network.tsv")
GENE_LIST_PATH = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/09_grn/part2/targeted_grn_v2/production/gene_universe_used.txt")
OUT_DIR = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TF_ORDER = [
    "Cebpb", "Fos", "Irf7", "Jun", "Nfatc1",
    "Nfatc2", "Nfkb1", "Rela", "Spi1", "Stat1", "Stat2"
]

print("[1] Loading frozen gene universe...")
genes = [x.strip() for x in GENE_LIST_PATH.read_text().splitlines() if x.strip()]
genes = list(dict.fromkeys(genes))
print(f"    genes: {len(genes):,}")

print("[2] Loading AnnData...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"    original: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

missing = [g for g in genes if g not in adata.var_names]
if missing:
    raise ValueError(f"{len(missing)} frozen genes missing: {missing[:10]}")
adata = adata[:, genes].copy()
print(f"    subset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

# -----------------------------------------------------------------------------
# Ms / Mu
# -----------------------------------------------------------------------------
if "Ms" not in adata.layers or "Mu" not in adata.layers:
    print("[3] Ms/Mu not found — computing moments...")
    if "X_scVI" not in adata.obsm:
        raise ValueError("X_scVI missing")
    scv.pp.normalize_per_cell(adata, layers=["spliced", "unspliced"])
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, key_added="regvelo_neighbors")
    adata.uns["neighbors"] = adata.uns["regvelo_neighbors"]
    adata.obsp["connectivities"] = adata.obsp["regvelo_neighbors_connectivities"]
    adata.obsp["distances"] = adata.obsp["regvelo_neighbors_distances"]
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    print("    moments computed.")
else:
    print("[3] Reusing existing Ms/Mu layers.")

Ms = np.asarray(adata.layers["Ms"], dtype=np.float32)
Mu = np.asarray(adata.layers["Mu"], dtype=np.float32)
np.save(OUT_DIR / "Ms.npy", Ms)
np.save(OUT_DIR / "Mu.npy", Mu)
print(f"    Ms/Mu saved: {Ms.shape}")

# -----------------------------------------------------------------------------
# X_scVI：关键修正 — 取前 10 列（引擎兼容）
# -----------------------------------------------------------------------------
if "X_scVI" not in adata.obsm:
    raise ValueError("X_scVI not found in adata.obsm")

X_scVI_full = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
print(f"    X_scVI original dim: {X_scVI_full.shape[1]}")

if X_scVI_full.shape[1] > 10:
    print(f"    ⚠️  降维: {X_scVI_full.shape[1]}d → 10d (取前 10 列)")
    X_scVI = X_scVI_full[:, :10]
elif X_scVI_full.shape[1] < 10:
    raise ValueError(f"X_scVI dim {X_scVI_full.shape[1]} < 10, cannot proceed")
else:
    X_scVI = X_scVI_full

np.save(OUT_DIR / "X_scVI.npy", X_scVI)
print(f"    X_scVI saved: {X_scVI.shape}")

# -----------------------------------------------------------------------------
# W：CSR .npz，方向 row=target, col=regulator
# -----------------------------------------------------------------------------
print("[4] Building CSR W from network.tsv...")
net = pd.read_csv(NETWORK_PATH, sep="\t")
gene_to_idx = {g: i for i, g in enumerate(genes)}
n_genes = len(genes)

W_dense = np.zeros((n_genes, n_genes), dtype=np.float32)
for row in net.itertuples(index=False):
    if row.target in gene_to_idx and row.TF in gene_to_idx:
        W_dense[gene_to_idx[row.target], gene_to_idx[row.TF]] = 1.0

actual_edges = int(np.count_nonzero(W_dense))
if actual_edges != 194_843:
    raise ValueError(f"W edge count: {actual_edges:,} != 194,843")

W_csr = csr_matrix(W_dense)
save_npz(OUT_DIR / "W.npz", W_csr)

csr_bytes = W_csr.data.nbytes + W_csr.indices.nbytes + W_csr.indptr.nbytes
print(f"    W CSR saved: {W_csr.shape} | nnz={W_csr.nnz:,}")
print(f"    CSR storage: {csr_bytes/1024**2:.2f} MB")
print(f"    W orientation: row=target, col=regulator")

print(f"\n[Done] All inputs saved to: {OUT_DIR}")
print("    Ms.npy, Mu.npy, X_scVI.npy (10d), W.npz")
