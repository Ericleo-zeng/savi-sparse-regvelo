#!/usr/bin/env python3
"""
sparse_regvelo_preflight_check_v2.py
修正版：适配你本地实际路径
"""

import sys
import argparse
from pathlib import Path

import scanpy as sc
import numpy as np

# -----------------------------------------------------------------------------
# 修正后的实际路径（根据你的 tree 输出）
# -----------------------------------------------------------------------------
ENGINE_PATH = Path("./sparse_regvelo_engine.py")  # 当前目录下
ADATA_PATH = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
OUT_DIR = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("[Check 1] 引擎文件存在性")
print("=" * 70)
print(f"  查找: {ENGINE_PATH.absolute()}")
print(f"  存在: {ENGINE_PATH.exists()}")
if not ENGINE_PATH.exists():
    print("  ❌ 请 cd 到 sparse_regvelo 目录后再运行本脚本")
    sys.exit(1)
else:
    print("  ✅ 引擎文件找到")

# 快速 grep 关键参数
src = ENGINE_PATH.read_text(encoding="utf-8")
has_npz = "npz" in src.lower() and ("w" in src or "csr" in src)
has_bf16 = "bf16" in src.lower() or "float16" in src.lower()
has_xscvi_dim = "x_scvi" in src.lower()

print(f"  源码含 npz/CSR 线索: {has_npz}")
print(f"  源码含 bf16 线索: {has_bf16}")

print("\n" + "=" * 70)
print("[Check 2] AnnData 检查")
print("=" * 70)

if not ADATA_PATH.exists():
    print(f"❌ AnnData 不存在: {ADATA_PATH}")
    sys.exit(1)

adata = sc.read_h5ad(ADATA_PATH, backed="r")
print(f"  文件: {ADATA_PATH}")
print(f"  形状: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"  obsm keys: {list(adata.obsm.keys())}")

has_x_scvi = "X_scVI" in adata.obsm
x_scvi_dim = adata.obsm["X_scVI"].shape[1] if has_x_scvi else None

print(f"  含 X_scVI: {has_x_scvi}")
if has_x_scvi:
    print(f"  X_scVI 维度: {x_scvi_dim}")
    if x_scvi_dim != 10:
        print(f"  ⚠️  维度不是 10！建议取前 10 列或 PCA 降维")
        print(f"     导出脚本将自动取前 10 列 (X_scVI[:, :10])")

adata.file.close()

print("\n" + "=" * 70)
print("[Check 3] 运行建议")
print("=" * 70)

w_ext = "npz" if has_npz else "npy"
print(f"  W 格式建议: .{w_ext}")

if x_scvi_dim and x_scvi_dim != 10:
    print(f"  X_scVI 处理: 取前 10 列 → X_scVI_10d.npy")

print(f"\n  下一步: 运行修正版 export_for_sparse_engine.py")
