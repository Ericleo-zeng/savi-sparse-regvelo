# E4 前置：从 adata 提取子集 labels / sample ids（一次性，约 1-2 分钟）
import numpy as np
import scanpy as sc

VAL = "/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
AD = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad"

idx = np.load(f"{VAL}/results/_subset/cell_index_n5000.npy")
adata = sc.read_h5ad(AD)
print("states available:", [c for c in adata.obs.columns if "state" in c.lower()
      or "micro" in c.lower() or "label" in c.lower() or "anno" in c.lower()])

STATE_KEY = "microglia_state"      # 按实际列名修改
SAMPLE_KEY = "sample"              # 按实际列名修改

np.save(f"{VAL}/results/_subset/state_labels_n5000.npy",
        adata.obs[STATE_KEY].values[idx].astype(str))
np.save(f"{VAL}/results/_subset/sample_ids_n5000.npy",
        adata.obs[SAMPLE_KEY].values[idx].astype(str))
print("done")
