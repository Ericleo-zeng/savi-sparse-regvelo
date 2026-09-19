#!/usr/bin/env bash
# 05_sensitivity.sh — V5/V8 敏感性验证驱动（在子集上跑，不动 production 输出）
#
# 前置：01_P0_修复说明.md 中的 T0.1（rate_max provenance）与 T0.3（--K 覆盖）已修复。
# 用法：
#   cd /home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo
#   bash 05_sensitivity.sh            # 全部跑
#   或逐段复制执行（建议逐段，观察内存）
#
# V6（GRN tier weighting）与 V7（proliferating genes）需要修改 W / gene universe，
# 属于更重的一步，见本文件末尾的说明。

set -euo pipefail

PROJ="/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/sparse_regvelo"
VAL="/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo"
RES="${VAL}/results"
SUB="${RES}/_subset"
N_SUB=5000
INPUT="/mnt/t9/datasets/ad_microglia_fate_landscape_output/10_regvelo/sparse_engine_input"

# 确保子集输入存在（若 02/03 已跑过则秒回）
cd "${VAL}"
python - <<EOF
import config as C
C.ensure_subset(${N_SUB})
EOF

# ================================================================
# V5: rate_max 敏感性（需重跑 L2，仅子集 5000 cells，CPU）
#   条件：manual 20 / manual 100 ；production adaptive 值作为参照
# ================================================================
for RM in 20 100; do
  CKPT="${RES}/V5_rate_max/ckpt_r${RM}"
  L3OUT="${RES}/V5_rate_max/l3_r${RM}"
  if [ ! -f "${CKPT}/metadata.json" ]; then
    echo "[V5] L2 rate_max=${RM} ..."
    cd "${PROJ}"
    python sparse_regvelo_engine.py \
      --ms  "${SUB}/Ms_sub_n${N_SUB}.npy" \
      --mu  "${SUB}/Mu_sub_n${N_SUB}.npy" \
      --x-scvi "${SUB}/X_sub_n${N_SUB}.npy" \
      --w   "${INPUT}/W.npz" \
      --l2-only --rate-max ${RM} \
      --l2-checkpoint "${CKPT}" \
      --out-dir "${RES}/V5_rate_max/logs_r${RM}"
  fi
  if [ ! -f "${L3OUT}/l3_metadata.json" ]; then
    echo "[V5] L3 rate_max=${RM} ..."
    cd "${PROJ}"
    python run_l3_only.py \
      --checkpoint "${CKPT}" \
      --ms "${SUB}/Ms_sub_n${N_SUB}.npy" \
      --mu "${SUB}/Mu_sub_n${N_SUB}.npy" \
      --out-dir "${L3OUT}" --batch-size 2048
  fi
done

# V5 参照：adaptive 模式在子集上跑一次（即"子集版 production 口径"）
CKPT="${RES}/V5_rate_max/ckpt_adaptive"
L3OUT="${RES}/V5_rate_max/l3_adaptive"
if [ ! -f "${CKPT}/metadata.json" ]; then
  echo "[V5] L2 adaptive ..."
  cd "${PROJ}"
  python sparse_regvelo_engine.py \
    --ms  "${SUB}/Ms_sub_n${N_SUB}.npy" \
    --mu  "${SUB}/Mu_sub_n${N_SUB}.npy" \
    --x-scvi "${SUB}/X_sub_n${N_SUB}.npy" \
    --w   "${INPUT}/W.npz" \
    --l2-only \
    --l2-checkpoint "${CKPT}" \
    --out-dir "${RES}/V5_rate_max/logs_adaptive"
fi
if [ ! -f "${L3OUT}/l3_metadata.json" ]; then
  cd "${PROJ}"
  python run_l3_only.py \
    --checkpoint "${CKPT}" \
    --ms "${SUB}/Ms_sub_n${N_SUB}.npy" \
    --mu "${SUB}/Mu_sub_n${N_SUB}.npy" \
    --out-dir "${L3OUT}" --batch-size 2048
fi

# ================================================================
# V8: K 敏感性（只重跑 L3，需要 T0.3 的 --K 覆盖参数）
#   用子集 checkpoint（02/03 已建），K=10(production口径)/20/40
# ================================================================
SUB_CKPT="${SUB}/l2_checkpoint_sub_n${N_SUB}"
for K in 10 20 40; do
  L3OUT="${RES}/V8_K_sensitivity/l3_K${K}"
  if [ ! -f "${L3OUT}/l3_metadata.json" ]; then
    echo "[V8] L3 K=${K} ..."
    cd "${PROJ}"
    python run_l3_only.py \
      --checkpoint "${SUB_CKPT}" \
      --ms "${SUB}/Ms_sub_n${N_SUB}.npy" \
      --mu "${SUB}/Mu_sub_n${N_SUB}.npy" \
      --out-dir "${L3OUT}" --batch-size 2048 --K ${K}
  fi
done

echo ""
echo "[05] 完成。下一步对比："
echo "  cd ${VAL}"
echo "  python 06_compare_conditions.py --ref results/V5_rate_max/l3_adaptive \\"
echo "      --test results/V5_rate_max/l3_r20 results/V5_rate_max/l3_r100"
echo "  python 06_compare_conditions.py --ref results/V8_K_sensitivity/l3_K40 \\"
echo "      --test results/V8_K_sensitivity/l3_K10 results/V8_K_sensitivity/l3_K20"

# ================================================================
# V6 / V7 说明（本脚本不自动执行）
# ================================================================
# V6 GRN tier weighting：修改 export_for_sparse_engine.py 中 W 构建逻辑
#   （按 tier 赋权重而非 1.0），导出 W_tiered.npz 到独立目录，然后照 V5
#   的 L2→L3 流程跑子集，用 06_compare_conditions.py 对比。
# V7 proliferating genes：准备剔除增殖基因的 gene_universe 子集，用
#   export 脚本导出新 Ms/Mu/W（G 变小），整套 L2→L3 重跑子集。
#   两项都建议先在 5000 cells 子集上做方向性检查，再决定是否上全量。
