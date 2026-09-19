#!/usr/bin/env bash
# 05_sensitivity.sh — V5/V8 敏感性验证驱动（在子集上跑，不动 production 输出）
#
# 前置（2026-09-15 更新）：
#   T0.1（rate_max provenance 记录）— 已修复
#   T0.3（--K 覆盖）— 已修复
#   T0.4（--rate-max 边界值覆盖：20.0 曾兼任 argparse 默认值与合法请求值，
#         导致 --rate-max 20 静默走 adaptive；引擎 187/224/309/312/572 五处已修）
#         — 已修复，详见 known_issues.md
#   T0.5（子集参考 checkpoint = 生产 per-gene 参数 + 子集 per-cell 量的拼接物，
#         仅适用于 L3 数值验证，禁止作 L2 参数级对比参考）— 已记录
#
# 用法：
#   cd /home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo
#   bash 05_sensitivity.sh            # 全部跑（已有合规产物则自动跳过）
#   或逐段复制执行（建议逐段，观察内存）
#
# 幂等语义（2026-09-15 加固）：checkpoint 守卫校验 metadata 内容
# （rate_max_actual + rate_max_mode + N），而非仅校验文件存在——T0.4 期间曾出现
# "文件存在但内容错误"的脏 checkpoint（requested=20/actual=107.31/mode=adaptive）
# 被静默复用。强制重跑：rm -rf 对应的 ckpt_*/ 或 l3_*/ 目录即可。
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

# ---------------------------------------------------------------- 内容守卫函数
# ckpt_ratemax_ok <ckpt_dir> <expected_actual|"-"> <expected_mode>
#   expected_actual="-" 时只校验 mode 与 N（用于 adaptive，actual 由数据决定）
ckpt_ratemax_ok() {
  python - "$1" "$2" "$3" "${N_SUB}" <<'EOF'
import json, sys
d, actual, mode, n_sub = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    with open(f"{d}/metadata.json", encoding="utf-8") as f:
        m = json.load(f)
except Exception:
    sys.exit(1)
if m.get("rate_max_mode") != mode:
    sys.exit(1)
if m.get("N") != n_sub:
    sys.exit(1)
if actual != "-" and abs(float(m.get("rate_max_actual", -1)) - float(actual)) > 1e-6:
    sys.exit(1)
sys.exit(0)
EOF
}

# provenance 自检：L2 结束后确认"请求值=实际值=模式"，T0.1/T0.4 复发时 fail-fast
assert_manual_ckpt() {
  python - "$1" "$2" <<'EOF'
import json, sys
d, rm = sys.argv[1], float(sys.argv[2])
m = json.load(open(f"{d}/metadata.json", encoding="utf-8"))
ok = (m.get("rate_max_mode") == "manual"
      and abs(float(m.get("rate_max_requested", -1)) - rm) < 1e-6
      and abs(float(m.get("rate_max_actual", -1)) - rm) < 1e-6)
if not ok:
    print(f"[FATAL] {d} provenance 异常: requested={m.get('rate_max_requested')} "
          f"actual={m.get('rate_max_actual')} mode={m.get('rate_max_mode')}，"
          f"期望 {rm}/{rm}/manual（T0.4 复发？）", file=sys.stderr)
    sys.exit(1)
print(f"[OK] {d} provenance: requested=actual={rm}, mode=manual")
EOF
}

assert_adaptive_ckpt() {
  python - "$1" <<'EOF'
import json, sys
d = sys.argv[1]
m = json.load(open(f"{d}/metadata.json", encoding="utf-8"))
ok = (m.get("rate_max_mode") == "adaptive"
      and m.get("rate_max_requested") is None
      and float(m.get("rate_max_actual", -1)) > 0)
if not ok:
    print(f"[FATAL] {d} provenance 异常: requested={m.get('rate_max_requested')} "
          f"actual={m.get('rate_max_actual')} mode={m.get('rate_max_mode')}，"
          f"期望 null/>0/adaptive", file=sys.stderr)
    sys.exit(1)
print(f"[OK] {d} provenance: requested=null, actual={m['rate_max_actual']}, mode=adaptive")
EOF
}

# 确保子集输入存在（若 02/03 已跑过则秒回）
cd "${VAL}"
python - <<EOF
import config as C
C.ensure_subset(${N_SUB})
EOF

# ================================================================
# V5: rate_max 敏感性（需重跑 L2，仅子集 5000 cells，CPU）
#   条件：manual 20 / manual 100 / adaptive 对照（同数据重拟合，唯一干净对比基准）
#   L3 配置：K=10（显式 CLI 覆盖，T0.3）、batch_size=1024
# 2026-09-15 已跑结论（3b 对比）：rm20 cosine 中位 0.386 / rm100 0.699，均 FAIL
#   → rate_max 高度敏感，caveat 成立；详见 known_issues.md 与 V5 存档记录
# ================================================================
for RM in 20 100; do
  CKPT="${RES}/V5_rate_max/ckpt_r${RM}"
  L3OUT="${RES}/V5_rate_max/l3_r${RM}"
  if ! ckpt_ratemax_ok "${CKPT}" "${RM}" "manual"; then
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
    assert_manual_ckpt "${CKPT}" "${RM}"
  else
    echo "[V5] SKIP L2 rate_max=${RM}（checkpoint 内容已合规）"
  fi
  # L3 守卫仍为存在性检查：上游 ckpt 内容守卫已保证"脏 L3 源头"不存在；
  # 若需强制重跑 L3：rm -rf "${L3OUT}"
  if [ ! -f "${L3OUT}/l3_metadata.json" ]; then
    echo "[V5] L3 rate_max=${RM} ..."
    cd "${PROJ}"
    python run_l3_only.py \
      --checkpoint "${CKPT}" \
      --ms "${SUB}/Ms_sub_n${N_SUB}.npy" \
      --mu "${SUB}/Mu_sub_n${N_SUB}.npy" \
      --out-dir "${L3OUT}" --batch-size 1024 --K 10
  fi
done

# V5 对照：adaptive 模式在子集上重拟合（同数据，唯一干净对比的基准）
CKPT="${RES}/V5_rate_max/ckpt_adaptive"
L3OUT="${RES}/V5_rate_max/l3_adaptive"
if ! ckpt_ratemax_ok "${CKPT}" "-" "adaptive"; then
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
  assert_adaptive_ckpt "${CKPT}"
else
  echo "[V5] SKIP L2 adaptive（checkpoint 内容已合规）"
fi
if [ ! -f "${L3OUT}/l3_metadata.json" ]; then
  cd "${PROJ}"
  python run_l3_only.py \
    --checkpoint "${CKPT}" \
    --ms "${SUB}/Ms_sub_n${N_SUB}.npy" \
    --mu "${SUB}/Mu_sub_n${N_SUB}.npy" \
    --out-dir "${L3OUT}" --batch-size 1024 --K 10
fi

# ================================================================
# V8: K 敏感性（只重跑 L3，需要 T0.3 的 --K 覆盖参数）
#   用子集 checkpoint（02/03 已建），K=10(production口径)/20/40
# ⚠ T0.5 警示：SUB_CKPT 是"生产 per-gene 参数 + 子集 per-cell 量"的拼接
#   checkpoint（config.ensure_subset_checkpoint 的设计行为），仅适用于 L3
#   数值验证；禁止用作 L2 参数级对比（如 rate_max 敏感性）的参考。
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
echo "  # 有效对比（3b，判决依据）：同数据重拟合，唯一变量 rate_max"
echo "  python 06_compare_conditions.py --ref results/V5_rate_max/l3_adaptive \\"
echo "      --test results/V5_rate_max/l3_r20 results/V5_rate_max/l3_r100"
echo "  # 旁证（3a，含 T0.5 拟合数据量混淆，不作判决依据）:"
echo "  python 06_compare_conditions.py --ref results/V8_K_sensitivity/l3_K10 \\"
echo "      --test results/V5_rate_max/l3_adaptive"
echo "  # V8 K 敏感性（基于拼接 checkpoint 的 L3 数值自洽性）:"
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
# 注意：V6/V7 若涉及 L2 参数级对比，参考必须用同数据重拟合的 checkpoint
#   （如 V5_rate_max/ckpt_adaptive 模式），不得使用 _subset/l2_checkpoint_sub_*
#   （T0.5 拼接物）。
