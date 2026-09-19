# Supplementary Material: Number-to-source traceability

This document contains the number-to-source traceability table and figure plan (formerly Appendix A of the manuscript). It is provided as Supplementary Material and is also available in the validation repository.

## Number-to-source traceability

Analysis protocol: v1.0, dated 2026-09-14, archived at osf.io/pjvk8 (Open-Ended Registration, OSF Registries).

### Rounding policy

Body values are rounded or truncated for readability from the exact JSON checkpoint values listed in Table A1. Percentages are computed from the unrounded source values before rounding. Where a body number and A1 number differ in the last digit, the A1 JSON value is authoritative.

### Table A1. Claims and sources (all 2026-09-15, timestamped, archived)



| # | Claim (value) | Source artifact |
|---|---|---|
| 1 | V1 production n=2000/5000 residual tables; v P99 1.04e-2/1.17e-2 | results/V1_ode_residual{V1_result.json}; results/V1_ode_residual_prod_n5000/V1_result.json |
| 2 | V1 subset-faithful PASS (v P99 9.36e-3) | results/V1_subset_faithful/V1_result.json |
| 3 | E2 budget sweep identical across 10/20/30/50 (med 1.195e-3, P99 8.159e-3, max 6.55e-2; 0 unconv) | results/V1_iteration_sweep/V1_iter_sweep.json |
| 4 | E2 iteration sweep (indistinguishable 10→50) + three-arm K/tol decomposition (median ~28× K effect; P99 6.77–9.79e-3; max tol-sensitive 6.55e-2→1.72e-2; cos ≥0.999999) | results/V1_iteration_sweep/V1_iter_sweep.json |
| 5 | E1 cross-hardware: v max rel 7.13e-6, P99 1.43e-6, cos min 1.0; u max 3.03e-4 | results/V3_cpu_gpu_faithful/V3_result.json |
| 6 | L2/L3 determinism (β med 3.91741; max|Δv|=0.0) | V5 adaptive rerun logs; results/V5_rate_max/l3_adaptive_rep |
| 7 | T0.4 evidence (requested 20/actual 107.31/adaptive → 20/20/manual) | ckpt_r20 first-run vs post-fix metadata; engine lines 187/224/309/312/572 |
| 8 | T0.5 evidence (β max|Δ|=0.0; metadata repair) | _subset/l2_checkpoint_sub_n5000/metadata.json |
| 9 | V5 3b: rm20 0.38587/1.075; rm100 0.69885/1.148 | results/V5_rate_max compare_manual_vs_adaptive.log |
| 10 | V6: P90=5.354, P50=1.751; counts 19,485/77,937/97,421; sums 82,808.8/126,648.2; L2 β Δ=0.0, bias Δ=3.18 | export_manifest_v6(_b).json; ckpt array diffs; V6_grn_tiered/COMPARE_result.json |
| 11 | V7b: cosine 0.80198/min 0.08674/rel 0.6115; V7a: 0.67965/0.29946/0.7952 | results/V7_no_cycling/COMPARE_v7_result.json; results/V7_no_renorm/COMPARE_v7_result.json |
| 12 | V8: K=10/20 cosine 0.99999 (min 0.998); faithful K=20 1.00000 (min 0.99965) | V8_K_sensitivity/COMPARE_result.json; l3_adaptive_K20probe compare |
| 13 | Memory 5.82 GB @ (5000, 17714, K10, bs1024); K20 OOM @ bs1024; K20 @ bs512 passes | L3 console logs |
| 14 | Production config: adaptive 107.31; K=10; bs=2048; tol 1e-3; 74,984 × 17,714; W 194,843 | ckpt_adaptive metadata; production l3_metadata |
| 15 | E2c tail grounding: C1q_inflammatory 2.02× enriched in top-1% residual (19/50, spread across 9/12 samples; within-state permutation p=0.57); Fisher exact p=1.5e-3, OR=2.67 (Bonferroni-significant across 6 states); Homeo/DAM 0.65–0.67×; Prolif 0/50; per-state mean uniform 1.59–1.72e-3 | results/E2c_tail_grounding/E2c_result.json + E2c_sample_aware.json (Fisher rerun, sample distribution, permutation) |
| 16 | E4 fate control: full vs no_cycle Spearman med 1.00 / agree 0.892 / shift ≤0.18; vs norenorm 0.943/0.835; vs diffusion 0.829/0.612 | results/E4_fate_control/E4_result.json |

### Figure plan（投稿前排版）

| Fig | 内容 | 数据状态 |
|---|---|---|
| 1 | Engine architecture + 三项验证纪律流程图 | 定稿（fig1_architecture_disciplines.py/png/pdf） |
| 2 | Dominance ranking 四象限（gene universe / rate bound / GRN / K × 方向-幅度） | 定稿（fig2_dominance_landscape.py/png/pdf） |
| 3 | E2/E2b 残差分解（iteration sweep + K/tol 三层） | 定稿（fig3_e2_residual_decomposition.py/png/pdf） |
| 4 | E4 fate 级对照（full vs no-cycle 的 fate 全景/一致性） | 定稿（fig4_e4_fate_control.py/png/pdf） |
| 5 | E2c 残差长尾的 cell-state 分布（C1q_inflammatory 2.0× 富集） | 定稿（fig5_e2c_state_distribution.py/png/pdf） |
| 6 | SAVI L3 scaling curves: runtime vs. cells and genes, dual-axis GPU memory, scVelo and A1#13 references | 定稿（fig6_scaling_curve.py/png/pdf） |
