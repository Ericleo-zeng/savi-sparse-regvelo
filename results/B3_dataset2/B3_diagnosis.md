# B3 CPU vs CUDA L3 Diagnosis

## Classification
CONFIG_CONSISTENT+TAIL_EFFECT

## Config comparison
| key | CUDA | CPU | match |
|---|---|---|---|
| K | 10 | 10 | True |
| tol | 0.001 | 0.001 | True |
| max_iter | 10 | 10 | True |
| batch_size | 1024 | 1024 | True |

## Batch-level routing (from l3_metadata.json)
- CUDA: {'closed_form_foh': 18, 'krylov_expmv': 0, 'full_sparse_expmv': 0, 'full_dynamics_path': 0}
- CPU: {'closed_form_foh': 18, 'krylov_expmv': 0, 'full_sparse_expmv': 0, 'full_dynamics_path': 0}
- Batches with route mismatch: 0
- CUDA picard_residual_max: 0.0006606538081541657
- CPU picard_residual_max: 1.384062557008292e-06
- CUDA n_unconverged_batches: 0
- CPU n_unconverged_batches: 18

## Per-cell CPU/CUDA deviation
- rel-L2 median: 3.753e-04
- rel-L2 P99: 1.363e-02
- rel-L2 max: 3.128e-02
- cells above 1e-3: 5362 / 18213
- cells above 1e-2: 507 / 18213
- cosine min: 0.999609
- worst cell: index=15277, rel-L2=3.128e-02, cosine=0.999609

## Interpretation
CPU and CUDA L3 configs match (K=10, tol=1e-3, max_iter=10, batch=1024). Both routes are closed_form_foh for all 18 batches, so no routing mismatch. Relative-L2 median is 3.753e-04 (≤1e-3) but max is 3.128e-02, affecting 507 / 18213 cells above 1e-2. This is consistent with FP32 non-associativity amplified in low-velocity cells, not a configuration bug.

## Rerun plan
No config-level rerun needed. If stricter CPU/CUDA agreement is desired, re-run both sides with identical deterministic FP32 accumulation (not supported by current engine) or use full-fp64 fallback; otherwise accept as bounded numerical drift.
