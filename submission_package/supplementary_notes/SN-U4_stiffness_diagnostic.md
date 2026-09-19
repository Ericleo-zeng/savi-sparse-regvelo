# Supplementary Note SN-U4 — Per-cell stiffness diagnostic

This note documents the stiffness diagnostic mentioned in §3.1 and §4. The goal was to test whether the persistent upper tail of the production L3 residual was driven by locally stiff dynamics (i.e., cells where degradation and splicing terms are badly balanced relative to each other). The diagnostic did **not** resolve the tail after accounting for proximity to steady state.

## Estimator

For each cell, we computed the element-wise ratio

\[
r_i = \frac{\|\beta \odot u_i\|_2}{\|\gamma \odot s_i\|_2}
\]

where β and γ are the per-gene kinetic rates from the L2 fit, and u_i and s_i are the unspliced and spliced moment vectors for cell i. A ratio far from 1 indicates an imbalance between the splicing influx and degradation efflux; ratios near 1 indicate proximity to a pseudo-steady-state balance.

## Sample

- N = 5,000 cells (subset of the main microglia atlas).
- G = 17,714 genes.

## Overall distribution

| Statistic | Ratio value |
|---|---:|
| Minimum | 0.9916 |
| Median | 1.0055 |
| Mean | 1.0164 |
| P99 | 1.3388 |
| Maximum | 1.5252 |

The bulk of cells lies very close to 1.0, with only a modest upper tail. The P99 ratio (1.34) is substantially smaller than the corresponding residual P99 (≈1e-2), suggesting that simple kinetic imbalance does not explain the residual tail.

## By cell-state annotation

| State | n | Median | Mean | P99 | Maximum |
|---|---:|---:|---:|---:|---:|
| C1q_inflammatory | 942 | 1.0054 | 1.0150 | 1.2475 | 1.5017 |
| DAM_like | 462 | 1.0050 | 1.0154 | 1.3262 | 1.4353 |
| Homeostatic | 1,794 | 1.0057 | 1.0125 | 1.1547 | 1.5252 |
| IFN_responsive | 1,393 | 1.0057 | 1.0220 | 1.3641 | 1.4931 |
| PU1low_lymphoid | 359 | 1.0049 | 1.0181 | 1.3868 | 1.4867 |
| Proliferating | 50 | 1.0104 | 1.0301 | 1.3677 | 1.4141 |

No state showed a dramatically shifted ratio distribution. In particular, the C1q_inflammatory state, which was enriched in the top-1% residual tail (Fig. 5), had a stiffness profile comparable to other states.

## Approximate Jacobian condition number

We also computed an approximate condition number of the L3 Jacobian via one-sided singular-value probing (`scipy.sparse.linalg.svds(k=1)`):

| Quantity | Value |
|---|---:|
| σ_max | 196.10 |
| σ_min | 0.06428 |
| Condition number κ(J) | 3,050.73 |
| Compute time | 23.9 s |

This condition number is moderate and does not indicate globally ill-conditioned dynamics.

## Conclusion

Neither the per-cell stiffness ratio nor the approximate Jacobian condition number explains the state-structured residual tail. The tail is therefore provisionally attributed to unresolved numerical sensitivity (possibly FP32 accumulation patterns or local non-linearity in specific cells) rather than to a simple stiffness imbalance. This is recorded as an open caveat in §4.

## Note on main-text use

The numeric values in this note (ratios and κ(J)) are provided for transparency only. They were deliberately excluded from the main text because they do not resolve the tail mechanism; the main text reports only the qualitative conclusion that the diagnostic did not explain the tail.

## Source file

- Raw result JSON: `results/U4_diagnostic_result.json`
