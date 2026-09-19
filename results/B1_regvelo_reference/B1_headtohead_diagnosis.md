# B1 head-to-head diagnosis

## 1. Gene-axis alignment

- SAVI gene order length: 2000
- RegVelo placeholder order length: 2000
- Mapped RegVelo order identical to SAVI order: **True**
- First up to 5 positional mismatches: none
- Per-gene velocity-magnitude-sum Spearman rho: -0.0597 (p=7.55e-03)

Interpretation: if the gene axes were permuted, the per-gene magnitude correlation would be near zero. The observed value does not support aligned axes.

## 2. Velocity-norm comparison and API provenance

| side | norm median | norm P99 | norm min | norm max |
|---|---|---|---|---|
| SAVI | 1.0596e+02 | 1.5159e+02 | 9.0514e+01 | 2.3492e+02 |
| RegVelo | 2.9332e+01 | 4.7333e+01 | 2.0000e+01 | 4.8963e+01 |

- RegVelo/SAVI norm ratio (median): 2.7681e-01
- RegVelo/SAVI norm ratio (P99): 3.1224e-01
- RegVelo velocity source: `model.get_velocity(adata, return_numpy=True)`; official API: True

Interpretation: a ratio ≪ 1e-2 suggests a scale/unit inconsistency or lack of convergence in the RegVelo fit; a ratio of order 1 suggests the two fields are on comparable scales but may point in unrelated directions.

## 3. scVelo dynamical convergence anchor

- SAVI vs scVelo cosine median: nan
- RegVelo vs scVelo cosine median: nan
- scVelo status: ran

Interpretation rubric (pre-registered):
- If SAVI-vs-scVelo is positive and RegVelo-vs-scVelo is not, the RegVelo side likely has a convergence/scale issue.
- If both are near zero, the velocity field may not be identifiable at this reduced scale; the benchmark would need to be expanded rather than interpreted as a method-level conclusion.

## 4. Composite verdict

- Gene-axis mismatch: NO
- Scale mismatch (>2 orders): NO
- RegVelo official API used: YES
- After excluding gene-axis and scale issues, median cosine remains -0.0958.



## 5. N5000 upgrade rerun (this execution)

### 5.1 Design

- cells: 5000 (cell_index_n5000, seed=20260914)
- genes: 2000 (HVG source `/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo/results/B1_regvelo_reference/_inputs_hvg2k_n500/hvg_genes.txt`)
- HVG gene hash (sha256 first 16): fcae0d19135a8a32

### 5.2 Wall-clock / peak resources

| method | wall-clock (s) | peak GPU (GB) | notes |
|---|---|---|---|
| SAVI L2+L3 | 297.5 | L2=0.00, L3=0.19 | production protocol K=10, bs=1024 |
| RegVelo GPU | 1701.2 | 4.93 | max_epochs=200, early_stopping |
| scVelo dynamical | — | — | status=scvelo_failed: setting an array element with a sequence. |

### 5.3 Pairwise per-cell metrics

Cosine similarity (median / P99 / min):

| pair | median | P99 | min |
|---|---|---|---|
| savi_regvelo | 0.0464 | 0.5546 | -0.5092 |
| savi_scvelo | nan | nan | nan |
| regvelo_scvelo | nan | nan | nan |

Relative L2 (median / P99 / min):

| pair | median | P99 | min |
|---|---|---|---|
| savi_regvelo | 2.0807 | 5.7871 | 0.9104 |
| savi_scvelo | nan | nan | nan |
| regvelo_scvelo | nan | nan | nan |


- SAVI–RegVelo: median cosine = 0.0464 → `reformulation changes field (limitation)`
- scVelo anchor failed or excluded; no external anchor available.

Pre-registered rubric: median cosine ≥0.9 strong agreement, 0.7–0.9 partial, <0.7 reformulation changes field.
**All numbers above are frozen pending human review.**


## 6. TF coverage check (read-only audit)

- HVG source: `results/B1_regvelo_reference/_inputs_hvg2k_n500/hvg_genes.txt`
- Unique HVG genes loaded: 2000 (file has no trailing newline, `wc -l` reports 1999)
- Production W contains 11 TFs with outgoing edges: Cebpb, Fos, Irf7, Jun, Nfatc1, Nfatc2, Nfkb1, Rela, Spi1, Stat1, Stat2
- TFs present in HVG2k: 4/11 — Cebpb, Jun, Nfkb1, Spi1
- TFs missing from HVG2k: 7/11 — Fos, Irf7, Nfatc1, Nfatc2, Rela, Stat1, Stat2
- All W TFs present in HVG2k: 4/11
- Verdict: **TF_COVERAGE=insufficient** (missing ≥3 TFs)
- Suggested gene set: HVG2k ∪ missing TFs = 2007 genes
- Suggested gene list written to: `results/B1_regvelo_reference/suggested_gene_set_hvg2k_plus_missing_tfs.txt`

Implication: the current HVG2k gene set is enriched for high-variance spliced counts but excludes the majority of the GRN's transcription-factor sources. Any velocity comparison that relies on GRN structure (SAVI/RegVelo) should either use the suggested TF-augmented set or interpret the HVG2k result as a variance-enriched subspace rather than a GRN-representative subspace.


## 7. TF-inclusive N5000 rerun (final round)

### 7.1 Design

- cells: 5000 (cell_index_n5000, seed=20260914)
- genes: 2007 (TF-inclusive set `/home/lab-zeng/research_storage/工作文件/AD_Regvelo优化策略/validation_Sparse_regvelo/results/B1_regvelo_reference/suggested_gene_set_hvg2k_plus_missing_tfs.txt`)
- gene hash (sha256 first 16): 5adf31a2663b18b8
- W nnz: 22066; TF coverage: 11/11

### 7.2 Wall-clock / peak resources

| method | wall-clock (s) | peak GPU (GB) | notes |
|---|---|---|---|
| SAVI L2+L3 | — | — | reused from previous attempt |
| RegVelo GPU | — | — | reused from previous attempt |
| scVelo dynamical | 69.2 | CPU | neighbors→moments(n_pcs=30)→recover_dynamics→velocity(mode='dynamical') |

### 7.3 Pairwise per-cell metrics

Cosine similarity (median / P99 / min):

| pair | median | P99 | min |
|---|---|---|---|
| savi_regvelo | 0.0727 | 0.7851 | -0.1111 |
| savi_scvelo | 0.2506 | 0.8008 | -0.5563 |
| regvelo_scvelo | 0.1692 | 0.6867 | -0.7462 |

Relative L2 (median / P99 / min):

| pair | median | P99 | min |
|---|---|---|---|
| savi_regvelo | 6.0771 | 205.4652 | 1.1999 |
| savi_scvelo | 1.1862 | 23.1031 | 0.6047 |
| regvelo_scvelo | 0.9901 | 1.2283 | 0.7476 |

### 7.4 Side-by-side with control (section 5)

| attribute | control (HVG2k) | TF-inclusive |
|---|---|---|
| genes | 2000 | 2007 |
| W nnz | 7996 | 22066 |
| TF coverage | 4/11 | 11/11 |
| SAVI–RegVelo cosine median | 0.0464 | 0.0727 |
| SAVI–scVelo cosine median | N/A | 0.2506 |
| RegVelo–scVelo cosine median | N/A | 0.1692 |
| scVelo status | scvelo_failed: setting an array element with a sequence. | ran |


- SAVI–RegVelo: median cosine = 0.0727 → `reformulation changes field (limitation)`
- SAVI–scVelo: median cosine = 0.2506 → `reformulation changes field (limitation)`
- RegVelo–scVelo: median cosine = 0.1692 → `reformulation changes field (limitation)`

Pre-registered rubric: median cosine ≥0.9 strong agreement, 0.7–0.9 partial, <0.7 reformulation changes field.
**All numbers above are frozen pending human review.**


## 8. Conclusion for manuscript

### 8.1 Three-layer story

1. **Scalability / identifiability layer.** The reference RegVelo implementation cannot instantiate the full 17,714-gene GRN at the production scale (OOM even at 200 cells). Reducing to 2,000 genes makes the benchmark computable but places it in a regime where the velocity field is poorly identifiable: the control 5,000-cell comparison yields a median cosine of only **0.0464** between SAVI and RegVelo. The benchmark therefore compares methods on a small, variance-selected subspace rather than on a biologically representative gene universe.

2. **TF-coverage hypothesis: refuted.** A natural conjecture was that the low agreement was an artifact of the HVG2k gene set missing most of the GRN's transcription-factor sources (only 4/11 TFs present). Augmenting the gene set to include all 11 TFs (2,007 genes, W nnz 22,066, TF coverage 11/11) changed the SAVI–RegVelo median cosine from **0.0464** to **0.0727**—a marginal improvement that remains far below the 0.7 reformulation threshold. **TF-coverage hypothesis refuted (control 0.0464 vs TF-inclusive 0.0727).** The disagreement is therefore not primarily a gene-coverage artifact.

3. **scVelo anchor / uncertainty motivation.** With the fixed dynamical pipeline, scVelo recovered velocities for only **112** of the 2,007 genes (5.6% finite entries). Its median cosine with SAVI and RegVelo was **0.251** and **0.169**, respectively—again well below any agreement threshold. This confirms that the three methods produce markedly different velocity fields on the same 5,000-cell subspace. It also provides empirical motivation for the uncertainty/ensemble strategy proposed elsewhere: when independent RNA-velocity formulations disagree at this level, a single consensus velocity should not be reported without quantifying method uncertainty.

### 8.2 Per-cell velocity-norm Spearman rank correlations

Computed on the TF-inclusive 5,000-cell subset (scVelo restricted to its 112 recovered genes):

| pair | Spearman rho | p-value | n cells |
|---|---|---|---|
| SAVI–RegVelo | -0.0860 | 1.10e-09 | 5000 |
| SAVI–scVelo | 0.0442 | 1.76e-03 | 5000 |
| RegVelo–scVelo | 0.0335 | 1.77e-02 | 5000 |

Interpretation: rank correlations on per-cell speed magnitudes are modest (<0.3 for all pairs), indicating that the methods not only disagree in direction (low cosine) but also do not agree on which cells are fast versus slow.

### 8.3 Status

All B1 numbers are frozen; status set to **human_review_B1_final**.


## 9. L2/L3 fidelity decomposition probe (v6.1)

We inspected the RegVelo 0.4.2 internal API to determine whether trained kinetic parameters can be exported and injected into the SAVI L3 propagator, which would separate L2 (kinetic-fit) from L3 (propagation) contributions to the SAVI–RegVelo divergence.

**Attempted API paths:** `REGVELOVI.setup_anndata`, `REGVELOVI(..., W=W_tensor, soft_constraint=True)`, `model.module.named_parameters()`, `model.module._get_rates()`, `model.module.v_encoder.beta_mean_unconstr / gamma_mean_unconstr`, `model.module.alpha_1_unconstr`, `model.module.decoder(...) -> px_rho`, `model.module.get_px(...)`, `model.get_velocity(...)`.

**Finding:** The public API exposes per-gene kinetic means (`gamma`, `beta`, `alpha_1`) via `_get_rates()`, but the velocity computation couples these with per-cell latent variables: `inference_outputs["beta"]` and `"gamma"` vary across cells, and the generative decoder predicts per-cell/gene latent-time factor `px_rho`. RegVelo velocity is `beta * mean_u(ind_t) - gamma * mean_s(ind_t)` where `(mean_u, mean_s)` are derived from a closed-form induction-time solution, not from observed Ms/Mu propagated by an ODE.

**Conclusion:** A clean cross-implementation L2/L3 decomposition is **blocked**. RegVelo's kinetic parameters are exportable, but they are not mappable to the static per-gene beta/gamma plus GRN-bias format required by SAVI L3. The reference implementation's kinetic parameters and propagation step are therefore not independently interchangeable with SAVI's. This structural non-equivalence should be added to the Limitations paragraph on reference-implementation comparison.

**Artifacts:** `results/B1_regvelo_reference/B1_fidelity_decomposition.{json,md}`.

