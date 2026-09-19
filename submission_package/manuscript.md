# SAVI: Scalable sparse analytic RNA velocity with numerical validation and uncertainty analysis

Wei Zeng^1,2

^1 Department of Neurology, The Third School of Clinical Medicine, Southern Medical University, Shenzhen 518000, China
^2 Department of Neurology, The Seventh Hospital of Wuhan (Binjiang Central Hospital of Renmin Hospital of Wuhan University), Wuhan 430060, China

Correspondence: Wei Zeng, ericleo_zeng@hotmail.com

---

## Abstract

**Motivation:** We developed SAVI, a sparse analytic implementation of GRN-constrained RNA velocity, and validated it on a 74,984-cell microglial atlas from a 5xFAD mouse model. Production-scale numerical audits remain under-reported.

**Results:** Outputs reproduce exactly, with CPU/CUDA velocity deviation 7.1e-6. Rate bound and gene content dominate velocity variation (cosine 0.386–0.80); GRN weighting preserves direction (cosine ≥0.9965) and segment refinement reduces residual ~28-fold. Fate probabilities are rank-stable (Spearman 1.00; 89% top-state agreement). Scaling is near-linear to 74,984 cells while peak GPU memory remained insensitive to cell count over the tested range, and the protocol transfers to dentate gyrus. Direct comparison with the reference implementation was precluded at production scale by memory limits. Reduced-scale comparisons showed substantial field differences; sparse implementation should not be treated as numerically equivalent to existing engines. 

**Conclusion:** This study demonstrates production-scale numerical validation and indicates that gene selection and rate bounds dominate uncertainty over discretization or prior weighting.

**Availability:** Engine source and validation records are at https://github.com/Ericleo-zeng/savi-sparse-regvelo; the dataset is controlled-access under institutional DUA.

**Contact:** Correspondence: Wei Zeng, ericleo_zeng@hotmail.com.

---

## 1 Introduction

RNA velocity adds directionality to snapshot single-cell data (La Manno *et al.*, 2018; Bergen *et al.*, 2020) and supplies a directed signal to fate-mapping frameworks such as CellRank (Lange *et al.*, 2022; Weiler *et al.*, 2024). Palantir (Setty *et al.*, 2019) provides a related probabilistic fate framework based on transcriptomic-state geometry rather than requiring RNA velocity as input. GRN-constrained velocity models such as RegVelo (Wang *et al.*, 2026) further regularize inference with regulatory priors, but coupled dynamical inference scales poorly with gene number. The engine examined here, SAVI (Sparse Analytic Velocity Inference; implementation: sparse_regvelo), was reconstructed for atlas-scale production. It uses a decoupled per-gene kinetic fit (L2), followed by segment-wise closed-form propagation of the full nonlinear system (L3), with no dense gene-by-gene learnable matrix, single-precision numerics and an analytic Jacobian.

We deliberately do **not** claim a new velocity method. However, the sparse analytic reformulation itself is a computational contribution: it replaces coupled inference with decoupled fitting and closed-form propagation, enabling GRN-constrained velocity at atlas scale. sparse_regvelo is an independent re-implementation of the RegVelo model class with reformulated numerics (§2.2bis); its role in this study is to provide a production-grade object whose outputs can be *validated*. This goal differs from recent cross-method benchmarks that rank tools across datasets (Luo *et al.*, 2026; Wu *et al.*, 2026): we audit a single engine under fixed configuration, hardware and input perturbations to bound the uncertainty it carries into downstream fate inference. This validation focus follows a simple observation: **scalability does not imply validity**. An engine can be architecturally sound yet fail in practice because override flags are silently ignored, reference checkpoints are assembled from mismatched artifacts or results depend on hyperparameters whose influence has not been measured. Downstream fate frameworks inherit uncertainty carried by the velocity layer.

We report a validation campaign on the production dataset (74,984 cells × 17,714 genes; 12 samples; three genotypes: control, 5xFAD and 5xFAD;Cd28-cKO; ages 3/6/8 months). The study makes five contributions: **(1) numerical validation** of stored production outputs against tightened references through convergence budgets, cross-hardware consistency and determinism; **(2) a sensitivity landscape** over engine hyperparameters and input choices, together with a fate-level control, separating hyperparameter effects (rate bound), curatorial effects (gene selection), prior-weight effects (GRN scale) and discretization effects (segment count); **(3) validation-driven debugging**, in which provenance instrumentation exposed a rate-bound override bug and a mis-labelled reference checkpoint, both fixed and re-verified; **(4) reproducibility evidence** from bit-exact reruns of both fitting levels and a quantified propagation-memory model; and **(5) a scaling and cross-dataset portability analysis** demonstrating computational feasibility and transferability of the validation protocol.

Every quantitative claim is traceable to timestamped logs, checkpoint metadata and comparison JSONs (Supplementary traceability table).

## 2 Materials and Methods

### 2.1 Data and production configuration

The production dataset comprised a 74,984-cell microglial atlas (three genotypes (control, 5xFAD, 5xFAD;Cd28-cKO) × three ages, 12 samples), integrated and annotated upstream. Engine inputs were spliced/unspliced moments (Ms, Mu; 74,984 × 17,714 float32), a 30-d scVI latent representation (Lopez et al., 2018) truncated to 10 dimensions and a regulatory prior W (17,714 × 17,714 CSR; 194,843 edges; 11 transcription factors; row = target). Production L3 used segment count K = 10, cell mini-batch 2048, Picard tolerance 1e-3, iteration budget 10 and the adaptive per-gene rate bound rate_max = max(20, 1.2 × P99.5(β, γ)) = 107.31. All propagation used single-precision numerics (FP32) under an 8 GB device-memory ceiling. K = 10 was retained as the production setting; §3.8 demonstrates directional neutrality over K = 10–40, indicating that segment count has little effect on velocity direction in this regime, although it contributes to numerical amplitude error relative to a tightened reference.

### 2.2 Engine overview

The engine comprises three levels. **L1** computes steady-state initial values and applies a per-gene gate that flags steady-state violations. **L2** fits per-gene kinetics (β, γ) through a decoupled procedure: an entropy-regularized subsampled diffusion-pseudotime ordering followed by a 10-iteration refinement stage over background/transition-state assignment thresholds under configurable rate bounds. **L3** propagates each cell over K pseudotime segments. Within each segment, a Picard fixed-point iteration applies a closed-form first-order-hold update; segments exceeding the residual threshold are routed to Krylov-exponential or full-dynamics fallbacks, and gated cells follow a full-dynamics path. Velocity is computed algebraically as v = β⊙u − γ⊙s. The Jacobian is analytic on GRN edges through the softplus derivative, and a per-gene bias closes the steady-state loop in closed form.

### 2.2bis Relationship to RegVelo

sparse_regvelo re-implements the RegVelo model class from scratch; it is not a modification of the reference implementation, and we do not claim line-level equivalence. Three layers must be distinguished: **(i) model class**—GRN-constrained spliced/unspliced kinetics with softplus regulation, shared with RegVelo; **(ii) numerical path**—reformulated, with a decoupled L2 fit and a closed-form segment-wise L3 propagator replacing the reference implementation's coupled inference under the red lines of §2.2 (FP32, no dense (G,G) learnables, analytic Jacobian); and **(iii) added components**—the L1 gate and routing layer, closed-form bias closure and the provenance/completeness instrumentation described in §2.3. Formal equivalence between the numerical paths is *not* established here and is listed as a limitation (§4); the validation objects are the production outputs of this engine under the stated configuration.

### 2.3 Validation protocol

We use *validation* in the empirical sense: verification of specified numerical properties—convergence, deterministic reproducibility, cross-hardware agreement and bounded sensitivity—not formal proof of model correctness or equivalence to the RegVelo reference implementation. Unless stated otherwise, sensitivity arms used a fixed 5,000-cell subset, K = 10, batch size 1024 and FP32, with the same adaptive rate-bound protocol as production. Velocity fields were compared by per-cell cosine similarity and relative L2. Pre-specified pass lines were cosine median > 0.99 with relative-L2 median < 0.05 (strict), supplemented by a directional line of cosine median > 0.95 for variable-importance questions. These thresholds were used as pre-specified engineering acceptance criteria rather than as universal biological validity thresholds. Relative-L2 was computed per cell as ||v_test − v_ref|| / max(||v_ref||, 10⁻¹²); cosine similarity was computed per cell as (v_test · v_ref) / (||v_test|| · ||v_ref||), with the same 10⁻¹² floor. All aggregations used the per-cell distribution; zero-norm vectors were protected by the floor and did not occur for the non-zero velocity fields in this study. The full protocol is archived as v1.0 (osf.io/pjvk8; registered 2026-09-17).

Three validation disciplines were applied throughout. **(i) Reference discipline:** parameter-level comparisons use only checkpoints refitted on identical data, never frozen-parameter composites. **(ii) Provenance discipline:** every checkpoint records requested versus actual rate bounds, mode and SHA-256 fingerprints of β, γ, W and metadata; every L3 output embeds its source-checkpoint lineage. **(iii) Completeness discipline:** comparison tools reject half-written outputs by requiring the L3 metadata file, which is written only after successful completion.

### 2.4 Closing experiments of this revision

**E1 (cross-hardware).** CPU and CUDA propagated the same 5,000 cells from the faithful adaptive reference checkpoint with identical batch size. **E2 (iteration attribution).** 500 cells were propagated at max_iter ∈ {10, 20, 30, 50} (K = 10, tol = 1e-3) against a tightened reference (K = 20, tol = 1e-6, 50 iterations), together with a K × tol decomposition grid. **E2c (tail grounding).** Per-cell velocity residuals from the E2 framework were cross-tabulated against cell-state annotations to test whether the persistent upper tail concentrated in specific biological subpopulations.

**E3 (input decomposition).** The 57-gene removal arm was evaluated under two conventions: moments recomputed after removal (renormalizing; arm V7b) and production moments column-restricted to the retained genes (arm V7a), thereby separating gene content from normalization. **E4 (fate-level control).** For each arm we built a velocity-informed transition kernel on the shared 10-dimensional latent space: a kNN graph (k = 30, Euclidean) with CellRank VelocityKernel weights c_ij = cos(v_i, s_j − s_i) and w_ij = softmax(c_ij/σ_i), σ_i = median_j|c_ij|, combined with a diffusion kernel T_c (Gaussian weights on squared latent distances, median bandwidth, max-symmetrized, row-normalized) as T = 0.8·T_v + 0.2·T_c, then row-normalized. Absorption targets were 30 latent-nearest cells per annotation-state centroid; per-cell fate vectors solve (I − Q)X = R by sparse float64 direct solution (SciPy spsolve). Agreement was quantified by row-wise Spearman correlation of the six-state fate vectors, with ties broken by state index; zero-variance rows were excluded from the reported NaN-robust median and minimum.

### 2.5 Reference implementation and scaling study

We compared SAVI against the official RegVelo 0.4.2 implementation (Wang *et al.*, 2026). The reference could not instantiate the full 17,714-gene GRN layer at production scale: it failed before the first epoch on a 200-cell smoke test with 11.47 GB of the 11.50 GB device allocated, and the failure recurred on a 5,000-cell subset reduced to the top 5,000 spliced-variance genes (HVG5k), where PyTorch had allocated 11.07 GB and only 192 MiB remained free. Because the dense GRN layer exceeds a ≤12 GB GPU, head-to-head velocity-field equivalence could not be established at full scale.

To audit the disagreement under computable conditions, we ran both implementations on a 5,000-cell subset of the microglia dataset. On a 2,000-gene HVG set, SAVI and RegVelo yielded a median per-cell cosine similarity of only 0.046. We tested whether this was an artifact of missing transcription factors in the HVG set by augmenting it to include all 11 TFs present in the GRN (2,007 genes, full TF coverage); the median cosine improved only marginally to 0.073. A scVelo dynamical anchor recovered velocities for 112 genes and agreed with SAVI and RegVelo at median cosines of 0.251 and 0.169, respectively. The three fields showed low pairwise directional agreement on the tested reduced gene space. Adding all 11 TFs produced only a modest increase in cosine similarity (0.046 to 0.073), suggesting that missing TFs alone do not explain the discrepancy in this reduced-space comparison; instead, it indicates that the three formulations produce materially different velocity fields on the same reduced subspace, providing an empirical motivation for quantifying method uncertainty rather than reporting a single consensus velocity. The reference implementation computes velocity through a closed-form induction-time solution, whereas SAVI propagates the kinetic system segment-wise by Picard iteration; these are distinct numerical solution strategies within the same model class; numerical agreement must therefore be established empirically rather than assumed.

SAVI L3 scaling was measured under fixed K = 10 and batch size 1,024 on the same 11.5 GB device. Cell scaling at G = 17,714 from 5,000 to 74,984 cells gave runtimes of 10.8 s, 20.8 s, 51.4 s, 113.4 s and 204.6 s—nearly linear in cell count—while peak GPU memory stayed at 6.52 GB, i.e., cell-count insensitive at this ceiling. Gene scaling at N = 5,000 from 5,000 to 17,714 genes gave runtimes of 1.4 s, 3.8 s and 10.1 s and memory rose with gene count. The A1#13 reference point (5,000 cells, 17,714 genes, K = 10, batch 1,024) peaked at 5.82 GB; its tightened counterpart (K = 20, batch 1,024) exceeded device memory and is noted only in the Fig. 6 caption.

scVelo runtimes on the same hardware (37.1 s at 5,000 cells and 106.3 s at 10,000 cells, deterministic model, no GRN constraint) are shown as a dynamical-modeling representative in Fig. 6; details are in Supplementary Table S2.

## 3 Results

### 3.1 Production outputs converge; the velocity tail is not explained by iteration budget (V1, E2)

We first audited the *stored* production solution by re-propagating sampled cells against a tightened reference (K × 2 = 20, tol = 1e-6, 50 iterations; converged in every batch). On 2,000 production-scale cells, s (median 3.14e-4, P99 3.61e-3) and u (median 8.02e-4, P99 6.19e-3) met the acceptance lines (median < 2e-3, P99 < 1e-2); velocity met the median line (1.17e-3) but not the P99 line (1.045e-2). At n = 5,000, the verdict was unchanged (v P99 1.17e-2, max 4.22e-2). The larger subset showed a slightly more pronounced upper tail, so the P99 exceedance was not confined to the smaller sample. The same audit passed on the subset-fitted reference (v P99 9.36e-3), and the production log corroborated convergence (74/74 batches below 1e-3, all closed-form route).

An iteration-budget sweep (E2) excluded non-convergence as the source: increasing max_iter from 10 to 50 produced indistinguishable velocity fields (v relative-L2 median 1.195e-3, P99 8.159e-3, max 6.55e-2; zero unconverged batches). A three-arm K/tolerance decomposition (E2b; all arms at 50 iterations against a K = 20, tol = 1e-6 reference) separated the error into components. The **median discrepancy was strongly controlled by segment count**, decreasing ~28-fold when K increased from 10 to 20 at fixed tolerance (1.195e-3 → 4.196e-5). By contrast, **P99 errors remained in a narrow range (6.769e-3–9.791e-3)**, whereas the **maximum error decreased from 6.55e-2 to 1.72e-2 when tolerance was tightened** from 1e-3 to 1e-6. The upper tail was therefore comparatively insensitive to K and iteration budget but not tolerance-independent. Median per-cell cosine remained ≥0.999999 in every arm, with minima 0.9993–0.9999, so the discrepancies were direction-preserving. **Verdict: the strict line fails on one tail statistic by 4.5% at production scale; the bulk discrepancy is dominated by segment discretization, the high-percentile tail responds only partially to the tested numerical controls, and field direction remains materially unchanged.**

The extreme residual tail showed a state-associated enrichment rather than a uniform distribution across annotations. Grounding per-cell residuals from all 5,000 subset cells against cell-state annotations (E2c) showed uniform per-state mean residuals (1.59–1.72e-3) but a state-structured extreme quantile. The top-1% highest-residual cells were 2.0-fold enriched in C1q_inflammatory (19/50 versus 9.4 expected; Fisher exact p = 1.5e-3, odds ratio 2.67; Bonferroni-significant across six states) and depleted in Homeostatic and DAM_like (0.65–0.67-fold), while Proliferating cells contributed none (0/50). These 19 cells were distributed across 9 of the 12 samples (maximum share 0.26; permutation p = 0.57), arguing against a single-sample artefact. The tail therefore concentrated in inflammatory-activated microglia. This identifies a state-associated numerical sensitivity pattern, but does not establish its mechanistic origin; SN-U4 did not resolve it.

### 3.2 Cross-hardware consistency is validated for the consumed quantity (V3, E1)

On the faithful reference checkpoint, CPU and CUDA propagations of the same 5,000 cells at identical batch size agreed to a velocity maximum relative deviation of 7.13e-6, P99 1.43e-6 and per-cell cosine minimum 1.0. Deviations in u and s were also bounded (u tail max 3.03e-4, s max 5.77e-6)—the signature of FP32 non-associativity and well inside every downstream budget in §3.1. The earlier WARN, which was recorded on the composite reference checkpoint but not quantified, is thereby closed as bounded cross-device numerical drift for the quantity actually consumed.

### 3.3 Determinism and two defects identified by provenance auditing

Both fitting levels were deterministic. Repeated L2 runs reproduced terminal statistics bit-for-bit (β median 3.91741, γ median 2.05488, max 107.306 across independent runs), and repeated L3 runs from one checkpoint gave max|Δv| = 0.0. This establishes within-configuration determinism; cross-hardware equivalence is assessed separately in §3.2.

Provenance auditing identified two defects. **T0.4 (fixed).** The L2 rate bound used 20.0 both as the argparse default and as a legitimate request, so the guard `rate_max != 20.0` silently routed an explicit `--rate-max 20` into the adaptive branch. The first run recorded requested = 20.0 but actual = 107.31; after a five-line fix (default → None; not-None guards; null-safe recording), the rerun recorded 20.0/20.0/manual. **T0.5 (documented, metadata repaired).** The subset reference checkpoint used by earlier validations was an assembly: per-gene parameters were copied verbatim from production (max|Δβ| = 0.0), while per-cell quantities were regenerated, but its metadata described a subset fit. We repaired the metadata and re-scoped V2 and V8 as numerical self-consistency checks.

### 3.4 The kinetic rate bound materially affects velocity direction (V5)

Because the production bound arose from an adaptive heuristic whereas an earlier request specified 20, we tested whether this discrepancy was documentary or numerical. Three L2 models—adaptive (107.31), manual 20 and manual 100—were refitted on identical data and propagated under identical L3 settings. Against the adaptive reference, rate_max = 20 yielded velocity cosine median 0.386 (min 0.188) and relative-L2 median 1.08; rate_max = 100 yielded cosine median 0.699 (min 0.513) and relative-L2 median 1.15. Both failed the strict lines, as did the near-boundary comparison (100 vs 107.31). **The provenance discrepancy was numerical, not documentary.** Velocity direction is *not* robust to this choice: sensitivity was substantial and non-uniform, and the adaptive coefficient (P99.5 × 1.2) was a consequential hyperparameter in the tested configuration. Production, which used adaptive 107.31 and reproduces bit-for-bit, is unaffected.

### 3.5 GRN weighting changes propagation while preserving median direction (V6/V6b)

Production W was uniform 1.0 over 194,843 edges, whereas the GRN supplied continuous importance scores (P50 = 1.75, P90 = 5.35). We evaluated two quantile-tiered variants: aggressive (1.0/0.5/0.25; total 82,809, −57%) and mild (1.0/0.75/0.5; 126,648, −35%). The mild and aggressive weighting schemes produced progressively larger amplitude deviations. L2 statistics were bit-identical across variants (β max|Δ| = 0.0; bias max|Δ| = 3.18), so the prior influenced results only through the closed-form bias and L3 dynamics. Against the adaptive reference, aggressive weighting gave cosine median 0.99650 (min 0.493) and relative-L2 median 8.86e-2; mild weighting gave 0.99865 (min 0.740) and 5.47e-2. Both failed the strict amplitude line. **Velocity direction is robust at the median level (per-cell minima 0.493 and 0.740); magnitude carries a 5–9% GRN-scale uncertainty.**

### 3.6 Removing cell-cycle genes reshapes the field; gene content—not normalization—is the driver (V7, E3)

We removed 57 cell-cycle genes (G 17,714 → 17,657; W 194,843 → 194,216) and refitted on identical cells. On common genes against the full-universe reference, the recomputed-moments arm (V7b) gave cosine median 0.802 (min 0.087) and relative-L2 median 0.61; the no-renormalization control (V7a) gave 0.680 (min 0.299) and 0.80. Both fell under the directional line. Because V7a retained production moments without recomputation yet produced the larger perturbation, moment recomputation does not explain the field change. Gene content is therefore a major determinant. In the affected subpopulation, the inferred velocity direction was substantially influenced by the cycling program.

### 3.7 Field-level perturbations need not become fate-level instability (E4)

We asked whether the §3.6 field change propagated to fate-level conclusions. Despite velocity cosine medians of 0.68–0.80, full versus no-cycle fate vectors had Spearman median 1.00, 89.2% top-state agreement and ≤0.18 class-level probability shift; the no-renormalization arm was similar (0.943, 83.5%). Fate changes were confined to the expected cycle-dominated minority. Removing velocity altogether through a diffusion-only kernel perturbed fates more (61.2% top-state agreement) than removing the 57 cycling genes, so removing the 57 cycling genes perturbed fate assignments less than removing velocity altogether.

### 3.8 Segment count is directionally neutral (V8)

Using K = 10, 20 and 40 from a common checkpoint (frozen-parameter self-consistency scope), cosine medians were 0.99999 for K = 10 and 20 against K = 40 (minima 0.998; relative-L2 medians ≈1.1e-3). A faithful-checkpoint comparison of K = 20 versus K = 10 gave a cosine median of 1.00000 (min 0.99965). **K is directionally neutral** in this regime, whereas increasing K from 10 to 20 materially reduces the numerical residual relative to the tightened reference.

### 3.9 Decomposition of sensitivity sources across engine and input choices

Across the tested arms (Fig. 2), the rate bound produced the largest directional deviation (cosine 0.386–0.699), but it is a model hyperparameter rather than a biological input. Among curatorial/input choices, gene-content restriction was the most consequential (cosine 0.68–0.80 after removing 57 cell-cycle genes), while GRN weighting changed amplitude (5–9%) but left median direction intact (cosine ≥0.9965) and segment count was directionally neutral (cosine ≈1.0). Direction was robust to GRN weighting and segment count, but not to the rate bound or gene-content restriction in cycle-dominated cells.

### 3.10 Cross-dataset portability (dentate gyrus)

To test portability, we applied the same L2/L3 protocol to an independent 18,213-cell dentate gyrus neurogenesis dataset (GSE95753; 27,998 genes raw). After restricting to the 13,335 genes shared with the main universe, we built a regulatory prior from the raw TRRUST v2 mouse database (458 transcription factors, 2,866 edges; 1.5% of the main-atlas edge count). Because no curated scVI latent was available, we used the first 10 principal components of a 30-component PCA; this deviation is declared.

L2 adaptive fitting completed in 4,914 s. Production L3 (K = 10, batch 1,024) ran in 22.0 s with a peak GPU memory of 4.37 GB. Against a tightened reference (K = 20, tolerance 1e-6, 50 iterations; batch size 512 after a red-line retry; Supplementary Note SN-B3), the relative-L2 residual was median 4.77e-4, P99 1.32e-2 and max 3.04e-2, cosine minimum 0.99964. CPU and CUDA propagations agreed to relative-L2 median 3.75e-4 and maximum 3.13e-2, with 507 of 18,213 cells above the 1e-2 tail line; cosine minimum 0.99961. Under identical configurations and routing, the tail is consistent with the FP32 non-associativity pattern seen on the main atlas, not a cross-dataset bug.

Absolute residuals are not directly comparable to the main-atlas audit because the TRRUST prior is far sparser (2,866 versus 194,843 edges) and the latent representation differs. The transferable conclusion is that the validation disciplines—convergence, cross-hardware consistency and bounded sensitivity—hold on an independent dataset.

## 4 Discussion

**Principal findings.** Strict lines were failed by three sources: (i) one tail statistic 4.5% over line, traced to segment discretization and a high-percentile tail only partially controlled numerically (§3.1); (ii) the rate-bound hyperparameter (§3.4); and (iii) gene content (§3.6). Each became a quantified caveat; we find no evidence invalidating the stored outputs, which reproduce bit-for-bit and agree cross-hardware to 7.1e-6.

**A persistent tail with unresolved mechanism.** The extreme residual tail was enriched in C1q_inflammatory cells (§3.1), although the mechanistic basis of this enrichment remains unresolved; a per-cell stiffness diagnostic did not resolve it after accounting for proximity to steady state.

**A four-quadrant summary** (Fig. 2) summarizes: segment count is directionally stable but amplitude-sensitive; rate bound is numerically sensitive; gene universe is a directionally destabilizing input choice; GRN scale leaves direction stable but perturbs magnitude; provenance is reproducible only with instrumentation.

**From protocol to practice.** The three validation disciplines in §2.3 were each motivated by a concrete failure mode: reference discipline by the composite checkpoint (T0.5), provenance discipline by the silent rate-bound override (T0.4), and completeness discipline by the risk that crashed runs leave half-written outputs. They are offered as transferable guidance for auditing production single-cell engines.

**Interpretation for downstream fate mapping.** For velocity-weighted transition kernels, GRN weighting and segment discretization introduced little directional variation, while the rate bound remained consequential. Magnitude-sensitive analyses inherit a bounded 5–9% GRN-scale uncertainty. Fate vectors nevertheless remained rank-stable for typical cells, with redistribution confined to a cycle-dominated minority, whereas velocity-free baselines differed more (top-state agreement 0.61 versus 0.89). The dominant caveat is biological: the field was strongly driven by 57 cell-cycle genes, a property of gene content, not preprocessing. These findings are internal to the tested fate pipeline and must be re-established for alternatives.

**Limitations.** Sensitivity arms ran at N = 5,000; direction conclusions are supported by a production-parameter validation run at subset scale (§3.1, n = 5,000) and the full-atlas convergence audit (74,984 cells). Formal equivalence to the RegVelo reference implementation is not established (§2.2bis). V2 and V8 predate the reference-checkpoint repair and are reported as self-consistency; V1's tail is reported against a strict 10×-tolerance line that the K-gap suggests is conservative. Memory laws were measured on an 11.5 GB device: L3 peaks at 5.82 GB at (5,000 cells, 17,714 genes, K = 10, batch 1024) and scales with K × batch × genes. The fate-level control (§3.7) was run at subset scale. Findings are conditional on one atlas.

The head-to-head comparison in §2.5 is necessarily performed on a reduced gene set (≈2,000 genes versus 17,714 in production) because the reference RegVelo implementation runs out of memory on the full GRN even at 200 cells, and the failure recurred at HVG5k. This reduced-scale benchmark may not be representative of method behavior at full scale. Moreover, the independent scVelo dynamical anchor recovered velocities for only 112 genes, leaving most of the gene space unanchored. Readers should therefore interpret these cosine values as evidence of substantial disagreement under the tested reduced-space conditions, rather than as estimates of full-scale method disagreement. Direct export of RegVelo kinetic parameters into SAVI L3 is not supported by the public API, so a cross-implementation L2/L3 fidelity decomposition is not feasible. veloVI was excluded because the environment prohibits pip installs. The dentate-gyrus arm uses a much sparser TRRUST-derived prior (2,866 versus 194,843 edges) and a PCA-derived latent; its residuals are comparable only at the numerical-discipline level, not as a biological replication.

**Future work.** Extending the fate-level control in §3.7 to the full atlas and re-establishing it for each consuming pipeline remains future work. Cycle-score regression or cycle-gene exclusion should be considered when applying velocity-informed fate analyses to proliferating systems. Because K = 20 collapsed the median component at negligible directional cost (§3.8), a higher production K is a candidate for the next routine rerun.

## 5 Conclusion

We validated the production outputs of a sparse RNA-velocity engine at atlas scale, quantified its sensitivity landscape, and showed that rigorous validation converts silent defects and ambiguous failures into bounded, documented caveats. The velocity layer is reproducible, directionally robust to prior weighting and discretization, and carries quantified uncertainty budgets; these results define the uncertainty carried into downstream fate inference, where gene-universe perturbations proved fate-stable in this system (§3.7) and must be re-established per consuming pipeline. Among the curatorial perturbations tested, gene selection produced the largest directional change, and the rate bound is the largest hyperparameter-driven source of variation among those tested; both are bounded caveats.

## Data availability

Engine source, validation harness, run scripts and validation records are available at https://github.com/Ericleo-zeng/savi-sparse-regvelo; the dataset is controlled-access within the institutional domain under a data-use agreement. The dataset is from a 5xFAD/5xFAD;Cd28-cKO mouse model of Alzheimer's disease (three genotypes, three ages, 12 samples; 74,984 microglia); derived validation inputs (subset arrays, GRN prior, manifests) are deposited where governance permits.

The full number-to-source traceability table is provided as Supplementary Material and in the validation repository.

## Funding and conflict of interest

The authors received no specific funding for this work. The authors declare no competing interests.

## References

La Manno G, et al. RNA velocity of single cells. *Nature* 2018;560:494–498.
Bergen V, et al. Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology* 2020;38:1408–1414.
Lange M, et al. CellRank for directed single-cell fate mapping. *Nature Methods* 2022;19:159–170.
Weiler P, et al. CellRank 2: unified fate mapping in multiview single-cell data. *Nature Methods* 2024;21:1196–1205.
Setty M, et al. Characterization of cell fate probabilities in single-cell data with Palantir. *Nature Biotechnology* 2019;37:451–460.
Lopez R, et al. Deep generative modeling for single-cell transcriptomics. *Nature Methods* 2018;15:1053–1058.
Luo Y, Ren J, Yang Q, You Z, Zhou Y, Qin Q, Li Q. Benchmarking RNA velocity methods across 17 independent studies. *Cell Reports Methods* 2026;6(4):101367. DOI: 10.1016/j.crmeth.2026.101367.
Wu Y, Kong C, Liao X, Lin Z, Sun X, Liu J. Comprehensive benchmarking of RNA velocity methods across single-cell datasets. *Genome Biology* 2026;27:242. DOI: 10.1186/s13059-026-04182-z.
Wang W, Hu Z, Weiler P, Mayes S, Lange M, Fountain DM, Haug JO, Wang J, Xue Z, Sauka-Spengler T, Theis FJ. RegVelo: Gene-regulatory-informed dynamics of single cells. *Cell* 2026;189(12):3773–3800.e44. DOI: 10.1016/j.cell.2026.04.022.

---

## Figure Legends

**Figure 1. SAVI engine architecture and validation disciplines.** The three-level pipeline (L1 gate, L2 per-gene kinetic fit, L3 closed-form propagation) and the three validation disciplines (Reference, Provenance, Completeness).

**Figure 2. Sensitivity landscape across engine and input choices.** Perturbation arms plotted as median cosine versus median relative-L2 against the faithful reference, with strict thresholds. Segment count passes; GRN weighting preserves direction; rate bound and gene restriction fail.

**Figure 3. E2 iteration sweep and K × tolerance decomposition.** Overlapping iteration budgets exclude non-convergence; median residual collapses ~28-fold from K=10 to 20; P99 is stable and max is tolerance-sensitive. Median per-cell cosine remained ≥0.999999 across arms, with lower cell-level minima.

**Figure 4. E4 fate-level control of gene-universe perturbations.** Top-state agreement and Spearman correlation of fate vectors after removing 57 cycle genes. Fate probabilities remain rank-stable despite field-level cosine 0.68–0.80.

**Figure 5. E2c cell-state distribution of top-1% residual cells.** C1q_inflammatory cells are 2.0-fold enriched (OR=2.67, p=1.5×10⁻³), indicating a state-structured tail.

**Figure 6. SAVI L3 scaling curves.** Dual-panel plot of runtime versus cell count and gene count, with dual-axis GPU memory; scVelo dynamical (no GRN constraint) is shown as a reference. The A1#13 memory point (5,000 cells, 17,714 genes, K = 10, batch 1,024; 5.82 GB) and its K = 20 OOM counterpart are annotated.
