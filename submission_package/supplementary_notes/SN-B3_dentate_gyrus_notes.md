# Supplementary Note SN-B3 — Dentate gyrus cross-dataset portability arm

This note documents the secondary cross-dataset validation reported in §3.10. The goal was to test whether the numerical validation disciplines (convergence, cross-hardware consistency, bounded sensitivity) transfer to an independent dataset and a sparser, uncurated regulatory prior, not to replicate the biological conclusions of the main atlas.

## Dataset

- **Accession:** GSE95753 — “Transcriptome analysis of single cells from the mouse dentate gyrus” (verified via NCBI eUtils).
- **Raw object:** `DentateGyrus.loom`.
- **Cells after loading:** 18,213.
- **Raw genes:** 27,998.
- **Genes overlapping the main microglia universe (G = 17,714):** 13,335 (75.3% of the main universe; 47.6% of the dentate-gyrus gene set).

## Adaptations and declared deviations

1. **Latent representation.** No curated scVI latent was available for this dataset. We used the first 10 principal components of a 30-component PCA computed on the spliced counts. This is a declared deviation from the main-atlas protocol.
2. **Regulatory prior.** We built W directly from the raw TRRUST v2 mouse database (458 transcription factors, 2,866 edges among the 13,335 shared genes). The resulting prior is far sparser than the main-atlas prior (2,866 versus 194,843 edges; 1.5% of the main-atlas edge count).
3. **Input dimensions.** Propagation used N = 18,213 cells and G = 13,335 genes.

## L2 kinetic fitting

- **Runtime:** 4,914.3 s.
- **Peak GPU memory delta:** 0.0 GB (the L2 routine in this configuration did not allocate additional GPU memory beyond baseline).
- The adaptive per-gene rate bound was applied in the same way as the main atlas.

## L3 propagation results

### Production settings

| Setting | Value |
|---|---|
| K | 10 |
| Tolerance | 1e-3 |
| Max iterations | 10 |
| Batch size | 1,024 |
| Runtime | 22.0 s |
| Peak GPU memory | 4.37 GB |

### Tightened reference

| Setting | Value |
|---|---|
| K | 20 |
| Tolerance | 1e-6 |
| Max iterations | 50 |
| Batch size | 512 |
| Runtime | 68.61 s |
| Peak GPU memory | 4.10 GB |

The tightened reference required a red-line OOM retry: batch size 1,024 exceeded the 8 GB internal device ceiling (8.19 GB ≥ 8 GB), so the run was repeated at batch size 512. Batch size does not affect per-cell Picard propagation numerics, but the deviation is declared.

### Residual versus tightened reference

| Metric | Median | P99 | Min | Max |
|---|---:|---:|---:|---:|
| Relative L2 | 4.77e-4 | 1.32e-2 | 9.20e-5 | 3.04e-2 |
| Cosine similarity | 0.9999999 | 0.99999999 | 0.99964 | 0.999999996 |

## CPU/CUDA cross-hardware consistency

Both CPU and CUDA runs used identical configurations (K = 10, tol = 1e-3, max_iter = 10, batch = 1,024). All 18 batches routed to the `closed_form_foh` path on both devices, with zero route mismatches.

| Metric | Median | P99 | Min | Max |
|---|---:|---:|---:|---:|
| Relative L2 | 3.75e-4 | 1.36e-2 | 1.94e-6 | 3.13e-2 |
| Cosine similarity | 0.9999999 | 0.9999999999 | 0.99961 | 0.999999999998 |

- 5,362 / 18,213 cells exceeded the 1e-3 relative-L2 line.
- 507 / 18,213 cells exceeded the 1e-2 tail line.
- The worst cell (index 15,277) had relative-L2 = 3.13e-2 and cosine = 0.99961.

## Interpretation

The CPU/CUDA deviation pattern on the dentate-gyrus dataset is consistent with the FP32 non-associativity tail observed on the main atlas (median ≈1e-4, max ≈3e-2, cosine minima ≈0.9996), not with a cross-dataset configuration bug. The L3 propagation therefore remained numerically disciplined under the alternate prior and latent representation.

Because the TRRUST prior is much sparser and the latent representation differs, **absolute residual values in this arm are not directly comparable to the main-atlas audit**. The transferable conclusion is that the validation disciplines—convergence, cross-hardware consistency and bounded sensitivity—hold on an independent dataset.

## Source files

- Cross-dataset harness: `B3_second_dataset_finalize.py`, `B3_second_dataset_continue.py`
- Diagnosis: `results/B3_dataset2/B3_diagnosis.md`
- Raw result JSON: `results/B3_dataset2/B3_result.json`
