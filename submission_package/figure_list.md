# Figure list

## fig1_architecture_disciplines

- Script: `fig1_architecture_disciplines.py`

- Raster: `fig1_architecture_disciplines.png`

- Vector/PDF: `fig1_architecture_disciplines.pdf`

- **Caption:** SAVI engine architecture and validation disciplines. The three-level pipeline (L1 gate, L2 per-gene kinetic fit, L3 closed-form propagation) and the three validation disciplines (Reference, Provenance, Completeness).

## fig2_dominance_landscape

- Script: `fig2_dominance_landscape.py`

- Raster: `fig2_dominance_landscape.png`

- Vector/PDF: `fig2_dominance_landscape.pdf`

- **Caption:** Dominance landscape of engine and input choices. Perturbation arms plotted as median cosine versus median relative-L2 against the faithful reference, with strict thresholds. Segment count passes; GRN weighting preserves direction; rate bound and gene restriction fail.

## fig3_e2_residual_decomposition

- Script: `fig3_e2_residual_decomposition.py`

- Raster: `fig3_e2_residual_decomposition.png`

- Vector/PDF: `fig3_e2_residual_decomposition.pdf`

- **Caption:** E2 iteration sweep and K × tolerance decomposition. Overlapping iteration budgets exclude non-convergence; median residual collapses ~28-fold from K=10 to 20; P99 is stable and max is tolerance-sensitive. Direction is preserved (cosine ≥0.999999).

## fig4_e4_fate_control

- Script: `fig4_e4_fate_control.py`

- Raster: `fig4_e4_fate_control.png`

- Vector/PDF: `fig4_e4_fate_control.pdf`

- **Caption:** E4 fate-level control of gene-universe perturbations. Top-state agreement and Spearman correlation of fate vectors after removing 57 cycle genes. Fate probabilities remain rank-stable despite field-level cosine 0.68–0.80.

## fig5_e2c_state_distribution

- Script: `fig5_e2c_state_distribution.py`

- Raster: `fig5_e2c_state_distribution.png`

- Vector/PDF: `fig5_e2c_state_distribution.pdf`

- **Caption:** E2c cell-state distribution of top-1% residual cells. C1q_inflammatory cells are 2.0-fold enriched (OR=2.67, p=1.5×10⁻³), indicating a state-structured tail.

## fig6_scaling_curve

- Script: `fig6_scaling_curve.py`

- Raster: `fig6_scaling_curve.png`

- Vector/PDF: `fig6_scaling_curve.pdf`

- **Caption:** SAVI L3 scaling curves. Runtime versus cell count and gene count, dual-axis GPU memory footprint, and reference points for scVelo and the A1#13 memory benchmark.
