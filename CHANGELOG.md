# Changelog

## v1.0.0 — 2026-09-20
Initial public release accompanying the SAVI manuscript (submitted).

- Sparse RegVelo engine: closed-form sparse kernel (`closed_form_kernel.py`),
  decoupled velocity engine, Krylov decay-spectrum utilities.
- Validation harness: B1 RegVelo head-to-head reference; B2 scaling
  (gene/cell axes); B3 second-dataset validation; V-series numerical
  certification (ODE residual, batch invariance, CPU/GPU consistency,
  rate-max, GRN tiers, no-cycling / no-renorm ablations, K-sensitivity).
- Submission package: manuscript (md/pdf/docx) and supplementary notes.
- CI workflow template (`.github` sources) and Dockerfile.

Earlier internal development history is intentionally not part of this
public repository.
