# SAVI: Sparse Analytic RNA Velocity with Numerical Validation

SAVI (Sparse Analytic Velocity Inference) is a sparse analytic implementation of
GRN-constrained RNA velocity with a systematic numerical-validation and
uncertainty-decomposition framework, validated on a 74,984-cell microglia atlas
from a mouse 5xFAD/Cd28-cKO study (Ayata et al., 2025; GEO: GSE296768).

## Repository layout
- `sparse_regvelo/` — engine source (L1 gate, decoupled L2 kinetic fit,
  closed-form L3 propagator) + tests
- `results/` — validation records: V1–V8, E2/E2c/E4, B1 (RegVelo head-to-head
  + diagnosis), B2 (scaling), B3 (portability)
- `fig1`–`fig6_*` — figures with generation scripts
- `protocol_v1.0.pdf` — pre-specified analysis protocol (2026-09-14)
- `00_*`–`12_*` scripts + `config.py` — validation harness
- `submission_package/` — manuscript and supplementary notes
- `claim_evidence_map.tsv` — claim-to-evidence traceability

## Reproduce
Configure paths in `config.py`, then run scripts per `protocol_v1.0.pdf`.

## Data
Validated on the 74,984-cell microglia atlas derived from Ayata et al. (2025;
GEO: GSE296768). No raw expression matrices are included; derived validation
inputs are provided under `results/`.

## Links
- Analysis protocol (OSF): https://osf.io/pjvk8
- Manuscript: `submission_package/manuscript.md`

## Cite
- SAVI implementation and validation: this repository and the accompanying
  manuscript (submitted).
- RegVelo model class: Wang et al., Cell 2026.
- scVelo (dynamical reference): Bergen et al., Nat Biotechnol 2020.
