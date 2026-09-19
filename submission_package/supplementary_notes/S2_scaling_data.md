# Supplementary Table S2 — SAVI L3 scaling data and reference runtimes

This note supports Fig. 6 and §2.5 of the main text. All SAVI L3 timings were measured end-to-end on the same NVIDIA GeForce RTX 5070 Ti Laptop GPU (11.50 GB VRAM) under the production engine configuration: single-precision (FP32), K = 10 segments, Picard tolerance 1e-3, iteration budget 10, adaptive per-gene rate bound. The production gene universe is G = 17,714.

## SAVI L3 cell scaling (fixed G = 17,714, batch size = 1,024)

| Cells (N) | Genes (G) | Batch size | Runtime (s) | Peak GPU memory (GB) | Status |
|---|---:|---:|---:|---:|---|
| 5,000 | 17,714 | 1,024 | 10.831 | 6.524 | success |
| 10,000 | 17,714 | 1,024 | 20.843 | 6.524 | success |
| 25,000 | 17,714 | 1,024 | 51.425 | 6.524 | success |
| 50,000 | 17,714 | 1,024 | 113.351 | 6.524 | success |
| 74,984 | 17,714 | 1,024 | 204.563 | 6.524 | success |

Memory was insensitive to cell count over the tested range; runtime increased approximately linearly with N (R² ≈ 0.9999 versus a straight line through the origin).

## SAVI L3 gene scaling (fixed N = 5,000, batch size = 1,024)

| Cells (N) | Genes (G) | Batch size | Runtime (s) | Peak GPU memory (GB) | Status |
|---|---:|---:|---:|---:|---|
| 5,000 | 5,000 | 1,024 | 1.392 | 1.991 | success |
| 5,000 | 10,000 | 1,024 | 3.823 | 3.763 | success |
| 5,000 | 17,714 | 1,024 | 10.068 | 6.524 | success |

Gene scaling increased both runtime and memory.

## Reference memory benchmarks

| Benchmark | N | G | K | Batch size | Peak GPU memory (GB) | Status | Note |
|---|---:|---:|---:|---:|---:|---|---|
| A1#13 | 5,000 | 17,714 | 10 | 1,024 | 5.82 | success | Main-text reference point used in Fig. 6 |
| A1#13 tightened | 5,000 | 17,714 | 20 | 1,024 | >11.50 | OOM | Exceeded device capacity; omitted from plot |

## scVelo dynamical-model reference runtimes (CPU, no GRN constraint)

| Cells (N) | Genes (G) | Runtime (s) | Peak CPU memory (GB) | Status |
|---|---:|---:|---:|---|
| 5,000 | 17,714 | 37.055 | 2.126 | success |
| 10,000 | 17,714 | 106.273 | 4.251 | success |

These runs used scVelo’s deterministic dynamical model (`scv.tl.recover_dynamics` followed by `scv.tl.velocity(mode='dynamical')`) and are shown in Fig. 6 only as a dynamical-modeling comparator. They do not enforce the GRN prior used by SAVI, so the numbers are not intended as a direct method-equivalence benchmark.

## Measurement details

- SAVI timestamps were recorded immediately before and after the L3 propagation call; they exclude L2 fitting and I/O.
- GPU memory was sampled via `torch.cuda.max_memory_allocated()` and converted to GiB (÷ 1024³).
- scVelo memory was sampled via `tracemalloc` peak resident RSS on the CPU.
- Software versions: `regvelo==0.4.2`, `scvelo==0.3.4`, `torch==2.11.0+cu128`, `numpy==2.4.2`, Python 3.12.

## Source files

- Scaling harness: `B2_scaling_collect.py`
- Raw result JSON: `results/B2_scaling/B2_result.json`
- Figure script: `fig6_scaling_curve.py`
