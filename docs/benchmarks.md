# Benchmarks: metaTF-py implementations vs R baselines

All numbers are measured by the campaign's blind benchmark harnesses
(`runs/round2`, `runs/round3`, `runs/round4` `experiment_results.json`;
R baselines from `runs/round1/baseline_metrics.jsonl`, produced by
`bench/round1_runner.R` on the same host).

- **Timing protocol**: 1 untimed warmup + 3 timed repeats, median
  (`time.perf_counter`); exceptions: rust_genie3 parity = 1 timed
  (deterministic), fast t6/t16 = 1 timed each.
- **Inputs**: smoke tier = `test_exp_data` 1000 genes × 154 cells;
  example tier = `mouse_HSC_formation_expression` (larger); the 300-gene
  variants are the smoke matrix's first 300 genes. Network = mm_blood
  (development) or mm_pantissue (**held out** — never used by candidates
  before its benchmark).
- **Gates**: numeric_port = spearman(flat) ≥ 0.995 AND per-cell top-100
  flip ≤ 1%; rank_port = spearman ≥ 0.95 AND per-target top-50 overlap
  ≥ 0.7; genie3_parity_bit = max|Δ| ≤ 1e-7 vs the R single-core reference.
- Host: 32 hardware threads; Python 3.11.15, numpy 2.2.6, scipy 1.16.3,
  numba 0.62.1; R 4.3.3 with GENIE3 1.24.0 / GSVA 1.50.5 / viper / AUCell /
  ppcor / metaTF 0.1.1 sources.

## Winners shipped in metatf 0.1.0

| method (layer)         | tier (network)                  | R baseline (s) | metatf (s) | speed-up | parity vs R                          | gate          |
|------------------------|---------------------------------|---------------:|-----------:|---------:|--------------------------------------|---------------|
| viper aREA u (py)      | smoke, mm_blood                 | 0.193          | 0.084      | 2.3×     | ρ=1−2e-13, maxΔ 3.2e-14              | PASS          |
| viper aREA w (py)      | smoke, mm_blood                 | 0.182          | 0.077      | 2.4×     | ρ 0.99996, maxΔ 0.25 (R dup-edge self-inconsistency) | PASS |
| viper aREA u (py)      | example, mm_blood               | 2.254          | 0.381      | 5.9×     | ρ = 1.0, maxΔ 1.3e-12                | PASS          |
| viper aREA w (py)      | example, mm_blood               | 1.561          | 0.390      | 4.0×     | ρ = 1−1e-16, maxΔ 2.0e-12            | PASS          |
| viper aREA u (py)      | example, **mm_pantissue (held out)** | 2.257     | 0.362      | 6.2×     | ρ = 1.0, maxΔ 1.4e-12                | PASS          |
| viper aREA w (py)      | example, **mm_pantissue (held out)** | 2.241     | 0.356      | 6.3×     | ρ = 1−1e-16, maxΔ 1.4e-12            | PASS          |
| aucell (py)            | smoke, mm_blood                 | 2.677          | 0.043      | 62.3×    | ρ 0.9963, maxΔ 0.0296, same 14 sets  | PASS          |
| aucell (py)            | example, mm_blood               | 6.286          | 0.438      | 14.3×    | ρ 0.99999837                         | PASS          |
| aucell (py)            | example, **mm_pantissue (held out)** | 5.878     | 0.374      | 15.7×    | ρ 0.99999827                         | PASS          |
| gsva Poisson (py)      | smoke, mm_blood                 | 3.741          | 2.145      | 1.7×     | ρ 0.99999999999999, maxΔ 5.6e-16     | PASS          |
| gsva Poisson (py)      | example, mm_blood               | 29.904         | 26.143     | 1.1×     | ρ 0.9999999997, maxΔ 1.5e-4 (14k×614 cells) | PASS |
| ulm subset (py)        | —                               | —              | 0.29       | —        | bit-identical to the round-2 port    | (identity)    |
| ulm full_axis (py)     | example, mm_blood               | —              | 0.388      | —        | vs decoupler 2.2.0 run_ulm: maxΔ 5.4e-12 (t), 8.1e-13 (p) | closure PASS |
| pcor (rust)            | smoke 1000 genes                | 1.298          | 0.049      | 26.5×    | maxΔ 2.6e-14 (full 1000×1000)        | PASS          |
| pcor (rust)            | 14,169 genes (perf only)        | —              | 1.832      | —        | no R reference (R O(G³) infeasible)  | perf only     |
| puic (rust)            | 300 genes                       | 3.159          | 0.018      | 178.9×   | ρ 0.9999975, maxΔ 0.02               | PASS          |
| puic (rust)            | 1000 genes                      | 25.286         | 0.119      | 213.0×   | ρ 0.9999985, maxΔ 0.016              | PASS          |
| genie3 parity (rust)   | 300 genes, set.seed(1), c1      | 112.861        | 67.789     | 1.7×     | maxΔ 4.16e-16; top50 overlap 1.0     | PASS (bit)    |
| genie3 fast (rust)     | 1000 genes vs 6-core R ref      | 135.227        | 77.2 (t6) / 37.3 (t16) | 1.75× / 3.6× | ρ 0.824, top50 0.804 — cross-stream envelope (R vs itself single↔6-core: 0.93) | rank gate vs 6-core ref: not met (by design; see semantics.md §6) |
| sincerities (py+numba) | 300 genes, 6 time points        | 14.760         | 29.786     | **0.50× (slower than R)** | ρ 0.9584, top50 0.945          | PASS (rank)   |

Notes:
- viper smoke-tier speed-ups are modest because R's own runtime is 0.2 s
  (fixed overhead dominates); the example tier shows the 4–6× asymptotic
  rate.
- pcor NumPy fallback timing (same host): ~3 s on the 1000-gene smoke
  (rankdata + cov + LAPACK pinv) — slower than R's 1.3 s on this input,
  identical numbers to 1e-14. The Rust backend is the fast path.
- SINCERITIES is shipped **slower than R** (disclosed): the numba port
  trades ~2× runtime for removal of the R + glmnet + kSamples dependency
  chain and run-to-run determinism (R's cvFolds is unseeded). Bit-identical
  to the pure-Python round-3 port; rank-parity vs R holds.

## Superseded earlier rounds (for the record)

| candidate (round)       | verdict                                                        |
|-------------------------|----------------------------------------------------------------|
| rust_pcor (r2)          | REJECTED: inverted correlation-of-ranks, not covariance (ρ 0.256 vs R); import NameError under stale sys.modules |
| py_viper_area (r2)      | REJECTED: wrong eset.filter + plotting position (ρ 0.88–0.98)  |
| py_aucell (r2)          | SUPERSEDED: kept sets at exactly 80% missing (strict `>` boundary) |
| py_ulm (r2)             | kept as the `subset` semantics path inside `py_ulm_dual` verbatim |
| py_genie3 (r3, pure py) | REJECTED: floor(sqrt) mtry bug; 153 s, rank gate not met        |
| py_sincerities (r3)     | SUPERSEDED by the bit-identical numba build (523 s → 30 s)      |
