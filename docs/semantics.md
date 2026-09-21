# Semantics: R → Python/Rust mapping and divergence catalogue

This package reimplements the exact metaTF R call chain. The table maps the
upstream R entry points to this package; the sections after it catalogue every
place where Python/Rust behaviour is *defined differently from R* (or where R
itself is inconsistent). Everything here is auditable in
`docs/manifests/<candidate>/MANIFEST.json` and reproduced by the test suite.

## R → Python mapping

| metaTF (R)                                    | metatf (this package)                                | winner (round)                 |
|-----------------------------------------------|------------------------------------------------------|--------------------------------|
| `inferWeightExpMat(method="pcor")` (ppcor::pcor spearman `$estimate`) | `infer_grn(exp, "pcor")` / `metatf.grn.pcor.partial_corr_spearman` | rust_pcor_v2 (r3) + NumPy fallback |
| `inferWeightExpMat(method="PUIC")`            | `infer_grn(exp, "puic")` / `metatf.grn.puic.run_puic`  | rust_puic (r3)                  |
| `inferWeightExpMat(method="genie3")`, `GENIE3()` | `infer_grn(exp, "genie3", mode=...)` / `metatf.grn.genie3.run_genie3` | rust_genie3 (r4) |
| `run_sincerities()` / `inferWeightExpMat(method="sincerities")` | `infer_grn(exp, "sincerities", col_data=...)` / `metatf.grn.sincerities.run_sincerities` | py_sincerities_fast (r4) |
| `runViper(gene_list=..., normalization=TRUE)` | `regulon_activity(exp, net, "viper")` / `metatf.scorers.viper_area.run_viper_area` | py_viper_area_v2 (r3) |
| `runAUCell(gene_list=..., normalization=TRUE)`| `regulon_activity(exp, net, "aucell")` / `metatf.scorers.aucell.run_aucell` | py_aucell_v2 (r3) |
| `runGSVA(gene_list=..., method="gsva")`       | `regulon_activity(exp, net, "gsva")` / `metatf.scorers.gsva.run_gsva` | py_gsva (r4) |
| (no R equivalent in metaTF)                   | `regulon_activity(exp, net, "ulm", semantics=...)` / `metatf.scorers.ulm.run_ulm` | py_ulm_dual (r2/r4) |
| `dfToList()`                                  | `metatf.df_to_regulons(net, weighted=...)`            | verbatim R semantics           |

Shared upstream conventions kept by `infer_grn` (repo/R/inferGRNs.R
`inferWeightExpMat` L150–205): the `exp_cutoff` gene filter (default `0` =
keep `rowSums > 0`), regulator subsetting after inference, output
regulator × target.

## Divergence catalogue

### 1. `regulonActivity` matrix method — `norm_act` undefined (R BUG, not ported)
The R `regulonActivity(..., method="matrix")` path errors with
`object 'norm_act' not found` (reproduced in the campaign's round-1 baseline:
3/3 viper matrix runs failed with exactly this error). metatf does not ship
the matrix method; `viper/aucell/gsva/ulm` are the supported scorers.

### 2. `dfToList` drops the weight column
`dfToList` builds character vectors — weights are silently discarded for the
unweighted entry (AUCell/GSVA paths and viper `u` entry). `metatf.df_to_regulons`
reproduces this exactly (`weighted=False`); pass `weighted=True` for the
weighted named-vector entry. This is upstream behaviour, not a port bug.

### 3. AUCell attendance boundary: `missingPercent >= 0.8` (equality drops)
A gene set whose missing-gene fraction is **exactly 0.8 is dropped**. The
campaign's round-2 port used a strict `>` and wrongly kept them; round 3
fixed it (anchor: AUCell source). On the smoke inputs the sets at exactly
0.8 are `Tfap2d` and `Zfp445` — R drops them, and so does metatf
(`tests/test_aucell.py` pins both the real-data drop and the synthetic
boundary).

### 4. viper duplicate edges — the R reference is itself inconsistent
`mm_blood_network` contains 131 duplicated `(tf, target)` pairs. R's
named-vector indexing counts a duplicated edge **twice** in the weighted
entry (and once per unique target after `dfToList` in the unweighted entry).
metatf reproduces both entries exactly: `weighted=True` keeps raw duplicate
rows (w-entry semantics), `weighted=False` goes through `df_to_regulons`
unique targets (u-entry semantics). Consequence for parity: the u-entry
reference closes to 3.2e-14 max |Δ|; the w-entry reference closes to
spearman 0.99996 with max |Δ| 0.25 concentrated on regulons with duplicated
edges — that residual is R-side nondeterminism (which duplicate wins in the
named vector), not a formula difference.

### 5. ULM — two statistics, two decoupler versions (dual semantics, deliberate)
metaTF itself has no ULM; metatf ports decoupler twice, explicitly:
- `semantics="subset"` (default): decoupler **v1.9.2 documented formula** —
  Pearson r over the TF's matched target subset, df = n_targets − 2, with a
  den>0 guard (constant-weight/zero-expression → t=0, p=1).
- `semantics="full_axis"`: decoupler **v2.2.0 implementation** — weights
  zero-padded to the full gene axis, grand-mean centered `_cov/_cor`, df =
  n_genes − 2, **no guard** (0/0 → NaN, as in decoupler).
  External anchor: full_axis closes to decoupler 2.2.0 `run_ulm` to
  **5.4e-12 (t) / 8.1e-13 (raw p)** on exp_example × mm_blood (round 4).
Additional metaTF-side conventions in both modes: duplicates averaged
(decoupler 2.2.0 *rejects* duplicate edges outright), weight==0 edges
dropped (decoupler keeps them → its NaN case cannot arise here).

### 6. GENIE3 — parity mode is bit-level; fast mode is a documented resampling envelope
- `mode="parity"` replicates the **R 4.3 Mersenne-Twister stream** of
  `set.seed(seed)` through every `.C` call (single core). Against the
  300-gene R reference (1000 trees, seed 1): max |Δ| **4.2e-16**, i.e. the
  15-digit CSV quantisation of the reference; top-50 per-target overlap 1.0.
  `n_threads` is ignored in this mode (single stream) — by design.
- `mode="fast"` gives every target an independent MT stream
  (SplitMix64 of the seed). The deviation from R is exactly the bootstrap
  draw sequence. Cross-stream comparison vs the R 6-core reference:
  spearman 0.824, top-50 overlap 0.804 — the same class as R against its
  own re-run with a different core count (R self-consistency measured 0.93
  single vs 6-core in-campaign). Use parity mode when reproducing a paper
  number; fast mode for throughput.
- `mtry`: R `.setMtry("sqrt")` is **round(sqrt(p))**, not floor (a bug in the
  round-3 pure-Python port, fixed in the Rust winner).

### 7. SINCERITIES — 0.5× vs R (slower), glmnet→own elastic net, cvFolds seed
Honest disclosures, per the round-4 manifest:
- The numba build runs at **~0.5× the R runtime** (29.8 s vs 14.8 s on the
  300-gene smoke): slower than R. Kept because it removes the R/ETH
  dependency chain; use R when SINCERITIES runtime dominates.
- R uses `glmnet` (Fortran); metatf uses a **self-written non-negative
  elastic-net coordinate descent** replicating glmnet's default lambda path
  (λ_max = max|z'y|/n / max(α,0.01), ratio 0.01 when n<p else 1e-4, 100
  points, warm starts). Solver-path deviations are why parity is rank-level
  (spearman 0.958, top-50 overlap 0.945 vs R) rather than numeric.
- `cvFolds` in R's SINCERITIES_PLUS has **no seed argument and is not
  reproducible run-to-run**; the Python port takes `seed` (default 0) and is
  deterministic. Cross-checking a specific R PLUS run is therefore
  best-effort (distribution-level), not value-level.
- The numba port is **bit-identical to the round-3 pure-Python port** on the
  same input (np.array_equal audits) — acceleration only, no semantic drift;
  `parallel=True/False` produce identical bits.

### 8. pcor — cov(rank) + ginv tolerance path
`ppcor::pcor(t(exp), method="spearman")$estimate`: the inverted matrix is
the **covariance of ranks** (per-gene rank variances kept — NOT the
correlation matrix; this was the round-2 bug: 2.4e-3 max error on tie-heavy
data), with `MASS::ginv` truncation `λ > sqrt(.Machine$double.eps) · λmax`
(strict `>`, relative to the largest eigenvalue). The 1/(n−1) divisor of
`cov()` cancels in cov2cor and is omitted. `det < eps` branch: metatf always
takes the ginv path (for scRNA data the covariance is always rank ≤ cells−1,
so this is R's branch too). Constant gene rows raise instead of R's NA
propagation (metaTF's `rowSums>0` filter removes them upstream anyway).
The NumPy fallback implements the same formula via
`np.linalg.pinv(rcond=1.4901161193847656e-8)`; measured agreement:
Rust↔R 2.6e-14, Rust↔NumPy 1.4e-14 (full 1000×154 real input).

### 9. Row-order of outputs (benign, locale)
R `split()` orders regulon keys by **locale-dependent collation** (e.g.
`Lbx1` before `LOC102639251` under en_US.UTF-8, after it under C/Python).
metatf orders dict keys by code-point sort. Numeric results are identical
(tests align on row/col labels), but the physical row order of viper/aucell/
gsva outputs can differ from an R run under a non-C locale. GRN matrices
(pcor/puic/genie3/sincerities) index by gene name and are unaffected.

### 10. GSVA scope
Only `method="gsva", kcdf="Poisson"` (rnaseq path) at GSVA 1.50.5 semantics
is ported — the exact call metaTF makes. Gaussian kernel, ssGSEA, z-score,
PLAGE are out of scope. `ppois` uses the Rmath floor tolerance
`floor(q + 1e-7)`; ties in the per-sample gene order use numpy stable sort
(R `order()` is not stable — recorded as a ULP-class uncertainty in the
manifest; measured full-matrix agreement vs R: max |Δ| 5.6e-16).

## Parity gate summary (what the tests assert)

| method     | gate (campaign protocol)                | measured (vendored inputs)      |
|------------|------------------------------------------|----------------------------------|
| viper      | spearman ≥ 0.995, top-100 flip ≤ 1%     | u: ρ=1−2e-13, maxΔ 3.2e-14; w: ρ 0.99996 |
| aucell     | spearman ≥ 0.995, flip ≤ 1%             | ρ 0.9963, maxΔ 0.0296            |
| gsva       | spearman ≥ 0.995, flip ≤ 1%             | ρ 0.99999999999999, maxΔ 5.6e-16 |
| ulm subset | bit-identical to round-2 port           | identity by construction         |
| ulm full_axis | numeric closure vs decoupler 2.2.0    | maxΔ 5.4e-12 (t), 8.1e-13 (p)    |
| pcor       | max |Δ| vs R ≤ 1e-10                    | 2.6e-14 (full 1000-gene matrix)  |
| puic       | spearman ≥ 0.995                        | ρ 0.9999975, maxΔ 0.02           |
| genie3 parity | max |Δ| ≤ 1e-7 vs R (bit protocol)    | 4.2e-16 (=CSV quantisation), top50 overlap 1.0 |
| genie3 fast   | rank envelope vs R 6-core ref         | ρ 0.824, top50 0.804 (R self: 0.93) |
| sincerities  | spearman ≥ 0.95 AND top50 ≥ 0.7       | ρ 0.9584, top50 0.945            |
