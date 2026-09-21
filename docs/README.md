# metatf-py — metaTF without an R runtime

`metatf` is a Python/Rust reimplementation of the metaTF analysis chain
(upstream R package **metaTF 0.1.1**, single-cell regulon analysis tools),
assembled from the verified winners of the metaTF upgrade campaign
(rounds 2–4, 2026). Every method ships with campaign-measured parity
evidence against R references and vendored test fixtures that reproduce
those gates.

- GRN inference: `pcor`, `puic`, `genie3`, `sincerities`
- Regulon activity: `viper` (aREA), `aucell`, `gsva`, `ulm` (dual semantics)

**No R runtime, no R packages, no renv are needed anywhere.**

## Install

```bash
# pure-Python package (pcor has a NumPy fallback; puic/genie3 need the ext)
pip install -e .                      # from this repository root

# optional Rust accelerators (single wheel: pcor + puic + genie3)
./build.sh                            # cargo + maturin; or: pip install metatf-rust
```

Python >= 3.10; core deps `numpy`, `pandas`, `scipy`. Optional extras:
`metatf[numba]` (sincerities acceleration, strongly recommended),
`metatf[anndata]` (.h5ad input), `metatf[parquet]`, `metatf[test]`.
Install `metatf[plot]` for the optional Matplotlib view helpers.

## Quick start

```python
import pandas as pd
from metatf import infer_grn, regulon_activity, df_to_regulons

exp = pd.read_csv("expression.csv", index_col=0)     # genes x cells
net = pd.read_csv("network.csv")                     # tf, target, weight

# GRN inference (regulator x target weight matrices)
W_pcor = infer_grn(exp, method="pcor")               # NumPy fallback OK
W_puic = infer_grn(exp, method="puic")               # needs metatf-rust
W_genie3 = infer_grn(exp, method="genie3", mode="parity", seed=1)  # bit-level R stream
col_data = pd.read_csv("time.csv")                   # sample,time (>=3 points)
W_sinc = infer_grn(exp, method="sincerities", col_data=col_data)

# Regulon activity (regulons x cells)
nes     = regulon_activity(exp, net, method="viper")           # NES
auc     = regulon_activity(exp, net, method="aucell")          # normAUC
es      = regulon_activity(exp, net, method="gsva")            # GSVA ES
t, pval = regulon_activity(exp, net, method="ulm",
                           semantics="subset")                 # decoupler v1.9.2 formula
```

CLI:

```bash
metatf infer-grn --method pcor --input exp.csv --output W.csv --threads 8
metatf activity --method viper --input exp.csv --network net.csv --output nes.csv
metatf activity --method ulm --input exp.csv --network net.csv \
    --semantics full_axis --output t.csv          # also writes t_pval.csv
metatf infer-grn --method sincerities --input exp.csv --col-data time.csv --output W.csv
```

## Threads

Default worker threads = **min(8, CPU cores)** with a **hard cap of 63**.
(R's integer-index guard; rayon misbehaves on some >63-thread NUMA hosts.)
Override per call with `threads=` / `--threads`; values above 63 are clamped.

## What happens without the Rust wheel

| method  | behaviour                                                     |
|---------|---------------------------------------------------------------|
| pcor    | falls back to the NumPy same-path implementation (cov-of-ranks + ginv, campaign-verified to ~1e-14 vs Rust and vs R) |
| puic    | `RuntimeError` with install instructions                      |
| genie3  | `RuntimeError` with install instructions                      |
| viper / aucell / gsva / ulm / sincerities | pure Python(+numba), unaffected |

## Documentation

- `docs/semantics.md` — R→Python mapping, every semantic divergence, per-method
  parity protocols and their measured numbers
- `docs/benchmarks.md` — timing/speedup/parity tables vs R baselines
- `metatf[plot]` — optional matplotlib helpers matching the original
  heatmap, reduced-dimension, expression-distribution and Radviz views
- `docs/plotting.md` — view mapping and publication export examples
- `docs/BUILD.md` — building the extension wheel, pinned versions, cross-platform notes
- `docs/manifests/<candidate>/MANIFEST.json` — the campaign implementation
  manifests for the eight winning candidates (copied verbatim)
- `NOTICE` is at `docs/NOTICE`; licence of the upstream metaTF package: MIT
  (repo/DESCRIPTION).

## Tests

```bash
pytest -m fast      # 43 tests with Rust; pure-Python installs skip Rust-backed checks
pytest -m slow      # full-reference parity: GENIE3 bit-level (~70 s), SINCERITIES rank (~3 min)
pytest              # everything
```

Vendored fixtures under `tests/data/` (inputs derived from the metaTF repo
extdata; references produced by the campaign's R round-1 runner). Rust-backed
tests are skipped with an explicit reason when the extension is absent.
