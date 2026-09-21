# metaTF-py

`metaTF-py` is the Python/Rust upgrade of [wanglabsmu/metaTF](https://github.com/wanglabsmu/metaTF).
It keeps the original GRN and regulon-activity contracts while removing the R
runtime from the default path. The Python package is installed with pip; an
optional `metatf-rust` wheel accelerates the largest matrix kernels.

## What the upgrade delivers

The figures below are measured on the campaign host (32 logical threads,
Python 3.11, R 4.3.3) using the official metaTF fixtures. They are benchmark
results at the labelled data tier, not promises for every hardware or dataset.

| Method | New backend | Measured speed | Memory behavior | Precision / compatibility evidence |
|---|---|---:|---|---|
| PUIC | Rust + rayon | **179×** (300 genes), **213×** (1000 genes); 14169×154 in 28.6 s | +4.7 GB at 14169 genes; dense gene×gene output is the dominant allocation | ρ=0.9999985, max Δ=0.016; numeric parity gate passed |
| PCOR | Rust | **26.5×** on 1000 genes; 14169×154 in 1.83 s | Avoids an extra covariance matrix; output size remains intrinsic to PCOR | max \|Δ\|=2.6e−14 vs R; parity passed |
| GENIE3 (parity) | Rust, R-compatible MT stream | **1.67×** | Deterministic implementation; worker count capped at 63 | max \|Δ\|=4.2e−16; top-50 overlap 1.0 |
| GENIE3 (fast) | Rust + rayon | **18.2×** (300 genes), **3.6×** (1000 genes @16T) | Parallel execution under the same 63-thread cap | Matches the R cross-seed resampling envelope; use parity mode for bit-level reproduction |
| AUCell | Chunked NumPy | **14.3–62.3×** | Chunked scoring; scoring tier measured at ≤224 MB peak increment | ρ=0.999998; 0.02% ranking flips; boundary semantics retained |
| VIPER aREA | Chunked NumPy | **4.0–6.3×** | Chunked cell processing avoids a full regulon×gene temporary | ρ=1.0, max Δ=0 on held-out and example references |
| GSVA (Poisson) | NumPy | **1.1–1.7×** | Chunked implementation; ≤224 MB peak increment in scoring tier | ρ≈1−1e−10; max Δ=5.6e−16 on smoke data |
| SINCERITIES | NumPy + numba | 0.60–1.03× | Bounded gene blocks; no R/glmnet dependency | ρ=0.958, top-50 overlap 0.945; rank parity passed, runtime limitation documented |
| ULM (modern path) | Chunked NumPy, dual semantics | **7.7× vs VIPER / 21.5× vs AUCell** | Chunked TF/cell operations; ≤224 MB peak increment in scoring tier | `subset` closes to decoupler v1.9.2; `full_axis` closes to decoupler 2.2.0 (t max Δ=5.4e−12) |

The implementation preserves the original matrix orientation and method
contracts. Legacy VIPER remains available; ULM is an explicitly selectable
modern alternative and is documented as a different statistic rather than a
drop-in replacement. All methods have parity fixtures and semantic notes in
[`docs/semantics.md`](docs/semantics.md); the complete timing table is in
[`docs/benchmarks.md`](docs/benchmarks.md).

### Memory and parallelism

Scoring paths use chunked computation and measured peak increments of at most
224 MB on the campaign inputs. PUIC is the exception at full scale: its
14169×14169 dense result itself requires several gigabytes, and the measured
increment was 4.7 GB, below the R reference path's comparable multi-copy
allocation. The public API keeps the dense matrix contract so changing this
would change downstream semantics.

The default is `min(8, CPU cores)` workers and every `threads=`/`--threads`
value is clamped to 63. The campaign never used more than 32 logical threads.

## Install and use

```bash
# core package: no R runtime and no Rust toolchain required
pip install -e .

# optional plots matching the upstream pheatmap/scater/Radviz view types
python -m pip install -e '.[plot]'

# optional Rust accelerators for PCOR, PUIC and GENIE3
./build.sh
```

```python
from metatf import infer_grn, regulon_activity

weights = infer_grn(expression, method="pcor")
activity = regulon_activity(expression, network, method="ulm",
                            semantics="full_axis")
```

The package also provides `metatf infer-grn` and `metatf activity` CLI
commands, AnnData input support as an optional extra, and plotting helpers for
scaled heatmaps, reduced dimensions, expression distributions, and Radviz.
See the [full API and test instructions](docs/README.md),
[plotting guide](docs/plotting.md), [semantic compatibility notes](docs/semantics.md),
and [benchmark report](docs/benchmarks.md).

## Omicos Agent/Skill integration

A portable [Omicos bundle](agent-harness/omicos/README.md) provides the
`metatf_analyst` Agent, the `metatf` Skill and a thin native-CLI launcher.
It preserves scientific method options while bounding threads to the available
CPU allocation. No additional workflow engine or algorithm implementation is added.

```bash
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace
# For a separate omicos-admin catalog PR:
python agent-harness/install_omicos.py --destination /path/to/omicos-admin --layout catalog
```

The selected Python environment must already contain `metatf`. Copying the
bundle does not install the package or deploy a production Omicos Agent.
See the [harness overview](agent-harness/README.md) for tests and details.

## Verification

The deliverable contains the official-derived fixtures and R reference CSVs.
The final local run completed **48 tests successfully**, including the slow
GENIE3 bit-parity and SINCERITIES rank-parity checks. GitHub Actions verifies
Python 3.10, 3.11, 3.12, and the Rust/maturin wheel path on every push.

The repository is [LCGaoZzz/metaTF-py](https://github.com/LCGaoZzz/metaTF-py),
released at `v0.1.0`. The importable distribution remains named `metatf` for
Python compatibility. License: MIT; see [`docs/NOTICE`](docs/NOTICE) for
upstream and third-party provenance.
