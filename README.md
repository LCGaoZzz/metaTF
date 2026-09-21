# metaTF Python/Rust

Fast, R-compatible GRN inference and regulon activity for single-cell data.
This repository is a Python/Rust upgrade of [wanglabsmu/metaTF](https://github.com/wanglabsmu/metaTF):
the pure-Python package installs without R or a Rust toolchain, while an
optional `metatf-rust` wheel provides the PUIC, PCOR, and GENIE3 accelerators.

The implementation is validated against the official metaTF test fixtures and
R reference outputs. It keeps the original methods and matrix orientation,
adds the modern ULM activity path, and caps worker threads at 63.

```bash
pip install -e .
python -m pip install -e '.[plot]'       # optional pheatmap/scater-style views
./build.sh                               # optional Rust accelerator wheel
```

```python
from metatf import infer_grn, regulon_activity

weights = infer_grn(expression, method="pcor")
activity = regulon_activity(expression, network, method="ulm",
                            semantics="full_axis")
```

Read the [full API and test instructions](docs/README.md), [plotting guide](docs/plotting.md),
[semantic compatibility notes](docs/semantics.md), and [benchmark report](docs/benchmarks.md).

License: MIT. See [NOTICE](docs/NOTICE) for upstream and third-party provenance.
