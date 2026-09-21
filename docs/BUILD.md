# Building metatf

## Two packages, one repository

1. **`metatf`** (this repo root) — pure Python. `pip install -e .` needs
   nothing but setuptools; no Rust toolchain, no compiled artifacts.
2. **`metatf-rust`** (`rust/`) — the optional accelerator wheel: one cargo
   workspace with the three verbatim winner crates plus a `metatf_rust`
   facade cdylib that exposes them as submodules of a single extension
   module (`metatf_rust.rust_pcor_v2` / `.rust_puic` / `.rust_genie3`).

**Why a separate extension wheel instead of one mixed wheel:** a maturin
mixed build compiles Rust on *every* install, which would break the
requirement that `pip install -e .` works on machines with no toolchain, and
maturin builds exactly one extension module per invocation (three separate
mixed builds would mean three wheels anyway). The facade workspace gives
ONE wheel covering all three accelerators; `metatf` discovers it at import
time and degrades gracefully without it (pcor → NumPy fallback; puic/genie3
→ actionable error). The standalone campaign wheels
(`rust_pcor_v2`/`rust_puic`/`rust_genie3`) are also detected as a
compatibility fallback.

## Build the extension

```bash
./build.sh                 # cargo test + maturin build + pip install + selfcheck
# or manually:
cd rust
PATH="$HOME/.cargo/bin:$PATH" python -m maturin build --release -i "$(command -v python)" -o dist
python -m pip install --force-reinstall --no-deps dist/metatf_rust-*.whl
```

Requirements: cargo/rustc (tested 1.88.0; any stable ≥ 1.74 should do), a C
compiler (cc), maturin ≥ 1.5 (< 2.0), Python ≥ 3.10 with the headers/dev
package of the target interpreter.

Plotting is optional so the numerical core stays lightweight:

```bash
python -m pip install -e '.[plot]'
```

The helpers `plot_heatmap`, `plot_reduced_dim`, `plot_regulon_reduced_dim`,
`plot_expression`, and `plot_radviz` reproduce the main pheatmap/scater/
ggplot2/Radviz view types used by the upstream vignettes.

### Pinned versions

| crate / tool | pin   | note                                        |
|--------------|-------|---------------------------------------------|
| pyo3         | =0.22.6 | workspace-wide; `extension-module` feature |
| numpy (rust) | =0.22.1 |                                            |
| rayon        | =1.10.0 |                                            |
| maturin      | >=1.5,<2.0 | built/tested with 1.15.0                |
| rustc        | tested 1.88.0 | edition 2021                           |

Transitive dependencies are pinned by the committed `rust/Cargo.lock`
(generated on first build, kept in-tree for reproducibility).

### pyo3 0.22 notes (already applied in-tree)

- The facade creates submodules with `PyModule::new_bound` (the Bound-API
  name in 0.22; renamed to `new` in 0.23).
- The three winner crates carry a one-token edit relative to the campaign
  sources: their `#[pymodule] fn` is now `pub` so the facade can call it
  cross-crate. No numeric code was touched; diff against
  `docs/manifests/*/MANIFEST.json` provenance or the campaign directories.
- Each crate's `crate-type` gained `"rlib"` (facade linkage) while keeping
  `"cdylib"`, so each still builds standalone exactly as in the campaign.

## Cross-platform wheels

`metatf-rust` is a native extension: build one wheel per target platform
(CPython version × OS × arch). From a checkout:

```bash
# native wheel (e.g. linux x86_64, cp311)
python -m maturin build --release -i "$(command -v python)" -o dist

# abi3 wheel (works across CPython 3.10+ of one platform) — add to
# rust/metatf_rust/Cargo.toml under [dependencies.pyo3]: features =
# ["extension-module", "abi3-py310"] — NOTE: abi3 changes the RNG/float code
# paths only in that it disables CPython-specialised fast paths; the campaign
# parity numbers were measured on the version-specific build, so parity-gate
# the abi3 build before shipping it.

# cross-compile (e.g. from linux for windows)
rustup target add x86_64-pc-windows-gnu
python -m maturin build --release --target x86_64-pc-windows-gnu -o dist
```

macOS: build on macOS (linking against the system Python framework);
`MACOSX_DEPLOYMENT_TARGET` is picked up by pyo3-build-config automatically.

## Verification after build

```bash
python - <<'EOF'
import metatf
print(metatf.rust_available())       # all three True
EOF
pytest -m fast                        # 43 passed with the Rust wheel
pytest -m slow                        # 5 passed (~11 min; genie3 bit parity + sincerities)
```
