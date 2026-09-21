# MANIFEST — metatf-py deliverable (round 5 assembly)

Assembled 2026-09-20 from the metaTF upgrade campaign winners
(rounds 2–4). Package: `metatf` 0.1.0 (pure Python) + optional `metatf-rust`
0.1.0 (single facade wheel over the three Rust winner crates).

## Layout

```
metatf-py/
├── pyproject.toml          pure-Python package (setuptools); pip install -e .
├── build.sh                builds + installs the metatf-rust wheel, selfchecks
├── src/metatf/             package: __init__ / api / adapters / cli / _rust
│   ├── scorers/            viper_area, aucell, gsva, ulm  (verbatim winners)
│   └── grn/                pcor (rust+numpy fallback), puic, genie3 (wrappers),
│                           sincerities (verbatim numba winner)
├── rust/                   cargo workspace: rust_pcor_v2, rust_puic, rust_genie3
│   │                       (verbatim winner crates) + metatf_rust facade cdylib
│   ├── Cargo.lock          committed for reproducible builds
│   └── (dist/ and target/ are generated locally and are not committed)
├── tests/                  48 tests (43 fast / 5 slow) + vendored fixtures
│   └── data/               inputs (exp_smoke, mm_blood_network, test_col_data)
│       └── refs/           8 R reference CSVs (pcor = 300x300 leading sub-block)
└── docs/                   README, semantics, benchmarks, BUILD, NOTICE,
    └── manifests/          the 8 winner MANIFEST.json files (verbatim copies)
```

Total tree size ≈ 19.6 MB (fixtures included; rust/target build dir excluded).

## Verified state (campaign and final assembly)

- `pip install -e .` — OK (pure Python path, no toolchain)
- `cargo test --release` — OK: all Rust crates and GENIE3 RNG unit tests
- `./build.sh` — requires maturin in the target Python environment; when
  available it builds and installs the platform-specific `metatf-rust` wheel
  and runs the extension selfcheck
- `pytest -m fast` — **43 passed, 5 deselected** with the Rust wheel; the
  pure-Python path runs 35 and skips 8 Rust-backed checks
- `pytest -m slow` — **5 passed** (~11 min: GENIE3 300-gene bit-level parity,
  SINCERITIES rank parity + determinism)

## Key artifact SHA-256

| file | bytes | sha256 |
|---|---:|---|
| `src/metatf/__init__.py` | 1,668 | `b3312d81a89c795574aa8a88f80e2c0d63491142b330aa784a2b4e59490083e4` |
| `src/metatf/api.py` | 11,890 | `5157e2bd2b699a06f99825de41a42f81c0cb5ad1e9c9cf370b966810eef4123c` |
| `src/metatf/adapters.py` | 5,909 | `823ffbce96b1677b3066aa6b5a91b3440f2eb994b12e5111d442fc1f201b6e23` |
| `src/metatf/cli.py` | 5,529 | `f9899219f20fdf74446ecd635ca6fbe05e34737596e10cde884e01b08cdb7b6e` |
| `src/metatf/_rust.py` | 2,925 | `54501742ea68bb302c4e636a9e0894f17625643e5bad2c9587a6a1edb1b5097e` |
| `src/metatf/plotting.py` | 18,768 | `b8660b699a0bad2425e3bb2ea6a39ab0c5bc6c78914c7dea2c2f80438874144d` |
| `src/metatf/scorers/viper_area.py` | 9,602 | `b6246fa714dad39ca2e7e6d0b6e309ebd0b6522972c4b4b79ad428f247722761` |
| `src/metatf/scorers/aucell.py` | 5,128 | `908d90c72f46a44d24ba42a18d28ac01d5eaf6ac90c95d9fd0640f5a47a17ae0` |
| `src/metatf/scorers/gsva.py` | 11,809 | `0062c9e82884b494ce7f4ddeb4d45eab3cde23cdd7cd7fb33336a85362db01ce` |
| `src/metatf/scorers/ulm.py` | 9,761 | `b04b821d9fcba51668da3cce1f3c523c399dc7eda02b87a39e1cef9300e926e5` |
| `src/metatf/grn/pcor.py` | 3,623 | `92b21040212175963fdc1546f06652f4fad43843a87b816d6e2cb15e18598930` |
| `src/metatf/grn/puic.py` | 2,133 | `c44127de2f3d7e1e3f81e0abe7a2c29c26711a1c7ef8e3e2701d768d9d9ceed4` |
| `src/metatf/grn/genie3.py` | 9,006 | `7cd65f96d9c13c64b2370f61de00dd8f768985c420f7cfdb5d3f958514ee9469` |
| `src/metatf/grn/sincerities.py` | 31,045 | `2d2b061225ae0632423cd481062c7d168b1e9b5f12f0698fbc37997b02fc2282` |
| `rust/Cargo.toml` | 641 | `a3205ca6c89eb36889b00c8baaa20c7295270e74f7a5d6912fd62328a0b62e36` |
| `rust/Cargo.lock` | 8,631 | `7d1b4964020cdb91fa88397ad21053ab0659560fac6dc6fa95123d3c85aadd67` |
| `rust/pyproject.toml` | 589 | `bdba9ff7f22a2a8c3b395811e6efc697cbeccaf458dc0294c81606f2bc122f90` |
| `rust/metatf_rust/src/lib.rs` | 1,152 | `050444a28cf41d87c3a61fc271738043a1720c9db9b817d3120177e4793f0dad` |
| `rust/python/metatf_rust/__init__.py` | 1,083 | `a3c2b714b439acc3d77b1c561a713ebaab59814766c10f4ab03ffa5cc56633bd` |
| `rust/rust_pcor_v2/src/lib.rs` | 12,414 | `2f1273f794d5ec27745e32e04df8b9ab93dfb65c59d56da779e31b7fb90887be` |
| `rust/rust_puic/src/lib.rs` | 19,663 | `7dc692f724cc5a472ef1856200f02f65134198a4c99099a5f18bb2755fc5651b` |
| `rust/rust_genie3/src/lib.rs` | 7,596 | `4c65903bc23364267fdedba616359f209ddd14cc3e42a94113b49df3963e3b06` |
| `rust/rust_genie3/src/genie3.rs` | 25,708 | `423b9e37976e212e49de2704a7da34dfb10dff991c9a53de73ce0302ff421d4c` |
| `rust/rust_genie3/src/rng.rs` | 13,550 | `2108f2f853c444779fc365bbdc5c9ebaaa0803080a91d2696112165396359fa0` |
| `pyproject.toml` | 1,619 | `7dc049e8481f02e9304f8d17fe10289923c9e58b83de089c61685c33b42ae370` |
| `build.sh` | 2,413 | `6e2ecb357c2c8f5b2c232946c9728fb2ecda2fdcb2120e24758947fbf5fa66d8` |
| `tests/test_plotting.py` | 1,420 | `3d1aababcbcb83cd60ef34736d1aec4cf13fd90baaf9517fbbf78729f8f47ee7` |
| `tests/conftest.py` | 4,146 | `b255026f7784bf2d94d6b034aec8e775c5f22e60aa409e7561d9beb66d031f35` |
| `tests/data/exp_smoke.csv` | 489,213 | `caa4112de844fbad5f8f5423756675a3c766dbdeefeae272fd8b88c9f83c1f7b` |
| `tests/data/mm_blood_network.csv.gz` | 5,160,355 | `df8b94f42e42ce922af060eed0f56a1fe39eb0ba1d7f159888562107c4cb84d9` |
| `tests/data/test_col_data.csv` | 2,530 | `6a9a0c1bf8b41e570834df360a07c64adca3cb168c64937faa134c9f92f7cd7f` |
| `tests/data/refs/viper_smoke_u.csv` | 2,396,303 | `89832d51fcc9b4a62ceba9f109f7fb5969bd1b901e61fbc138a903cd009d7c2e` |
| `tests/data/refs/viper_smoke_w.csv` | 2,403,998 | `194516757aebd885e178cbf77445bb90d30e20547e247575ff7407972df162f8` |
| `tests/data/refs/aucell_smoke2.csv` | 10,444 | `d082d197c47e6b2f550c4a4cc6af1ed02f0a5940016c6298e7b5e8c46c5347e1` |
| `tests/data/refs/gsva_smoke2.csv` | 2,552,038 | `595199fd0ecc281ceee6e42573dacbf5d62ef21df60d99cc809b692e8f2ed592` |
| `tests/data/refs/genie3_300_c1.csv` | 1,805,235 | `0452096fdbe063ce63c82dc28817930bb74aa97064f2d4b7d9e96defd772605e` |
| `tests/data/refs/puic300_smoke.csv` | 1,244,194 | `53ce5ddb3bb11e9eaf2fef208e0644fdd4bd90f10da44d98841ee88189453797` |
| `tests/data/refs/sincerities300_smoke.csv` | 1,020,378 | `eccbced0305e9be9468e8daa1bee2c38da0e1a73ef15bcdb493d9e0521c429f1` |
| `tests/data/refs/pcor_smoke_top300.csv` | 1,730,595 | `139cb0e9785686754897302b787bdb31086712e23351bbb88faca5e85be76a6f` |
| `docs/README.md` | 4,715 | `18fe36ca9906e883585978763680cfa750b76acfac2797ac2fdbbe34f8ce10ef` |
| `docs/semantics.md` | 10,545 | `fddf5899d05057a5c7af5d3c82221d7223f1453109634a4730f6233e7b1d111a` |
| `docs/benchmarks.md` | 6,427 | `f0d3f843aecfcc0be25c259187643d9a0c8ecbca0255d5875e4f674c389ece57` |
| `docs/BUILD.md` | 4,508 | `18947563d4df85b4192f8d4fb3a7495585043a1fc08f476874739a7046bc9f0d` |
| `docs/NOTICE` | 4,052 | `cebfba0b8a80265049fd7f10af7b46662355a27484727b35e2fdd5d1ad03a790` |
| `docs/plotting.md` | 1,468 | `da275a3866fb060afd619b6ec59da104565098f1d4fe0ac013cb266c8821d3fa` |

Vendored reference provenance: `runs/round1/refs/*.csv` (R round-1 runner);
inputs derived from `repo/inst/extdata` (see tests/conftest.py header).
Winner sources under `src/metatf/scorers|grn` and `rust/*` are verbatim
copies of the campaign candidates; the ONLY in-tree edits are documented in
docs/BUILD.md (crate-type gains "rlib"; `pub` on the three #[pymodule] fns).

## Quick verification

```bash
cd metatf-py
python -m pip install -e .                       # pure-Python install
python -c "import metatf; print(metatf.__version__, metatf.rust_available())"
python -m pytest tests -m fast -q                # 43 passed with Rust; pure-Python skips Rust checks
./build.sh                                       # optional Rust wheel
python -m pytest tests -m slow -q                # 5 passed (~11 min)
```
