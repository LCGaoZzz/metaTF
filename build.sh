#!/usr/bin/env bash
# Build the metatf-rust extension wheel (aggregate: rust_pcor_v2 + rust_puic +
# rust_genie3 in one cdylib) and install it into the active Python environment.
#
# Requirements: cargo/rustc (https://rustup.rs), maturin (pip), a C compiler.
# Environment notes carried from the campaign (rounds 2-4):
#   - cargo/rustc often live in ~/.cargo/bin (NOT on PATH by default)
#   - maturin must run as `python -m maturin` with `-i <sys.executable>`,
#     otherwise it may bind a different interpreter and produce an unusable wheel
set -euo pipefail
cd "$(dirname "$0")/rust"

export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "cargo not found (install rustup)"; exit 1; }

PY="${PYTHON:-python3}"

echo "== cargo test (genie3 RNG unit tests vs R 4.3.3 reference values) =="
cargo test --release 2>&1 | tail -20

echo "== maturin build (wheel, -i $PY) =="
"$PY" -c 'import maturin' 2>/dev/null || {
  echo "maturin is required in the target Python environment; install with: $PY -m pip install \"maturin>=1.5,<2\"" >&2
  exit 1
}
rm -rf dist
"$PY" -m maturin build --release -i "$PY" -o dist

echo "== install wheel =="
"$PY" -m pip install --force-reinstall --no-deps dist/metatf_rust-*.whl

echo "== clean-import + backend selfcheck =="
"$PY" - <<'EOF'
import numpy as np
import metatf_rust
print("import metatf_rust ->", metatf_rust.__file__)
assert metatf_rust.rust_genie3 is not None
assert metatf_rust.rust_puic is not None
assert metatf_rust.rust_pcor_v2 is not None

# pcor: 300x40 dual-Gram path
m = np.ascontiguousarray(np.random.default_rng(0).normal(size=(300, 40)))
out = metatf_rust.rust_pcor_v2.partial_corr_spearman(m)
assert out.shape == (300, 300) and np.isfinite(out).all()
assert np.allclose(np.diag(out), 1.0) and np.allclose(out, out.T, atol=1e-10)

# puic: 20x100
W, regs, tgts = metatf_rust.rust_puic.puic(
    np.ascontiguousarray(np.random.default_rng(0).normal(size=(20, 100))),
    [f"g{i}" for i in range(20)])
assert W.shape == (20, 20) and np.isfinite(W).all()

# genie3: MT RNG selfcheck vs R 4.3.3 runif reference
assert metatf_rust.rust_genie3.mt_first_unifs(1, 5)[0] == 0.26550866314209998
print("metatf_rust clean import + selfcheck OK")
EOF

echo "== metatf picks the extension up =="
"$PY" - <<'EOF'
import metatf
print("metatf rust_available:", metatf.rust_available())
assert all(metatf.rust_available().values())
EOF

echo "BUILD_OK"
