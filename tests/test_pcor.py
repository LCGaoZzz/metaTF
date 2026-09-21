"""pcor parity (NumPy fallback and/or Rust backend) vs the R reference.

The vendored reference is the [0:300, 0:300] leading sub-block of the full
1000 x 1000 R `inferWeightExpMat(method='pcor')` matrix (runs/round1/refs/
pcor_smoke.csv, 19 MB — sub-blockliced at vendoring time, byte-identical to
the source block).  PARTIAL CORRELATION IS NOT MARGINAL: the test therefore
runs pcor on the FULL 1000-gene input and compares the leading 300 x 300
block of the output — a genuine full-rank-path check, not a 300-gene rerun.

Campaign evidence (round 3): Rust vs R reference max |d| 2.6e-14 on the full
matrix; Rust vs NumPy same-path reference 1.4e-14.  Gate here: 1e-10.
"""
from __future__ import annotations

import numpy as np
import pytest

from metatf.grn.pcor import partial_corr_spearman, pcor_numpy, HAVE_RUST_PCOR
from metatf import infer_grn

from conftest import load_ref


@pytest.mark.fast
def test_pcor_numpy_matches_r_reference_subblock(exp_smoke):
    W = infer_grn(exp_smoke, method="pcor", backend="numpy")
    assert W.shape == (1000, 1000)
    ref = load_ref("pcor_smoke_top300.csv")
    got = W.iloc[:300, :300].to_numpy()
    want = ref.to_numpy()
    d = np.abs(got - want).max()
    assert d <= 1e-10, d
    assert np.allclose(np.diag(W.to_numpy()), 1.0, atol=1e-12)
    assert np.allclose(W.to_numpy(), W.to_numpy().T, atol=1e-10)


@pytest.mark.fast
def test_pcor_numpy_formula_on_tie_heavy_synthetic():
    """cov-of-ranks path, not correlation-of-ranks (the round-2 bug class):
    tie-heavy integer data has per-gene rank variances — pin the formula."""
    rng = np.random.default_rng(7)
    X = rng.integers(0, 5, size=(60, 30)).astype(float)
    P = pcor_numpy(X)
    # independent recompute (statsmodels-free): cov of ranks -> pinv -> -cov2cor
    from scipy import stats
    Rk = stats.rankdata(X, axis=1, method="average")
    cv = np.cov(Rk, rowvar=True, ddof=1)
    Th = np.linalg.pinv(cv, rcond=1.4901161193847656e-8)
    dd = np.sqrt(np.diag(Th))
    P2 = -Th / np.outer(dd, dd)
    np.fill_diagonal(P2, 1.0)
    assert np.array_equal(P, P2)
    # and it must DIFFER from the correlation-of-ranks path on this input
    Z = Rk - Rk.mean(axis=1, keepdims=True)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    C = np.linalg.pinv(Z @ Z.T, rcond=1.4901161193847656e-8)
    dd2 = np.sqrt(np.diag(C))
    Pcor = -C / np.outer(dd2, dd2)
    np.fill_diagonal(Pcor, 1.0)
    assert np.abs(P - Pcor).max() > 1e-6


@pytest.mark.fast
def test_pcor_constant_rows_raise():
    X = np.ones((5, 10))
    with pytest.raises(ValueError):
        partial_corr_spearman(X)


@pytest.mark.fast
@pytest.mark.skipif(not HAVE_RUST_PCOR, reason="rust_pcor_v2 not installed")
def test_pcor_rust_matches_numpy_and_reference(exp_smoke):
    X = np.ascontiguousarray(exp_smoke.to_numpy(dtype=np.float64))
    rust = partial_corr_spearman(X, backend="rust", n_threads=4)
    nump = pcor_numpy(X)
    d = np.abs(rust - nump).max()
    assert d <= 1e-10, d                       # campaign measured 1.4e-14
    ref = load_ref("pcor_smoke_top300.csv")
    assert np.abs(rust[:300, :300] - ref.to_numpy()).max() <= 1e-10
    # thread-count determinism (bit-identical per campaign manifest)
    r1 = partial_corr_spearman(X, backend="rust", n_threads=1)
    r8 = partial_corr_spearman(X, backend="rust", n_threads=8)
    assert np.array_equal(r1, r8)


@pytest.mark.fast
def test_pcor_default_backend_resolves(exp_smoke):
    exp = exp_smoke.iloc[:150, :]
    W = infer_grn(exp, method="pcor")           # auto: rust if present else numpy
    assert W.shape == (150, 150)
    W2 = infer_grn(exp, method="pcor", backend="numpy")
    assert np.abs(W.to_numpy() - W2.to_numpy()).max() <= 1e-10


@pytest.mark.fast
def test_pcor_fallback_engages_when_rust_missing(exp_smoke, monkeypatch):
    """No Rust backend importable -> pcor still runs via NumPy (same formula)."""
    import metatf._rust as backend_mod
    from metatf.grn import pcor as pcor_mod
    monkeypatch.setattr(backend_mod, "_load_facade", lambda: None)
    monkeypatch.setattr(backend_mod, "_ext", lambda sub: None)
    monkeypatch.setattr(pcor_mod, "_rust_fn", lambda: None)
    exp = exp_smoke.iloc[:120, :]
    W = infer_grn(exp, method="pcor")            # must NOT raise
    assert W.shape == (120, 120)
    Wn = infer_grn(exp, method="pcor", backend="numpy")
    assert np.array_equal(W.to_numpy(), Wn.to_numpy())
    # puic/genie3 must fail loudly with the install hint instead
    from metatf.grn.puic import run_puic
    with pytest.raises(RuntimeError, match="metatf-rust"):
        run_puic(exp)
    from metatf.grn import genie3 as genie3_mod
    monkeypatch.setattr(genie3_mod, "_ext_fns", lambda: None)
    with pytest.raises(RuntimeError, match="metatf-rust"):
        genie3_mod.run_genie3(exp, n_trees=5)
