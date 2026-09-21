"""GSVA (Poisson kernel) parity vs the R reference + determinism.

Campaign gate (round 4, numeric_port): spearman_flat >= 0.995, flip <= 1%.
Measured on these vendored inputs: spearman 0.9999999999999888,
max |d| 5.6e-16 (round-off of the R CSV 15-digit quantisation).
"""
from __future__ import annotations

import numpy as np
import pytest

from metatf.scorers.gsva import run_gsva

from conftest import aligned, load_ref, spearman_flat, topk_flip_pct


@pytest.mark.fast
def test_gsva_parity(exp_smoke, gene_sets):
    res = run_gsva(exp_smoke, gene_sets)
    ref = load_ref("gsva_smoke2.csv")
    assert res.shape == ref.shape
    # same member sets; row ORDER may differ from R (locale collation of
    # dfToList/split keys — benign divergence, docs/semantics.md)
    assert set(res.index) == set(ref.index)
    assert set(res.columns) == set(ref.columns)
    a, b = aligned(res, ref)
    assert spearman_flat(a, b) >= 0.995
    assert np.abs(a - b).max() <= 1e-10          # measured 5.6e-16
    assert topk_flip_pct(a, b) <= 0.01


@pytest.mark.fast
def test_gsva_ppois_floor_tolerance():
    # R ppois floor(q + 1e-7) tolerance: x = 3 - 1e-9 must land in bin k = 3
    from metatf.scorers.gsva import _ppois_lower
    q = np.array([3.0 - 1e-9, 3.0 - 1e-5])
    lam = np.array([3.5, 3.5])
    got = _ppois_lower(q, lam)
    from scipy.stats import poisson
    want = poisson.cdf([3, 2], lam)
    assert np.allclose(got, want, atol=1e-12)
    assert _ppois_lower(np.array([-1.0]), np.array([2.0]))[0] == 0.0


@pytest.mark.fast
def test_gsva_determinism_bitwise(exp_smoke, gene_sets):
    r1 = run_gsva(exp_smoke, gene_sets)
    r2 = run_gsva(exp_smoke, gene_sets)
    assert np.array_equal(r1.to_numpy(), r2.to_numpy())
