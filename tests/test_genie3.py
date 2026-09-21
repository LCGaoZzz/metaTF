"""GENIE3: RNG selfcheck, parity-mode determinism (fast), full R-reference
bit-level parity (slow).

Campaign evidence (round 4): parity mode on the 300-gene reference
(genie3_300_c1, R 1.24.0 set.seed(1) single core, nTrees=1000) gives
max |diff| 4.16e-16 (99.9% of entries <= 1e-16; bounded by the 15-digit CSV
quantisation of the reference), top-50 per-target overlap 1.0.  The gate
protocol is genie3_parity_bit with max_abs_diff <= 1e-7.

No 50-gene R reference exists; the fast test covers the small-set parity
mode with self-consistency + thread invariance (per the deliverable spec:
"自洽 + 与 300 参考 top-50 重叠" — the overlap half lives in the slow test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metatf.grn.genie3 import run_genie3, mt_selfcheck, HAVE_RUST_GENIE3
from metatf import infer_grn

from conftest import aligned, load_ref, spearman_flat, top50_per_target_overlap

pytestmark = pytest.mark.skipif(not HAVE_RUST_GENIE3,
                                reason="rust_genie3 extension not installed "
                                       "(./build.sh or pip install metatf-rust)")


@pytest.fixture(scope="module")
def exp50(exp_smoke):
    return exp_smoke.iloc[:50]


@pytest.mark.fast
def test_genie3_mt_rng_selfcheck():
    assert mt_selfcheck(), "MT RNG does not match R 4.3.3 runif reference values"


@pytest.mark.fast
def test_genie3_parity_mode_small_set_deterministic(exp50):
    W1, rn, tn = run_genie3(exp50, n_trees=100, mode="parity", seed=1)
    W2, _, _ = run_genie3(exp50, n_trees=100, mode="parity", seed=1)
    assert W1.shape == (50, 50)
    assert rn == list(exp50.index) and tn == list(exp50.index)
    assert np.array_equal(W1, W2), "parity mode is not deterministic"
    # thread count must not move parity mode (single MT stream)
    W3, _, _ = run_genie3(exp50, n_trees=100, mode="parity", seed=1, n_threads=8)
    assert np.array_equal(W1, W3)
    # seed changes the stream (bootstrap draws differ)
    W4, _, _ = run_genie3(exp50, n_trees=100, mode="parity", seed=2)
    assert not np.array_equal(W1, W4)


@pytest.mark.fast
def test_genie3_fast_mode_deterministic_and_normalized(exp50):
    W1, _, _ = run_genie3(exp50, n_trees=100, mode="fast", seed=1, n_threads=4)
    W2, _, _ = run_genie3(exp50, n_trees=100, mode="fast", seed=1, n_threads=4)
    assert np.array_equal(W1, W2), "fast mode is not deterministic"
    # per-target (column) importance normalised to sum 1 (diag included in the sum)
    assert np.allclose(W1.sum(axis=0), 1.0, atol=1e-12)
    assert np.all(np.diag(W1) >= 0)


@pytest.mark.fast
def test_genie3_input_validation(exp50):
    with pytest.raises(ValueError):
        run_genie3(exp50, mode="bogus")
    with pytest.raises(ValueError):
        run_genie3(exp50, seed=-1)
    with pytest.raises(ValueError):
        run_genie3(exp50, tree_method="GBM")


@pytest.mark.slow
def test_genie3_parity_full_300_reference(exp300):
    """Bit-level parity vs R GENIE3 1.24.0 set.seed(1), single core, 1000 trees.

    Input: exp_smoke first 300 genes in original order x all 154 cells, NO
    rowSums filter (the R reference was a direct GENIE3 call).
    ~70 s single-threaded.
    """
    W, rn, tn = run_genie3(exp300, n_trees=1000, mode="parity", seed=1)
    ref = load_ref("genie3_300_c1.csv")
    a, b = aligned(pd.DataFrame(W, index=rn, columns=tn), ref)
    d = np.abs(a - b).max()
    assert d <= 1e-7, d                       # campaign gate; measured 4.2e-16
    assert spearman_flat(a, b) >= 0.999999
    assert top50_per_target_overlap(a, b) >= 0.999   # measured 1.0
