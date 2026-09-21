"""PUIC parity vs the R reference (requires the Rust extension).

Campaign gate (round 3, numeric_port): spearman_flat >= 0.995; measured on
puic300_smoke: spearman 0.9999975, max |d| 0.02, 179x faster than R.
Bit-identical across thread counts (fixed accumulation order).
"""
from __future__ import annotations

import numpy as np
import pytest

from metatf.grn.puic import run_puic, HAVE_RUST_PUIC
from metatf import infer_grn

from conftest import aligned, load_ref, spearman_flat

pytestmark = pytest.mark.skipif(not HAVE_RUST_PUIC,
                                reason="rust_puic extension not installed "
                                       "(./build.sh or pip install metatf-rust)")


@pytest.mark.fast
def test_puic_parity_300(exp300):
    W = run_puic(exp300, n_threads=8)
    ref = load_ref("puic300_smoke.csv")
    assert W.shape == (300, 300)
    a, b = aligned(W, ref)
    assert spearman_flat(a, b) >= 0.999
    assert np.abs(a - b).max() <= 0.03          # measured 0.02
    assert ((a >= 0) & (a <= 1)).all()          # ECDF mean values


@pytest.mark.fast
def test_puic_thread_determinism_bitwise(exp300):
    w1 = run_puic(exp300, n_threads=1)
    w8 = run_puic(exp300, n_threads=8)
    assert np.array_equal(w1.to_numpy(), w8.to_numpy())
    w_again = run_puic(exp300, n_threads=8)
    assert np.array_equal(w8.to_numpy(), w_again.to_numpy())


@pytest.mark.fast
def test_puic_via_infer_grn(exp300):
    W = infer_grn(exp300, method="puic", threads=4)
    direct = run_puic(exp300, n_threads=4)
    assert np.array_equal(W.to_numpy(), direct.to_numpy())
    assert list(W.index) == list(exp300.index) == list(W.columns)
