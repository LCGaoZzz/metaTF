"""SINCERITIES rank parity vs the R reference (slow; numba required).

Campaign gate (round 4, rank_port): spearman_flat >= 0.95 AND per-target
top-50 overlap >= 0.7.  Measured: 0.9584 / 0.9453.  HONEST DISCLOSURE: the
numba build is ~0.5x the R runtime (slower than R on this host, 29.8 s vs
14.8 s) — kept because it is dependency-free; see docs/benchmarks.md.

Time axis: stage first-appearance ordinals AEC=1 .. Adult_HSC=6
(tests/data/test_col_data.csv, from repo test_col_data.rds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

numba = pytest.importorskip("numba", reason="sincerities requires numba")

from metatf.grn.sincerities import run_sincerities          # noqa: E402
from metatf import infer_grn                                 # noqa: E402

from conftest import aligned, load_ref, spearman_flat, top50_per_target_overlap


@pytest.mark.slow
def test_sincerities_rank_parity_300(exp300, col_data):
    res = run_sincerities(exp300, col_data)
    adj = pd.DataFrame(res["adj_matrix"], index=res["genes"], columns=res["genes"])
    ref = load_ref("sincerities300_smoke.csv")
    a, b = aligned(adj, ref)
    rho = spearman_flat(a, b)
    ov = top50_per_target_overlap(a, b)
    assert rho >= 0.95, rho                   # campaign rank_port gate
    assert ov >= 0.7, ov
    # normalized adjacency: max 1 (R: adj/max(adj)), diagonal included
    assert res["adj_matrix"].max() == pytest.approx(1.0, abs=1e-12)
    assert res["final_table"] is not None


@pytest.mark.slow
def test_sincerities_determinism_bitwise(exp300, col_data):
    r1 = run_sincerities(exp300, col_data, parallel=True)
    r2 = run_sincerities(exp300, col_data, parallel=True)
    assert np.array_equal(r1["adj_matrix"], r2["adj_matrix"])
    assert np.array_equal(r1["DISTANCE_matrix"], r2["DISTANCE_matrix"])
    # parallel vs serial must produce identical bits (task-set fixed)
    r3 = run_sincerities(exp300, col_data, parallel=False)
    assert np.array_equal(r1["adj_matrix"], r3["adj_matrix"])


@pytest.mark.slow
def test_sincerities_needs_col_data(exp300):
    with pytest.raises(ValueError):
        infer_grn(exp300, "sincerities")


@pytest.mark.slow
def test_sincerities_rejects_two_time_points(exp300, col_data):
    cd2 = col_data[col_data["time"].isin([1.0, 6.0])]
    exp2 = exp300.loc[:, cd2["sample"]]
    with pytest.raises(ValueError):
        run_sincerities(exp2, cd2)
