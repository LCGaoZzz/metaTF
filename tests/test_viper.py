"""viper (aREA) parity vs the R reference + determinism.

Campaign gate (round 3, numeric_port): spearman_flat >= 0.995 and
top-100 per-cell flip <= 1%.  Measured on these exact vendored inputs:
u-entry spearman 0.9999999999998217 (max |d| 3.2e-14); w-entry spearman
0.99996 (max |d| 0.25 — the R reference is itself inconsistent on the 131
duplicated (tf,target) edges; see docs/semantics.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from metatf import regulon_activity
from metatf.scorers.viper_area import run_viper_area

from conftest import aligned, load_ref, spearman_flat, topk_flip_pct


@pytest.mark.fast
def test_viper_parity_unweighted(exp_smoke, gene_sets):
    res = run_viper_area(exp_smoke, gene_list=gene_sets, weighted=False)
    ref = load_ref("viper_smoke_u.csv")
    a, b = aligned(res, ref)
    rho = spearman_flat(a, b)
    assert a.shape == b.shape
    assert rho >= 0.9999, rho
    assert np.abs(a - b).max() < 1e-9
    assert topk_flip_pct(a, b) <= 0.01


@pytest.mark.fast
def test_viper_parity_weighted(exp_smoke, net):
    res = run_viper_area(exp_smoke, net=net, weighted=True)
    ref = load_ref("viper_smoke_w.csv")
    a, b = aligned(res, ref)
    assert spearman_flat(a, b) >= 0.995          # campaign numeric_port gate
    assert topk_flip_pct(a, b) <= 0.01


@pytest.mark.fast
def test_viper_determinism_bitwise(exp_smoke, net):
    r1 = run_viper_area(exp_smoke, net=net, weighted=True, cell_chunk=64)
    r2 = run_viper_area(exp_smoke, net=net, weighted=True, cell_chunk=64)
    assert np.array_equal(r1.to_numpy(), r2.to_numpy())
    # chunk size changes blocking only; identical within FP noise
    r3 = run_viper_area(exp_smoke, net=net, weighted=True, cell_chunk=1024)
    assert np.allclose(r3.to_numpy(), r2.to_numpy(), atol=1e-12)


@pytest.mark.fast
def test_viper_api_matches_direct(exp_smoke, net, gene_sets):
    via_api = regulon_activity(exp_smoke, network=net, method="viper", weighted=False)
    direct = run_viper_area(exp_smoke, gene_list=gene_sets, weighted=False)
    assert via_api.shape == direct.shape
    assert np.array_equal(via_api.to_numpy(), direct.to_numpy())
    # output orientation: regulons x cells, cells == exp columns
    assert via_api.shape[1] == exp_smoke.shape[1]
