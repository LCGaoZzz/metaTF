"""AUCell parity + the >= 0.8 attendance boundary + determinism.

Campaign gate (round 3): spearman_flat >= 0.995, flip <= 1%.  Measured:
spearman 0.9963, max |d| 0.0296 on the vendored smoke inputs.  THE v2 fix:
sets at EXACTLY 80% missing genes are dropped (R `missingPercent >= 0.8`);
on the real inputs those are Tfap2d and Zfp445.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metatf.scorers.aucell import run_aucell

from conftest import aligned, load_ref, spearman_flat, topk_flip_pct


@pytest.mark.fast
def test_aucell_parity_and_boundary_drops(exp_smoke, gene_sets):
    res = run_aucell(exp_smoke, gene_sets)
    ref = load_ref("aucell_smoke2.csv")
    # exactly R's 14 kept sets (row ORDER may differ from R: R's split()
    # collation is locale-dependent — benign, see docs/semantics.md)
    assert set(res.index) == set(ref.index)
    assert res.shape == (14, 154)
    a, b = aligned(res, ref)
    rho = spearman_flat(a, b)
    assert rho >= 0.995, rho
    assert np.abs(a - b).max() <= 0.031            # measured 0.0296
    assert topk_flip_pct(a, b) <= 0.01


@pytest.mark.fast
def test_aucell_boundary_exact_80_percent_dropped():
    # 10 genes in the matrix. S80 lists 10 genes of which 8 are absent
    # (missing fraction EXACTLY 0.8) -> dropped (R `missingPercent >= 0.8`).
    # S80b misses 7 of 10 (0.7) -> kept.
    exp = pd.DataFrame(
        [[10, 0, 5], [9, 1, 4], [8, 2, 3], [7, 3, 2], [6, 4, 1],
         [5, 5, 0], [4, 6, 2], [3, 7, 3], [2, 8, 4], [1, 9, 5]],
        index=[f"g{i}" for i in range(10)], columns=[f"c{j}" for j in range(3)])
    s80 = ["g0", "g1"] + [f"x{i}" for i in range(8)]      # exactly 0.8 missing
    s70 = ["g0", "g1", "g2"] + [f"x{i}" for i in range(7)]  # 0.7 missing -> kept
    res = run_aucell(exp, {"S70": s70, "S80": s80})
    assert list(res.index) == ["S70"]
    # every set at exactly 0.8 -> ValueError
    with pytest.raises(ValueError):
        run_aucell(exp, {"only": s80})


@pytest.mark.fast
def test_aucell_hand_computed_staircase():
    # 5 genes x 1 cell, T = ceil(0.05*5) = 1: set {g0} (rank 1) -> AUC 1.0;
    # set {g1} (rank 2 > T) -> 0
    exp = pd.DataFrame({"c0": [5.0, 4.0, 3.0, 2.0, 1.0]},
                       index=[f"g{i}" for i in range(5)])
    res = run_aucell(exp, {"top": ["g0"], "second": ["g1"]}, auc_max_rank=1)
    assert res.loc["top", "c0"] == 1.0
    assert res.loc["second", "c0"] == 0.0


@pytest.mark.fast
def test_aucell_determinism_bitwise(exp_smoke, gene_sets):
    r1 = run_aucell(exp_smoke, gene_sets)
    r2 = run_aucell(exp_smoke, gene_sets)
    assert np.array_equal(r1.to_numpy(), r2.to_numpy())
