"""Shared fixtures + parity helpers for the metatf test suite.

Vendored inputs (derived from repo/inst/extdata test data, campaign round 1):
  exp_smoke.csv            1000 genes x 154 cells (test_exp_data.rds, counts)
  mm_blood_network.csv.gz  772,908 edges tf/target/weight (mm_blood_network.rds)
  test_col_data.csv        154 x (sample, time); time = stage first-appearance
                           ordinal AEC=1 .. Adult_HSC=6 (repo test_col_data.rds)

Vendored R references (runs/round1/refs, produced by bench/round1_runner.R):
  viper_smoke_u/w.csv      runViper unweighted/weighted entry, 874 x 154
  aucell_smoke2.csv        runAUCell, 14 x 154 (>=0.8-missing sets dropped)
  gsva_smoke2.csv          runGSVA kcdf=Poisson, 874 x 154
  genie3_300_c1.csv        GENIE3 set.seed(1) single core 1000 trees, 300 x 300
  puic300_smoke.csv        inferWeightExpMat PUIC, 300 x 300
  sincerities300_smoke.csv inferWeightExpMat sincerities, 300 x 300
  pcor_smoke_top300.csv    [0:300, 0:300] sub-block of the 1000 x 1000
                           inferWeightExpMat pcor reference (full ref is 19 MB;
                           the vendored block is EXACTLY the leading sub-matrix
                           of runs/round1/refs/pcor_smoke.csv — the test below
                           runs pcor on the FULL 1000-gene input and compares
                           its leading 300 x 300 block against this file)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent / "data"
REFS = DATA / "refs"

_READ = dict(float_precision="round_trip")


def load_ref(name: str) -> pd.DataFrame:
    return pd.read_csv(REFS / name, index_col=0, **_READ)


@pytest.fixture(scope="session")
def exp_smoke() -> pd.DataFrame:
    return pd.read_csv(DATA / "exp_smoke.csv", index_col=0, **_READ)


@pytest.fixture(scope="session")
def net() -> pd.DataFrame:
    return pd.read_csv(DATA / "mm_blood_network.csv.gz")


@pytest.fixture(scope="session")
def col_data() -> pd.DataFrame:
    return pd.read_csv(DATA / "test_col_data.csv")


@pytest.fixture(scope="session")
def gene_sets(net) -> dict:
    """dfToList(mm_blood_network) — R split + unique semantics."""
    from metatf import df_to_regulons
    return df_to_regulons(net, weighted=False)


@pytest.fixture(scope="session")
def exp300(exp_smoke) -> pd.DataFrame:
    return exp_smoke.iloc[:300]


def aligned(a: pd.DataFrame, b: pd.DataFrame):
    """Align two result frames on the R reference's rows/cols; return arrays."""
    common_rows = [r for r in b.index if r in a.index]
    common_cols = [c for c in b.columns if c in a.columns]
    assert common_rows and common_cols
    return (a.loc[common_rows, common_cols].to_numpy(),
            b.loc[common_rows, common_cols].to_numpy())


def spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    return float(stats.spearmanr(a.ravel(), b.ravel()).statistic)


def topk_flip_pct(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    """Campaign metric (tie-robust form): per-column top-k set disagreement.

    Membership is defined by dense rank <= k on each side (ties included), so
    constant / near-tied columns do not fabricate flips from arbitrary
    argsort ordering; disagreement is the Jaccard distance of the two sets.
    """
    from scipy import stats
    flips = []
    for j in range(a.shape[1]):
        ra = stats.rankdata(-a[:, j], method="min")
        rb = stats.rankdata(-b[:, j], method="min")
        sa = set(np.where(ra <= k)[0].tolist())
        sb = set(np.where(rb <= k)[0].tolist())
        union = sa | sb
        flips.append(1.0 - len(sa & sb) / len(union) if union else 0.0)
    return float(np.mean(flips))


def top50_per_target_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """GRN metric: mean per-target top-50 regulator overlap."""
    ovs = []
    for j in range(a.shape[1]):
        ka = set(np.argsort(-a[:, j])[:50].tolist())
        kb = set(np.argsort(-b[:, j])[:50].tolist())
        ovs.append(len(ka & kb) / 50.0)
    return float(np.mean(ovs))
