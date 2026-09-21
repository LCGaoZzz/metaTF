"""CLI smoke tests (subprocess, so kernel module caches can't interfere)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """Small csv inputs derived from the vendored smoke data."""
    d = tmp_path_factory.mktemp("cli")
    exp = pd.read_csv(DATA / "exp_smoke.csv", index_col=0, float_precision="round_trip")
    # 300 genes x 12 cells keeps enough AUCell-compliant regulons on this network
    exp.iloc[:300, :12].to_csv(d / "exp.csv")
    net = pd.read_csv(DATA / "mm_blood_network.csv.gz")
    m = net["target"].isin(exp.index[:300])
    net[m].to_csv(d / "net.csv", index=False)
    pd.read_csv(DATA / "test_col_data.csv").head(12).to_csv(d / "cd.csv", index=False)
    return d


def _run(*argv):
    return subprocess.run([sys.executable, "-m", "metatf.cli", *argv],
                          capture_output=True, text=True, timeout=600)


@pytest.mark.fast
def test_cli_infer_grn_pcor(tiny_dir):
    out = tiny_dir / "W.csv"
    r = _run("infer-grn", "--method", "pcor", "--input", str(tiny_dir / "exp.csv"),
             "--output", str(out))
    assert r.returncode == 0, r.stderr
    W = pd.read_csv(out, index_col=0)
    assert W.shape[0] == W.shape[1]
    assert W.index[0] in W.columns


@pytest.mark.fast
def test_cli_activity_aucell_and_ulm(tiny_dir):
    out = tiny_dir / "auc.csv"
    r = _run("activity", "--method", "aucell", "--input", str(tiny_dir / "exp.csv"),
             "--network", str(tiny_dir / "net.csv"), "--output", str(out))
    assert r.returncode == 0, r.stderr
    A = pd.read_csv(out, index_col=0)
    assert A.shape[1] == 12 and A.shape[0] >= 1

    tout = tiny_dir / "ulm_t.csv"
    r = _run("activity", "--method", "ulm", "--input", str(tiny_dir / "exp.csv"),
             "--network", str(tiny_dir / "net.csv"), "--output", str(tout),
             "--semantics", "full_axis")
    assert r.returncode == 0, r.stderr
    T = pd.read_csv(tout, index_col=0)
    P = pd.read_csv(tiny_dir / "ulm_t_pval.csv", index_col=0)
    assert T.shape == P.shape and T.shape[1] == 12


@pytest.mark.fast
def test_cli_errors(tiny_dir):
    r = _run("infer-grn", "--method", "sincerities", "--input",
             str(tiny_dir / "exp.csv"), "--output", str(tiny_dir / "x.csv"))
    assert r.returncode != 0 and "col-data" in r.stderr
    r = _run("activity", "--method", "nope", "--input", str(tiny_dir / "exp.csv"),
             "--network", str(tiny_dir / "net.csv"), "--output", str(tiny_dir / "y.csv"))
    assert r.returncode != 0
