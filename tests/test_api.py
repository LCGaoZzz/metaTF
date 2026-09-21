"""metatf API surface: exports, validation, thread policy, backend reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metatf
from metatf import infer_grn, regulon_activity, df_to_regulons
from metatf.api import default_threads


@pytest.mark.fast
def test_version_and_exports():
    assert metatf.__version__ == "0.1.0"
    for name in ("infer_grn", "regulon_activity", "df_to_regulons",
                 "rust_available", "rust_backend"):
        assert hasattr(metatf, name)


@pytest.mark.fast
def test_rust_available_report():
    avail = metatf.rust_available()
    assert set(avail) == {"rust_pcor_v2", "rust_puic", "rust_genie3"}
    assert all(isinstance(v, bool) for v in avail.values())


@pytest.mark.fast
def test_thread_policy():
    assert default_threads(None) == min(8, __import__("os").cpu_count() or 1, 63)
    assert default_threads(2) == 2
    assert default_threads(1000) == 63            # hard cap
    with pytest.raises(ValueError):
        default_threads(0)


@pytest.mark.fast
def test_infer_grn_method_validation(exp_smoke):
    with pytest.raises(ValueError):
        infer_grn(exp_smoke.iloc[:20], method="aracne")


@pytest.mark.fast
def test_regulon_activity_exactly_one_source(exp_smoke, gene_sets):
    with pytest.raises(ValueError):
        regulon_activity(exp_smoke, method="viper")            # neither
    net = pd.DataFrame({"tf": ["a"], "target": ["b"], "weight": [1.0]})
    with pytest.raises(ValueError):
        regulon_activity(exp_smoke.iloc[:20], network=net, gene_sets=gene_sets,
                         method="viper")                       # both


@pytest.mark.fast
def test_df_to_regulons_matches_r_dflist(net):
    gs = df_to_regulons(net, weighted=False)
    # R split: alphabetical TF keys; unique targets per TF
    assert list(gs.keys()) == sorted(gs.keys())
    tf0 = sorted(net["tf"].astype(str).unique())[0]
    want = list(dict.fromkeys(net.loc[net["tf"].astype(str) == tf0, "target"].astype(str)))
    assert gs[tf0] == want
    # duplicated (tf,target) edges exist in this network and are collapsed
    assert net.duplicated(subset=["tf", "target"]).sum() > 0
    for tf, targets in list(gs.items())[:50]:
        assert len(targets) == len(set(targets))
    # weighted variant: duplicates KEPT (R named-vector semantics)
    gw = df_to_regulons(net, weighted=True)
    assert len(gw[tf0]) == int((net["tf"].astype(str) == tf0).sum())


@pytest.mark.fast
def test_exp_cutoff_filter_branches():
    exp = pd.DataFrame({"c1": [0.0, 1.0, 0.0], "c2": [0.0, 2.0, 3.0]},
                       index=["z0", "ok", "c0"])
    from metatf.api import _exp_cutoff_filter
    out = _exp_cutoff_filter(exp, 0.0)
    assert list(out.index) == ["ok", "c0"]
    out = _exp_cutoff_filter(exp, 1)          # cells expressed > 1
    assert set(out.index) == {"ok"}           # only 'ok' expresses in both cells
    with pytest.raises(ValueError):
        _exp_cutoff_filter(exp, -0.5)


@pytest.mark.fast
def test_pcor_regulator_subset(exp_smoke):
    sub = exp_smoke.iloc[:120]
    W = infer_grn(sub, method="pcor", backend="numpy", regulators=list(sub.index[:50]))
    assert W.shape == (50, 120)
    Wfull = infer_grn(sub, method="pcor", backend="numpy")
    assert np.array_equal(W.to_numpy(), Wfull.iloc[:50].to_numpy())
