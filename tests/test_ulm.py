"""ULM dual-semantics contract tests.

subset     = decoupler v1.9.2 documented formula: r over the TF's matched
             target subset, df = n_targets - 2, constant-weight guard t=0/p=1
             (round2/py_ulm semantics, bit-preserved).
full_axis  = decoupler v2.2.0 implementation: zero-padded full-gene-axis
             weights, grand-mean centered Pearson, df = n_genes - 2, NO guard
             (sd=0 -> NaN propagates, as in decoupler).

Campaign evidence (round 4): full_axis closes to decoupler 2.2.0 run_ulm to
5.4e-12 (t) / 8.1e-13 (raw p) on exp_example x mm_blood — verified against
the installed decoupler, transcribed here as an independent recompute on a
spot-check slice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metatf.scorers.ulm import run_ulm, run_ulm_full_axis, EPS


@pytest.mark.fast
def test_ulm_rejects_unknown_semantics(exp_smoke, net):
    with pytest.raises(ValueError):
        run_ulm(exp_smoke, net, semantics="bogus")


@pytest.mark.fast
def test_ulm_subset_guard_vs_full_axis_defined(exp_smoke, net):
    """THE dual-semantics contract on constant-weight TFs.

    subset: with EXACTLY representable constant weights the den>0 guard fires
    (t = 0, p = 1); with arbitrary float weights (this network) the numerator
    and denominator are both float noise ~1e-13, giving |t| ~1e-8, p ~ 1 —
    i.e. subset reports NO activity for constant-weight TFs either way.
    full_axis: zero-padding makes those weight columns non-constant across
    the gene axis, so decoupler 2.2.0 arithmetic gives DEFINED t values.
    """
    act_sub, p_sub = run_ulm(exp_smoke, net, semantics="subset")
    act_full, p_full = run_ulm(exp_smoke, net, semantics="full_axis")
    w = net.groupby("tf")["weight"].nunique()
    const_tfs = [t for t in act_sub.index if t in act_full.index and w.get(t, 2) == 1]
    assert len(const_tfs) > 10          # mm_blood has ~176 such TFs post-filter
    const_tfs = const_tfs[:60]
    sub_t = act_sub.loc[const_tfs].to_numpy()
    sub_p = p_sub.loc[const_tfs].to_numpy()
    assert np.abs(sub_t).max() <= 1e-6, np.abs(sub_t).max()
    assert sub_p.min() >= 1.0 - 1e-6
    # full_axis: defined, finite, materially nonzero
    full_t = act_full.loc[const_tfs].to_numpy()
    assert np.isfinite(full_t).all()
    assert np.abs(full_t).max() > 1.0


@pytest.mark.fast
def test_ulm_subset_exact_guard_fires():
    """Weights of exactly 1.0: sums are exact, den == 0.0, guard t=0/p=1."""
    genes = [f"g{i}" for i in range(8)]
    rng = np.random.default_rng(3)
    exp = pd.DataFrame(rng.integers(0, 50, size=(8, 6)).astype(float),
                       index=genes, columns=[f"c{j}" for j in range(6)])
    net = pd.DataFrame({"tf": ["T"] * 6, "target": genes[:6],
                        "weight": [1.0] * 6})
    act, pv = run_ulm(exp, net, semantics="subset")
    assert np.all(act.loc["T"].to_numpy() == 0.0)
    assert np.all(pv.loc["T"].to_numpy() == 1.0)


@pytest.mark.fast
def test_ulm_full_axis_matches_decoupler_formula(exp_smoke, net):
    """Independent transcription of decoupler 2.2.0 _cov/_cor/_tval on a slice."""
    exp = exp_smoke.iloc[:200, :8]
    act, pv = run_ulm_full_axis(exp, net)
    # keep a few TFs with >= 5 matched targets
    genes = set(exp.index)
    m = net[net["target"].isin(genes)]
    cnt = m.groupby("tf")["target"].nunique()
    tfs = sorted([t for t in cnt.index if cnt[t] >= 5 and t in act.index])[:5]
    X = exp.to_numpy(dtype=np.float64)                       # genes x cells
    n_genes = X.shape[0]
    df = n_genes - 2
    for tf in tfs:
        sub = m[m["tf"] == tf]
        # duplicate (tf,target) averaged (metaTF-side prepare convention)
        w = sub.groupby("target")["weight"].mean()
        wcol = np.zeros(n_genes)
        for g, val in w.items():
            if g in genes:
                wcol[list(exp.index).index(g)] = val
        b = X
        cov = (b.T - b.mean()) @ (wcol - wcol.mean()) / (b.shape[0] - 1)
        ssd = np.std(wcol, ddof=1) * np.std(b, axis=0, ddof=1)
        r = cov / ssd
        t = r * np.sqrt(df / ((1 - r + 2.2e-16) * (1 + r + 2.2e-16)))
        assert np.allclose(act.loc[tf].to_numpy(), t, atol=1e-10)
        from scipy import stats
        p = stats.t.sf(np.abs(t), df) * 2
        assert np.allclose(pv.loc[tf].to_numpy(), p, atol=1e-12)


@pytest.mark.fast
def test_ulm_determinism_both_semantics(exp_smoke, net):
    exp = exp_smoke.iloc[:, :40]
    a1, p1 = run_ulm(exp, net, semantics="subset")
    a2, p2 = run_ulm(exp, net, semantics="subset")
    assert np.array_equal(a1.to_numpy(), a2.to_numpy())
    assert np.array_equal(p1.to_numpy(), p2.to_numpy())
    f1, fp1 = run_ulm(exp, net, semantics="full_axis")
    f2, fp2 = run_ulm(exp, net, semantics="full_axis")
    assert np.array_equal(f1.to_numpy(), f2.to_numpy())
    assert np.array_equal(fp1.to_numpy(), fp2.to_numpy())


@pytest.mark.fast
def test_ulm_hand_check_two_targets():
    genes = ["a", "b", "c", "d", "e", "f"]
    exp = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 2.0], [4.0, 5.0], [0.0, 1.0], [2.0, 2.0]],
        index=genes, columns=["c1", "c2"])
    net = pd.DataFrame({"tf": ["T"] * 3, "target": ["a", "b", "c"],
                        "weight": [1.0, 2.0, 3.0]})
    act, pval = run_ulm(exp, net, min_target_num=3, semantics="subset")
    from scipy import stats as st
    r = np.corrcoef(net["weight"].to_numpy(), exp.loc[["a", "b", "c"]].to_numpy()[:, 0])[0, 1]
    t_expect = r * np.sqrt(1.0 / ((1 - r + EPS) * (1 + r + EPS)))
    assert act.loc["T", "c1"] == pytest.approx(t_expect, abs=1e-12)
    assert pval.loc["T", "c1"] == pytest.approx(2 * st.t.sf(abs(t_expect), 1), abs=1e-12)
