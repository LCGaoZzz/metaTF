"""metatf public API: infer_grn + regulon_activity.

Every method dispatches to the campaign-winning implementation, copied
verbatim into metatf.scorers / metatf.grn.  This module only adds:

  * unified signatures and row/column conventions
      - expression: DataFrame genes x cells (enforced)
      - GRN output: DataFrame regulator x target (pcor: genes x genes,
        symmetric, diag 1)
      - activity output: DataFrame sets/TFs x cells (ulm: also a pval frame)
  * optional Rust backends (metatf-rust wheel / legacy campaign wheels);
    pcor falls back to NumPy when no Rust backend is importable
  * metaTF upstream pre-processing: the inferWeightExpMat exp_cutoff gene
    filter (default 0 = keep rowSums > 0) is applied for every GRN method,
    exactly as in repo/R/inferGRNs.R L158-170.

Method-specific keyword arguments are passed through unchanged to the
verified implementations (documented per method below).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .adapters import df_to_regulons

__all__ = ["infer_grn", "regulon_activity"]

_THREADS_CAP = 63  # hard ceiling for threads (documented in docs/README.md)


def default_threads(n: Optional[int] = None) -> int:
    """Thread default: min(8, cpu_count) with a hard cap of 63 (R's seq/int32
    guard; rayon starts misbehaving above 63 on some NUMA hosts)."""
    import os
    cores = os.cpu_count() or 1
    if n is None:
        return min(8, cores, _THREADS_CAP)
    if n < 1:
        raise ValueError("threads must be >= 1")
    return min(n, _THREADS_CAP)


def _as_exp(exp) -> pd.DataFrame:
    if isinstance(exp, pd.DataFrame):
        return exp
    if hasattr(exp, "X") and hasattr(exp, "var_names"):  # AnnData
        X = exp.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return pd.DataFrame(np.asarray(X).T, index=[str(v) for v in exp.var_names],
                            columns=[str(o) for o in exp.obs_names])
    raise TypeError("expression must be a genes x cells pandas DataFrame "
                    "(or AnnData with the optional anndata extra)")


def _exp_cutoff_filter(exp: pd.DataFrame, exp_cutoff: float) -> pd.DataFrame:
    """repo/R/inferGRNs.R inferWeightExpMat L158-170, verbatim semantics."""
    if exp_cutoff == 0:
        keep = exp.sum(axis=1) > 0
    elif 0 < exp_cutoff < 1:
        keep = (exp > 0).sum(axis=1) > round(exp.shape[1] * exp_cutoff)
    elif exp_cutoff >= 1:
        keep = (exp > 0).sum(axis=1) > round(exp_cutoff)
    else:
        raise ValueError("exp_cutoff must be >= 0")
    out = exp.loc[keep]
    if out.shape[0] == 0:
        raise ValueError("No gene names found in your input expression matrix!")
    return out


# ===========================================================================
# GRN inference
# ===========================================================================
def infer_grn(
    expression,
    method: str = "pcor",
    *,
    regulators: Optional[Sequence[str]] = None,
    threads: Optional[int] = None,
    seed: int = 1,
    exp_cutoff: float = 0.0,
    col_data: Optional[pd.DataFrame] = None,
    **kwargs,
):
    """Infer a gene regulatory network.

    Parameters
    ----------
    expression : DataFrame genes x cells (or AnnData).
    method : 'pcor' | 'puic' | 'genie3' | 'sincerities'.
    regulators : restrict to these regulator genes (default: all genes).
    threads : worker threads; default min(8, cores), hard cap 63.
    seed : RNG seed for genie3 (R ``set.seed`` semantics; campaign refs use 1).
    exp_cutoff : metaTF inferWeightExpMat gene filter (default 0 = rowSums > 0).
    col_data : REQUIRED for method='sincerities': DataFrame with columns
        (sample, time); time must be numeric with >= 3 distinct points.
    kwargs : method-specific (see below).

    Returns
    -------
    DataFrame regulator x target weight matrix
    (pcor: genes x genes symmetric, diag 1 — the ppcor $estimate matrix).

    Method-specific kwargs
    ----------------------
    pcor     : tol_rel (float, rust only — ginv relative tolerance)
    puic     : log_scale (bool, default False), diag ('auto'|'zero'|'one')
    genie3   : n_trees (1000), tree_method ('RF'|'ET'), K ('sqrt'|'all'|int),
               mode ('fast'|'parity'; parity = bit-level R single-core stream)
    sincerities : distance (1), method (1), noDIAG (0), SIGN (1),
               CV_nfolds (5), nlambda (100), parallel (True)

    Notes
    -----
    pcor works without Rust (NumPy fallback, same cov-of-ranks + ginv formula,
    campaign-verified to 2.6e-14 vs the R reference). puic/genie3 raise a
    RuntimeError with install instructions when no Rust backend is present.
    """
    exp = _as_exp(expression)
    method = method.lower()
    exp_f = _exp_cutoff_filter(exp, exp_cutoff)

    if regulators is None:
        regs = None
    else:
        regs = [g for g in regulators if g in set(exp_f.index)]
        if len(regs) == 0:
            raise ValueError("No regulator detected while the 'regulators' list is not empty!")

    if method == "pcor":
        from .grn.pcor import partial_corr_spearman
        genes = exp_f.index
        # R: ppcor::pcor(t(exp_mat))$estimate — regulator subset applied AFTER
        M = partial_corr_spearman(
            np.ascontiguousarray(exp_f.to_numpy(dtype=np.float64)),
            n_threads=default_threads(threads), backend=kwargs.pop("backend", None))
        out = pd.DataFrame(M, index=genes, columns=genes)
        return out.loc[list(regs)] if regs is not None else out

    if method == "puic":
        from .grn.puic import run_puic
        return run_puic(exp_f, regulators=regs,
                        log_scale=kwargs.pop("log_scale", False),
                        n_threads=default_threads(threads),
                        diag=kwargs.pop("diag", "auto"))

    if method == "genie3":
        from .grn.genie3 import run_genie3
        W, reg_names, tgt_names = run_genie3(
            exp_f, regulators=regs,
            tree_method=kwargs.pop("tree_method", "RF"),
            K=kwargs.pop("K", "sqrt"),
            n_trees=kwargs.pop("n_trees", 1000),
            mode=kwargs.pop("mode", "fast"),
            n_threads=default_threads(threads),
            seed=seed)
        return pd.DataFrame(W, index=reg_names, columns=tgt_names)

    if method == "sincerities":
        from .grn.sincerities import run_sincerities
        if col_data is None:
            raise ValueError("method='sincerities' requires col_data "
                             "(DataFrame with sample + numeric time columns)")
        res = run_sincerities(
            exp_f, col_data,
            distance=kwargs.pop("distance", 1),
            method=kwargs.pop("method", 1),
            noDIAG=kwargs.pop("noDIAG", 0),
            SIGN=kwargs.pop("SIGN", 1),
            CV_nfolds=kwargs.pop("CV_nfolds", 5),
            seed=seed if seed is not None else 0,
            nlambda=kwargs.pop("nlambda", 100),
            parallel=kwargs.pop("parallel", True))
        adj = pd.DataFrame(res["adj_matrix"], index=res["genes"], columns=res["genes"])
        return adj.loc[list(regs)] if regs is not None else adj

    raise ValueError(f"unknown method {method!r} "
                     "(expected pcor | puic | genie3 | sincerities)")


# ===========================================================================
# Regulon activity
# ===========================================================================
def regulon_activity(
    expression,
    network: Optional[pd.DataFrame] = None,
    gene_sets: Optional[Dict] = None,
    method: str = "viper",
    *,
    weighted: Optional[bool] = None,
    minsize: int = 5,
    semantics: str = "subset",
    threads: Optional[int] = None,
    **kwargs,
):
    """Score regulon activity per cell.

    Parameters
    ----------
    expression : DataFrame genes x cells (or AnnData).
    network : edge table with tf/target/weight columns, OR
    gene_sets : regulon dict (e.g. from metatf.df_to_regulons).  Pass one.
    method : 'viper' | 'aucell' | 'gsva' | 'ulm'.
    weighted : viper only. None = auto (True when a weight column is present);
        True = network weights become aREA likelihoods (raw edge table kept,
        duplicate edges count separately — R named-vector semantics);
        False = all-1 likelihoods (the metaTF dfToList entry, targets
        deduplicated per TF).
    minsize : viper minimum regulon size after gene intersection (metaTF: 5);
        ulm minimum matched targets (min_target_num).
    semantics : ulm only — 'subset' (decoupler v1.9.2 formula, df = n_targets-2,
        default) or 'full_axis' (decoupler v2.2.0 formula, df = n_genes-2,
        zero-padded full-axis weights, NaN not guarded).  See
        docs/semantics.md before switching.
    threads : accepted for signature uniformity (viper/aucell/gsva are
        single-threaded NumPy; ulm likewise).

    Returns
    -------
    DataFrame sets/TFs x cells for viper (NES), aucell (normAUC), gsva (ES).
    For ulm: tuple (activity, pval), both TFs x cells (activity = t statistic).

    Method-specific kwargs
    ----------------------
    viper  : eset_filter (True), nes (True), cell_chunk (256), return_es (False)
    aucell : auc_max_rank (None -> ceil(0.05*n_genes)), norm_auc (True),
             missing_percent (0.8, >= drops), ties ('average')
    gsva   : min_sz (5), max_sz (500), tau (1.0), mx_diff (True), abs_ranking (False)
    ulm    : cell_chunk (256, subset only), eps (2.2e-16)
    """
    exp = _as_exp(expression)
    method = method.lower()
    have_net = network is not None
    have_gs = gene_sets is not None
    if have_net == have_gs:
        raise ValueError("pass exactly one of network / gene_sets")
    default_threads(threads)  # validates

    if method == "viper":
        from .scorers.viper_area import run_viper_area
        if weighted is None:
            weighted = bool(have_net and "weight" in [str(c).lower() for c in network.columns])
        if have_net:
            net = network.copy()
            net.columns = [str(c).lower() for c in net.columns]
            if weighted:
                # R weighted entry: named weight vector per TF, duplicate
                # (tf,target) edges KEPT (R named-vector indexing).
                return run_viper_area(exp, net=net, weighted=True, minsize=minsize,
                                      **kwargs)
            # R unweighted entry (metaTF runViper gene_list = dfToList(net)):
            # character vectors with targets DEDUPLICATED per TF.
            gs = df_to_regulons(net, weighted=False)
            return run_viper_area(exp, gene_list=gs, weighted=False, minsize=minsize,
                                  **kwargs)
        return run_viper_area(exp, gene_list=gene_sets, weighted=bool(weighted),
                              minsize=minsize, **kwargs)

    # aucell / gsva take unweighted gene-set lists (metaTF strips weights:
    # dfToList character vectors; documented divergence).
    gs = gene_sets
    if gs is None:
        net = network.copy()
        net.columns = [str(c).lower() for c in net.columns]
        gs = df_to_regulons(net, weighted=False)

    if method == "aucell":
        from .scorers.aucell import run_aucell
        return run_aucell(exp, gs, **kwargs)

    if method == "gsva":
        from .scorers.gsva import run_gsva
        return run_gsva(exp, gs, **kwargs)

    if method == "ulm":
        from .scorers.ulm import run_ulm
        if network is None:
            from .adapters import regulons_to_net
            network = regulons_to_net(gene_sets)
        return run_ulm(exp, network, min_target_num=minsize, semantics=semantics,
                       **kwargs)

    raise ValueError(f"unknown method {method!r} (expected viper | aucell | gsva | ulm)")
