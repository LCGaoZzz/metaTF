"""py_aucell v2 — AUCell regulon activity, replicated from the AUCell package
semantics as invoked by metaTF runAUCell() (repo/R/regulonActivity.R L180-220):

    AUCell_buildRankings(exprMat, plotStats=FALSE)      # per-cell gene ranks
    AUCell_calcAUC(geneSets, rankings, normAUC=normalization)
    # aucMaxRank default = ceiling(0.05 * nrow(rankings))   (anchor aucell-3)

Core algorithm (anchor aucell-2, SCENIC Nat Methods 2017):
    For each cell, genes are ranked by expression descending (rank 1 =
    highest; ties = average rank / midrank).  The recovery curve counts, at
    each rank position t = 1..aucMaxRank, how many gene-set genes have been
    recovered.  AUC = area under that staircase:

        AUC(set, cell) = sum_{g in set, rank_g <= T} (T - rank_g + 1)

    which is exactly sum_{t=1..T} hits(t).  With normAUC=TRUE the score is
    divided by the maximal attainable AUC (all set genes at the very top):

        maxAUC = sum_{k=1..min(m, T)} (T - k + 1)
               = m*T - m*(m-1)/2          (when m <= T)

Attendance rule — THE round3 boundary fix (ISSUE-FACTS/r-intermediates.md §3):
    a gene set is dropped when its missing-gene fraction is >= 0.8 — the
    EQUALITY case is dropped too (`missingPercent >= 0.8`, R AUCell boundary).
    round2 used a strict `>` and wrongly kept sets at exactly 80% missing
    (on the real smoke inputs this is the Tfap2d/Zfp445 class).  If EVERY set
    is dropped, an error is raised.

Everything else is byte-identical in semantics to round2 (midrank average
ties, aucMaxRank = ceil(0.05·G), staircase normAUC denominator — all three
now CONFIRMED by anchor §3, which also names exactly these).

Memory: cells are streamed one at a time — the peak is a single rank vector
of length n_genes plus the output matrix (|sets| x n_cells), never the
14169 x 154 full ranking matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def run_aucell(exp, gene_sets, auc_max_rank=None, auc_max_rank_frac=0.05,
               norm_auc=True, missing_percent=0.8, ties="average"):
    """Compute AUCell AUC / normalized AUC for every (set, cell).

    Parameters
    ----------
    exp : pandas.DataFrame, genes x cells.
    gene_sets : dict {set_name: iterable of gene symbols}.
    auc_max_rank : explicit T; default ceil(auc_max_rank_frac * n_genes).
    norm_auc : normalize by maxAUC (AUCell normAUC=TRUE).
    missing_percent : drop sets whose missing-gene fraction is >= this
        (default 0.8 — exactly-80%-missing sets ARE dropped, anchor §3).
    ties : 'average' (midrank, default) or 'min'.
    """
    if not isinstance(exp, pd.DataFrame):
        raise TypeError("exp must be a genes x cells pandas DataFrame")
    X = np.ascontiguousarray(exp.to_numpy(dtype=np.float64))
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains NaN/Inf")
    n_genes, n_cells = X.shape

    gene_pos = {g: i for i, g in enumerate(exp.index)}

    # ---- attendance rule (>= boundary, anchor §3) ----------------------------
    matched, kept_names = [], []
    for name, genes in gene_sets.items():
        uniq = list(dict.fromkeys(genes))                     # unique, keep order
        idx = [gene_pos[g] for g in uniq if g in gene_pos]
        missing_frac = 1.0 - len(idx) / len(uniq) if uniq else 1.0
        if missing_frac >= missing_percent:                   # equality DROPS (v2)
            continue
        matched.append(np.asarray(idx, dtype=np.int64))
        kept_names.append(name)
    if len(kept_names) == 0:
        raise ValueError(
            "all gene sets meet the missing-gene threshold "
            f"(missing_percent >= {missing_percent}): nothing to compute")

    # ---- aucMaxRank ---------------------------------------------------------
    T = int(np.ceil(auc_max_rank_frac * n_genes)) if auc_max_rank is None else int(auc_max_rank)
    if T < 1:
        raise ValueError("aucMaxRank must be >= 1")
    if T > n_genes:
        raise ValueError("aucMaxRank cannot exceed the number of genes")

    # concatenated target indices for vectorized per-cell scoring
    sizes = np.array([len(m) for m in matched], dtype=np.int64)
    if np.any(sizes == 0):
        raise ValueError("compliant gene sets must contain at least one gene present in the matrix")
    offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    concat_idx = np.concatenate(matched)

    m_eff = sizes.astype(np.float64)                       # matched set sizes
    k = np.minimum(m_eff, T)
    max_auc = k * T - k * (k - 1.0) / 2.0                  # staircase maximum

    out = np.empty((len(kept_names), n_cells), dtype=np.float64)
    for j in range(n_cells):
        r_desc = stats.rankdata(-X[:, j], method=ties)     # 1 = highest expression
        r = r_desc[concat_idx]
        rec = (T - r + 1.0)
        rec[r > T] = 0.0
        out[:, j] = np.add.reduceat(rec, offsets)

    if norm_auc:
        out = out / max_auc[:, None]

    return pd.DataFrame(out, index=pd.Index(kept_names, name="gene_set"),
                        columns=exp.columns)
