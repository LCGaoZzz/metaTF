"""py_viper_area v2 — full-chain rewrite of the aREA path of viper::viper,
line-anchored to ISSUE-FACTS/r-intermediates.md §2 (verified from the installed
viper package source), for the exact call metaTF runViper() makes:

    viper(eset = exp,
          regulon = list(tfmode = +1 (all targets), likelihood = weights | all 1),
          nes = TRUE, minsize = 5, eset.filter = TRUE, cores,
          method = "none", dnull = NULL, pleiotropy = FALSE, adaptive.size = FALSE)

Executed chain (step numbers follow anchor §2):
  3. eset.filter: tmp = TF names ∪ ALL target names (of the regulon as passed);
     tt = tt[rownames(tt) %in% tmp, ] — the expression matrix is filtered to
     the regulon union BEFORE anything else. n for the plotting position is
     this filtered row count.
  4. per regulon: targets ∩ rownames(tt); surviving size < minsize(5) → the
     regulon is dropped.
  5. aREA(tt, regulon, minsize=0):
       w_norm1 = likelihood / max(likelihood)                    (per regulon)
       wts     = w_norm1 / colSums(w_norm1)                      (sum to 1)
       t2      = qnorm(rank_asc,average(tt, per cell)/(nrow(tt)+1))
       t1      = |u-0.5|*2 ; t1 += (1-colMax(t1))/2 ; t1 = qnorm(t1),
                 with u = rank/(n+1) — under tfmode=+1 this term's coefficient
                 (1-|mor|) is exactly 0 (no contribution; formula kept verbatim)
       sum1    = Σ_targets (mor·wts·t2)
       sum2    = Σ_targets ((1-|mor|)·wts·t1)              == 0 under mor=+1
       es      = (|sum1| + sum2·1[sum2>0]) · sign(sum1),  sign(0)=1
       nes     = es · sqrt(colSums(w_norm1^2))
  6. dnull=NULL → nes used as produced; no pleiotropy, no adaptive size, no
     additional processing of any kind beyond anchor §2.

Likelihoods: weighted entry → network weight; unweighted entry (character
regulons) → all 1 ("等权入口 likelihood=全1"). Unweighted regulons reduce to
nes = mean(t2)·sqrt(m), the textbook form.

Duplicate (tf,target) edges are KEPT as separate likelihood entries — the R
named-vector semantics (`t2[names(x$tfmode), ]` indexes duplicated row names
twice); round2's groupby-mean dedupe is removed.

Vectorization: all per-regulon weighted sums are performed as ONE sparse
CSR matrix product per (cell-chunk × quantile matrix); no python loop over
regulons anywhere in the numeric path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats


def _regulon_table(net=None, gene_list=None, weighted=True) -> pd.DataFrame:
    """(tf, target, likelihood) edge table in first-appearance TF order."""
    if net is not None:
        net = net.copy()
        cols = {str(c).lower(): c for c in net.columns}
        if not {"tf", "target"} <= set(cols):
            raise ValueError("net must contain columns 'tf' and 'target'")
        tf_col, tg_col = cols["tf"], cols["target"]
        if weighted and "weight" in cols:
            lik = net[cols["weight"]].to_numpy(dtype=float)
        else:                      # unweighted entry: likelihood all 1
            lik = np.ones(len(net), dtype=float)
        tbl = pd.DataFrame({
            "tf": net[tf_col].astype(str).to_numpy(),
            "target": net[tg_col].astype(str).to_numpy(),
            "likelihood": lik,
        })
    else:
        rows = []
        for tf, targets in gene_list.items():
            if isinstance(targets, dict):
                items = list(targets.items())
            elif hasattr(targets, "index") and hasattr(targets, "values"):
                items = list(zip(np.asarray(targets.index).astype(str),
                                 np.asarray(targets.values, dtype=float)))
            else:
                items = [(str(g), 1.0) for g in targets]
            for g, w in items:
                rows.append((str(tf), str(g),
                             float(w) if (weighted and w is not None) else 1.0))
        tbl = pd.DataFrame(rows, columns=["tf", "target", "likelihood"])
    if len(tbl) == 0:
        raise ValueError("empty regulon input")
    return tbl


def run_viper_area(exp, net=None, gene_list=None, weighted=True, minsize=5,
                   eset_filter=True, nes=True, cell_chunk=256, return_es=False):
    """aREA activity (anchor §2 chain) for a genes x cells expression matrix.

    Parameters
    ----------
    exp : pandas.DataFrame, genes x cells.
    net : DataFrame with columns tf/target/weight (weighted entry), or
    gene_list : dict {tf: targets iterable | named weights} — pass exactly one.
    weighted : net weights → likelihood (True); all-1 likelihood (False).
    minsize : regulon minimum size AFTER target intersection (metaTF: 5).
    eset_filter : anchor §2.3 union filter (viper eset.filter=TRUE).
    nes : return nes (default, metaTF normalization=TRUE) instead of es.
    return_es : additionally return the raw es matrix.
    """
    if not isinstance(exp, pd.DataFrame):
        raise TypeError("exp must be a genes x cells pandas DataFrame")
    if (net is None) == (gene_list is None):
        raise ValueError("pass exactly one of net / gene_list")

    X = np.ascontiguousarray(exp.to_numpy(dtype=np.float64))
    gene_names = np.asarray(exp.index, dtype=object).astype(str)
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains NaN/Inf")
    if X.shape[1] == 0:
        raise ValueError("expression matrix has no cells")

    # ---- edge table ----------------------------------------------------------
    tbl = _regulon_table(net, gene_list, weighted)
    if not np.isfinite(tbl["likelihood"].to_numpy(float)).all():
        raise ValueError("non-finite likelihood/weight values")

    # ---- step 3: eset.filter — rows ∩ (TF names ∪ all target names) ----------
    if eset_filter:
        union = set(pd.unique(tbl["tf"])) | set(pd.unique(tbl["target"]))
        keep = np.asarray(pd.Index(gene_names).isin(union))
    else:
        keep = np.ones(len(gene_names), dtype=bool)
    X = X[keep]
    gene_names = gene_names[keep]
    n = len(gene_names)                      # plotting-position row count
    if n < 2:
        raise ValueError("eset.filter left fewer than 2 rows")

    # ---- step 4: per-regulon target intersection + minsize -------------------
    pos = pd.Series(np.arange(n), index=gene_names)
    tbl = tbl[tbl["target"].isin(pos.index)]      # duplicates of present genes kept
    size = tbl.groupby("tf", sort=False)["target"].transform("size")
    tbl = tbl[size >= minsize]
    if len(tbl) == 0:
        raise ValueError(f"no regulon reaches minsize={minsize} after intersection")

    tf_names = list(pd.unique(tbl["tf"]))          # first-appearance order (R order)
    tf_codes = pd.Series(np.arange(len(tf_names)), index=tf_names)
    cols = tf_codes.loc[tbl["tf"]].to_numpy()
    rows = pos.loc[tbl["target"]].to_numpy()       # duplicate targets → duplicate rows
    lik = tbl["likelihood"].to_numpy(dtype=float)

    # ---- step 5: aREA core, fully vectorized ---------------------------------
    # per-regulon sufficient statistics
    gsz = np.bincount(cols, minlength=len(tf_names)).astype(float)
    lsum = np.bincount(cols, weights=lik, minlength=len(tf_names))
    lsq = np.bincount(cols, weights=lik * lik, minlength=len(tf_names))
    lmax = np.full(len(tf_names), np.nan)
    np.fmax.at(lmax, cols, lik)                    # per-regulon max likelihood
    if np.any(lmax <= 0):
        raise ValueError("regulon with max(likelihood) <= 0 (need positive weights)")

    w_norm1_edge = lik / lmax[cols]                # w_norm1 per edge
    wn1_sum = lsum / lmax                          # Σ w_norm1 per regulon
    wn1_sq_sum = lsq / (lmax * lmax)               # Σ w_norm1^2 per regulon
    wts_edge = w_norm1_edge / wn1_sum[cols]        # wts (Σ = 1 per regulon)

    mor = 1.0                                      # tfmode = +1 (metaTF semantics)
    # sparse incidence: rows = filtered genes, cols = regulons
    M1 = sparse.csr_matrix((mor * wts_edge, (rows, cols)),
                           shape=(n, len(tf_names)))          # mor·wts
    M2 = sparse.csr_matrix(((1.0 - abs(mor)) * wts_edge, (rows, cols)),
                           shape=(n, len(tf_names)))          # (1-|mor|)·wts ≡ 0
    sqrt_wn1sq = np.sqrt(wn1_sq_sum)

    n_cells = X.shape[1]
    es_blocks = []
    for s in range(0, n_cells, cell_chunk):
        Ec = X[:, s:s + cell_chunk]                          # n x chunk
        r_asc = stats.rankdata(Ec, axis=0, method="average") # ascending, ties averaged
        u = r_asc / (n + 1.0)                                # plotting position
        t2q = stats.norm.ppf(u)
        t1p = np.abs(u - 0.5) * 2.0
        t1p = t1p + (1.0 - t1p.max(axis=0)) / 2.0            # per-cell shift
        t1q = stats.norm.ppf(t1p)
        sum1 = np.asarray(M1.T @ t2q, dtype=np.float64)      # regulons x chunk
        sum2 = np.asarray(M2.T @ t1q, dtype=np.float64)      # ≡ 0 under mor=+1
        sgn = np.sign(sum1)
        sgn[sgn == 0] = 1.0                                  # R sign(0)=1
        es = (np.abs(sum1) + np.where(sum2 > 0, sum2, 0.0)) * sgn
        es_blocks.append(es)

    es = np.hstack(es_blocks)                                # regulons x cells
    nes_mat = es * sqrt_wn1sq[:, None]

    idx = pd.Index(tf_names, name="tf")
    out_nes = pd.DataFrame(nes_mat, index=idx, columns=exp.columns)
    if return_es:
        return out_nes, pd.DataFrame(es, index=idx, columns=exp.columns)
    return out_nes if nes else pd.DataFrame(es, index=idx, columns=exp.columns)
