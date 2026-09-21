"""Spearman partial correlation (ppcor::pcor $estimate parity).

Two interchangeable engines, same formula (anchor: cov-of-ranks + ginv):

  * Rust (rust_pcor_v2 crate, campaign round 3 winner): midrank -> CENTERED
    rank covariance (per-gene rank variances KEPT) -> Jacobi eigendecomposition
    on the dual Gram when cells <= genes -> STRICT ginv-tolerance truncation
    lambda > sqrt(.Machine$double.eps) * lambda_max (MASS::ginv semantics)
    -> -cov2cor, diag 1.  26x vs R on the 1000-gene smoke, max |delta|
    2.6e-14 vs the R reference.
  * NumPy fallback (this file, campaign-verified same-path reference):
    scipy.stats.rankdata 'average' -> np.cov(ddof=1) -> np.linalg.pinv(
    rcond=1.4901161193847656e-8) -> -cov2cor, diag 1.  No Rust needed.

The NumPy path is the exact reference the Rust crate was validated against
(runs/round3: max diff 1.4e-14 on the full real 1000x154 input), so results
agree to ~1e-14 across engines — well below the campaign parity gate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .._rust import _ext

__all__ = ["partial_corr_spearman", "pcor_numpy", "HAVE_RUST_PCOR"]

GINV_TOL = 1.4901161193847656e-8  # sqrt(f64 eps) == MASS::ginv default tol


def pcor_numpy(mat: np.ndarray, tol: float = GINV_TOL) -> np.ndarray:
    """NumPy same-path reference — COVARIANCE of ranks (not correlation!).

    R: ppcor::pcor(t(exp), method='spearman')$estimate with det(cvx) < eps ->
    MASS::ginv relative tolerance.  mat: genes x cells, C-contiguous f64.
    """
    Rk = stats.rankdata(mat, axis=1, method="average")
    cv = np.cov(Rk, rowvar=True, ddof=1)
    Theta = np.linalg.pinv(cv, rcond=tol)
    dd = np.sqrt(np.diag(Theta))
    P = -Theta / np.outer(dd, dd)
    np.fill_diagonal(P, 1.0)
    return P


def _rust_fn():
    ext = _ext("rust_pcor_v2")
    if ext is None:
        return None
    return getattr(ext, "partial_corr_spearman", None)


HAVE_RUST_PCOR = _rust_fn() is not None


def partial_corr_spearman(mat: np.ndarray, n_threads: Optional[int] = None,
                          tol_rel: Optional[float] = None,
                          backend: Optional[str] = None) -> np.ndarray:
    """Spearman partial-correlation matrix, genes x genes.

    backend: None (auto: Rust when importable, else NumPy) | 'rust' | 'numpy'.
    Constant rows raise ValueError (metaTF's rowSums>0 filter removes them
    upstream; R would propagate NA).
    """
    mat = np.ascontiguousarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 3:
        raise ValueError("mat must be genes x cells with >= 2 genes and >= 3 cells")
    if not np.isfinite(mat).all():
        raise ValueError("expression matrix contains NaN/Inf")
    if mat.shape[0] < 2:
        raise ValueError("need >= 2 genes")
    rng = np.ptp(mat, axis=1)
    if np.any(rng == 0):
        raise ValueError("constant gene rows: ppcor yields NA in R; "
                         "filter them first (metatf.infer_grn does this via exp_cutoff)")

    fn = _rust_fn()
    if backend == "numpy" or fn is None:
        if backend == "rust":
            raise RuntimeError("backend='rust' requested but rust_pcor_v2 is not "
                               "importable; build the extension with ./build.sh or "
                               "pip install metatf-rust")
        return pcor_numpy(mat)
    if n_threads is not None:
        if tol_rel is not None:
            return fn(mat, n_threads=int(n_threads), tol_rel=float(tol_rel))
        return fn(mat, n_threads=int(n_threads))
    return fn(mat)
