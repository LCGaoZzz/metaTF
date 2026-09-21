"""PUIC GRN inference — Python wrapper over the rust_puic crate.

Campaign round 3 winner (candidates/round3/rust_puic): 179-213x vs R,
flat Spearman ~0.999998 vs the R reference.  No pure-Python fallback exists;
without a Rust backend this raises with install instructions.

The wrapper reproduces the upstream repo/R/PUIC.R dispatch contract
(inferWeightExpMat L158-205): optional log2(x+1), per-gene equal-width
discretization B=floor(sqrt(C)), MI + specific-information decomposition,
redundancy, PUC, .FUxy ECDF symmetrization; exp_cutoff filtering lives in
metatf.api.infer_grn (upstream applies it before dispatch).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .._rust import rust_backend

__all__ = ["run_puic", "HAVE_RUST_PUIC"]


def _puic_fn():
    try:
        return rust_backend("puic").puic
    except RuntimeError:
        return None


HAVE_RUST_PUIC = _puic_fn() is not None


def run_puic(exp: pd.DataFrame, regulators: Optional[Sequence[str]] = None,
             targets: Optional[Sequence[str]] = None, log_scale: bool = False,
             n_threads: int = 8, diag: str = "auto") -> pd.DataFrame:
    """PUIC(expMat, regulators, targets, logScale, ncores, diag).

    Returns regulator (rows) x target (columns) weight matrix (ECDF mean,
    values in [0, 1]).
    """
    fn = _puic_fn()
    if fn is None:
        raise RuntimeError(
            "method 'puic' requires the Rust extension (rust_puic). "
            "Build it with ./build.sh from the metatf-py repository root "
            "(cargo + maturin; see docs/BUILD.md) or `pip install metatf-rust`. "
            "Only pcor has a pure-NumPy fallback.")
    mat = np.ascontiguousarray(exp.to_numpy(dtype=np.float64))
    W, reg_names, tgt_names = fn(
        mat,
        [str(x) for x in exp.index],
        None if regulators is None else [str(x) for x in regulators],
        None if targets is None else [str(x) for x in targets],
        bool(log_scale),
        int(n_threads),
        diag,
    )
    return pd.DataFrame(W, index=reg_names, columns=tgt_names)
