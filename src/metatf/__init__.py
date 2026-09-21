"""metatf — GRN inference and regulon activity without an R runtime.

Python/Rust reimplementation of the metaTF analysis chain (upstream R package
metaTF 0.1.1), assembled from the verified winners of the metaTF upgrade
campaign (rounds 2-4).  Two public entry points:

    >>> from metatf import infer_grn, regulon_activity
    >>> W = infer_grn(exp, method="pcor")              # pcor|puic|genie3|sincerities
    >>> nes = regulon_activity(exp, net, method="viper")  # viper|aucell|gsva|ulm

Semantics, parity evidence and known R-divergences: docs/semantics.md.
Benchmark table (R baseline vs this package): docs/benchmarks.md.

Rust accelerators are OPTIONAL: with only this package installed, `pcor`
falls back to a NumPy implementation of the same formula (cov-of-ranks +
ginv tolerance, campaign-verified); `puic` and `genie3` raise an error that
tells you how to install the extension wheel.  Install it with:

    pip install metatf-rust      # or ./build.sh from the repository root
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import infer_grn, regulon_activity          # noqa: F401
from .adapters import df_to_regulons                   # noqa: F401
from ._rust import rust_available, rust_backend        # noqa: F401
from .plotting import (plot_expression, plot_heatmap, plot_radviz,
                       plot_reduced_dim, plot_regulon_reduced_dim)  # noqa: F401

__all__ = [
    "infer_grn",
    "regulon_activity",
    "df_to_regulons",
    "rust_available",
    "rust_backend",
    "plot_heatmap",
    "plot_reduced_dim",
    "plot_regulon_reduced_dim",
    "plot_expression",
    "plot_radviz",
    "__version__",
]
