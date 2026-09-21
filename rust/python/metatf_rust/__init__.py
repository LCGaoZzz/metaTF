"""metatf_rust — aggregate Rust extension for the metatf package.

Submodules (verbatim campaign-winner crates, exposed as attributes by the
cdylib's #[pymodule]):
    rust_pcor_v2.partial_corr_spearman(mat, n_threads, tol_rel)
    rust_puic.puic(exp_mat, row_names, regulators, targets, log_scale, n_threads, diag)
    rust_genie3.genie3_targets(...) / mt_first_unifs / r_sum

This __init__ deliberately does NOT use maturin's generated template
(`from .x import *` + bare `x.__doc__`): that template binds the submodule
name only through an import side effect and raises NameError when
sys.modules holds a stale submodule entry (shared-kernel state). Loading
via importlib and taking attributes off the returned module cannot fail in
any sys.modules state (round-2 bug, fixed in round 3, preserved here).
"""
import importlib

_ext = importlib.import_module(".metatf_rust", __name__)   # the compiled cdylib

rust_pcor_v2 = _ext.rust_pcor_v2
rust_puic = _ext.rust_puic
rust_genie3 = _ext.rust_genie3

__doc__ = _ext.__doc__
__all__ = ["rust_pcor_v2", "rust_puic", "rust_genie3"]
