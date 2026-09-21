"""GENIE3 — Python wrapper over the rust_genie3 crate.

Campaign round 4 winner (candidates/round4/rust_genie3).  Bit-level faithful
Rust/PyO3 port of the GENIE3 1.24.0 BuildTreeEns C code including the R 4.3
Mersenne-Twister (set.seed path, fixup, get_random_integer consumption order).

Modes
-----
* mode='parity' (single thread): one MT stream seeded by set_seed(seed),
  replicating the R nCores=1 .C call sequence per target — bit-level equal to
  the R single-core reference (campaign: max |diff| 4.2e-16 on the 300-gene
  reference, bounded only by the CSV 15-digit quantization).
* mode='fast' (default, rayon): per-target independent MT streams
  (SplitMix64(master_seed, target ordinal)).  The deviation from parity is
  exactly the bootstrap sampling sequence; top-50 overlap vs the R 6-core
  reference is 0.80, on par with R's own run-to-run variation (0.93 vs
  itself).

Entry semantics (checked against ISSUE-FACTS/genie3-r-source.R + GENIE3.R):
  * exprMatrix genes x samples -> exprT = t(exprMatrix) (samples x genes)
  * per target: setdiff(regulatorNames, targetName) — order preserved,
    deduplicated, target itself removed (self-loop weight 0)
  * mtry = .setMtry(K, numRegulators): 'sqrt' -> round(sqrt(p)) (R round),
    'all' -> p, numeric -> K
  * as.single conversion at the .C boundary -> f32 inputs (column-major core
    table)
  * RF: nmin=1, sampling with replacement, IncNodePurity importance,
    divided by tree weight sum in C, then im/sum(im) with R's long-double
    accumulation replicated
  * exp_cutoff filter lives in metatf.api.infer_grn (upstream semantics)

No pure-Python fallback; without a Rust backend this raises with install
instructions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._rust import rust_backend, _ext

__all__ = ["run_genie3", "filter_exp_cutoff", "mt_selfcheck", "HAVE_RUST_GENIE3"]

# R 4.3.3 locally measured reference values (set.seed; runif), for mt_selfcheck.
_R_UNIF_REF: Dict[int, list] = {
    1: [0.26550866314209998, 0.37212389963679016, 0.57285336335189641,
        0.90820778999477625, 0.2016819310374558],
    123: [0.28757752012461424, 0.78830513544380665, 0.40897692181169987,
          0.88301740400493145, 0.9404672842938453],
    42: [0.91480604349635541, 0.93707541329786181, 0.28613953478634357,
         0.83044762606732547, 0.64174551889300346],
}


def _ext_fns():
    ext = _ext("rust_genie3")
    if ext is None:
        return None
    return ext


def _have() -> bool:
    try:
        rust_backend("genie3")
        return True
    except RuntimeError:
        return False


HAVE_RUST_GENIE3 = _have()


def mt_selfcheck() -> bool:
    """Bit-compare the Rust MT against local R 4.3.3 runif reference values."""
    ext = _ext_fns()
    if ext is None:
        raise RuntimeError("mt_selfcheck requires the rust_genie3 extension")
    ok = True
    for seed, ref in _R_UNIF_REF.items():
        got = ext.mt_first_unifs(seed, len(ref))
        same = all(a == b for a, b in zip(got.tolist(), ref))
        if not same:
            ok = False
    return ok


def filter_exp_cutoff(exp: pd.DataFrame, exp_cutoff: float = 0.0) -> pd.DataFrame:
    """exp_cutoff branch semantics (repo/R/inferGrNs.R L159-170):
    =0     -> keep rows with rowSums(exp) > 0 (default path)
    (0,1)  -> keep rows with #cells expressed > round(C * exp_cutoff)
    >=1    -> keep rows with #cells expressed > round(exp_cutoff)
    """
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


def _k_spec(K) -> str:
    """R .setMtry: numeric -> K; 'sqrt' -> round(sqrt(p)); otherwise p."""
    if isinstance(K, (int, float)) and not isinstance(K, bool):
        Ki = int(K)
        if Ki < 1:
            raise ValueError("K must be 'sqrt', 'all' or a strictly positive integer")
        return str(Ki)
    if K == "sqrt":
        return "sqrt"
    if K == "all":
        return "all"
    raise ValueError("K must be 'sqrt', 'all' or a strictly positive integer")


def _resolve_names(names: Sequence, all_names: list, what: str) -> list:
    """R semantics: character -> must exist in order (missing -> error);
    integer -> all_names[k] (1-based)."""
    out = []
    for x in names:
        if isinstance(x, (int, np.integer)):
            idx = int(x)
            if idx < 1 or idx > len(all_names):
                raise ValueError(f"{what} index out of range: {idx}")
            out.append(all_names[idx - 1])
        else:
            if x not in set(all_names):
                raise ValueError(f"{what} missing from the expression matrix: {x}")
            out.append(x)
    return out


def _setdiff_preserve(names: Sequence[str], target: str) -> list:
    """R setdiff(a, targetName): order preserved, deduplicated, target removed."""
    seen = set()
    out = []
    for g in names:
        if g == target or g in seen:
            continue
        seen.add(g)
        out.append(g)
    return out


def run_genie3(
    exp: pd.DataFrame,
    regulators: Optional[Sequence[str]] = None,
    targets: Optional[Sequence[str]] = None,
    tree_method: str = "RF",
    K: object = "sqrt",
    n_trees: int = 1000,
    mode: str = "fast",
    n_threads: Optional[int] = None,
    seed: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, list, list]:
    """GENIE3 main entry.  Returns (W, reg_names, tgt_names).

    W: regulator x target weight matrix (float64; each column normalised by
    im/sum(im), column sums 1; self-loops and non-selected regulators 0).
    parity mode replicates the R single-core reference stream bit-for-bit.
    """
    ext = _ext_fns()
    if ext is None:
        raise RuntimeError(
            "method 'genie3' requires the Rust extension (rust_genie3). "
            "Build it with ./build.sh from the metatf-py repository root "
            "(cargo + maturin; see docs/BUILD.md) or `pip install metatf-rust`. "
            "Only pcor has a pure-NumPy fallback.")

    genes = list(exp.index)
    gene_set = set(genes)
    if len(gene_set) != len(genes):
        raise ValueError("exp row names must be unique")
    if not np.isfinite(exp.to_numpy(dtype=np.float64)).all():
        raise ValueError("expression matrix contains non-finite values")
    if tree_method not in ("RF", "ET"):
        raise ValueError('tree_method must be "RF" or "ET"')
    if n_trees < 1:
        raise ValueError("n_trees must be >= 1")
    if mode not in ("parity", "fast"):
        raise ValueError('mode must be "parity" or "fast"')
    if not (0 <= seed <= 0xFFFFFFFF):
        raise ValueError("seed must fit in uint32 (R set.seed range)")
    k_spec = _k_spec(K)

    if regulators is None:
        regulator_names = genes
    else:
        regulator_names = _resolve_names(list(dict.fromkeys(regulators)), genes, "Regulator gene")
        if len(regulator_names) < 2:
            raise ValueError("Provide at least 2 potential regulators.")
    if targets is None:
        target_names = genes
    else:
        target_names = _resolve_names(list(dict.fromkeys(targets)), genes, "Target gene")

    pos = {g: i for i, g in enumerate(genes)}
    # as.single in one shot (f64->f32 rounding matches the R .C boundary)
    expr_t = np.ascontiguousarray(exp.to_numpy(dtype=np.float64).T, dtype=np.float32)

    reg_rows = list(dict.fromkeys(regulator_names))
    row_of = {g: i for i, g in enumerate(reg_rows)}
    regs_flat: list = []
    offsets: list = [0]
    tgt_pos: list = []
    these_lists: list = []
    for tname in target_names:
        these = _setdiff_preserve(regulator_names, tname)
        these_lists.append(these)
        regs_flat.extend(pos[g] for g in these)
        offsets.append(len(regs_flat))
        tgt_pos.append(pos[tname])
    reg_names = list(reg_rows)
    tgt_names = list(target_names)

    raw_flat, norm_flat = ext.genie3_targets(
        expr_t,
        np.asarray(regs_flat, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(tgt_pos, dtype=np.int64),
        int(n_trees),
        k_spec,
        tree_method,
        mode,
        int(seed),
        None if n_threads is None else int(n_threads),
    )
    norm_flat = np.asarray(norm_flat)

    W = np.zeros((len(reg_names), len(tgt_names)), dtype=np.float64)
    for j in range(len(tgt_names)):
        lo, hi = offsets[j], offsets[j + 1]
        rows = [row_of[g] for g in these_lists[j]]
        W[rows, j] = norm_flat[lo:hi]
    if verbose:
        print(f"[metatf.genie3] {len(tgt_names)} targets, mode={mode}, seed={seed}")
    return W, reg_names, tgt_names
