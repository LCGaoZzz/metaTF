"""IO adapters: expression matrices, networks, regulon lists, outputs.

Conventions used everywhere in metatf:
  * expression  : pandas.DataFrame, genes (rows) x cells (columns)
  * network     : pandas.DataFrame with columns tf / target / weight (an edge
                  table, one row per (tf, target) edge; duplicate edges are
                  meaningful for viper — see docs/semantics.md)
  * regulons    : dict {tf: [targets]} (unweighted, R ``dfToList`` semantics)
                  or {tf: {target: weight}} (weighted)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "read_expression",
    "read_network",
    "write_table",
    "df_to_regulons",
    "regulons_to_net",
]

PathLike = Union[str, Path]


def read_expression(path: PathLike) -> pd.DataFrame:
    """Read a genes x cells expression matrix.

    Supported: .csv / .tsv / .csv.gz / .tsv.gz / .parquet (genes as the first
    column / index), and .h5ad (AnnData; cells x genes is transposed;
    requires the optional ``anndata`` package).
    """
    p = Path(path)
    suffix = "".join(p.suffixes[-2:]) if p.name.endswith((".csv.gz", ".tsv.gz")) else p.suffix
    if suffix == ".h5ad":
        try:
            import anndata as ad
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading .h5ad requires the optional dependency 'anndata' "
                              "(pip install 'metatf[anndata]')") from e
        a = ad.read_h5ad(p)
        X = a.X
        if hasattr(X, "toarray"):          # scipy sparse
            X = X.toarray()
        exp = pd.DataFrame(np.asarray(X).T, index=[str(v) for v in a.var_names],
                           columns=[str(o) for o in a.obs_names])
        return exp
    if suffix in (".parquet",):
        df = pd.read_parquet(p)
    elif suffix in (".tsv", ".tsv.gz"):
        df = pd.read_csv(p, sep="\t", index_col=0, float_precision="round_trip")
    else:
        # round_trip: pandas' default fast float parser can be 1 ulp off
        df = pd.read_csv(p, index_col=0, float_precision="round_trip")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def read_network(path: PathLike) -> pd.DataFrame:
    """Read a tf/target/weight edge table (.csv/.tsv/.csv.gz/.parquet).

    Column names are matched case-insensitively; 'weight' is optional
    (filled with 1.0 when absent).
    """
    p = Path(path)
    suffix = "".join(p.suffixes[-2:]) if p.name.endswith((".csv.gz", ".tsv.gz")) else p.suffix
    if suffix == ".parquet":
        net = pd.read_parquet(p)
    elif suffix in (".tsv", ".tsv.gz"):
        net = pd.read_csv(p, sep="\t")
    else:
        net = pd.read_csv(p)
    net.columns = [str(c).lower() for c in net.columns]
    for c in ("tf", "target"):
        if c not in net.columns:
            raise ValueError(f"network file {p} is missing column {c!r}")
    if "weight" not in net.columns:
        net["weight"] = 1.0
    net["tf"] = net["tf"].astype(str)
    net["target"] = net["target"].astype(str)
    net["weight"] = net["weight"].astype(float)
    return net[["tf", "target", "weight"]]


def write_table(df: pd.DataFrame, path: PathLike) -> None:
    """Write a result DataFrame to csv / csv.gz / tsv / parquet (by suffix)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p)
    elif p.suffixes[-1:] == [".tsv"] or p.name.endswith(".tsv.gz"):
        df.to_csv(p, sep="\t")
    else:
        df.to_csv(p)


def df_to_regulons(net: pd.DataFrame, weighted: bool = False) -> Dict:
    """metaTF ``dfToList`` semantics (repo/R/utils.R, verified):

    * unweighted (default): ``split(as.character(target), tf)`` then
      ``lapply(unique)`` — one CHARACTER vector of unique targets per TF
      (duplicate targets within a TF collapsed, first-appearance order kept),
      TF keys ordered alphabetically (R ``split`` group order). The weight
      column is DROPPED — this is upstream behaviour, recorded as a semantic
      divergence in docs/semantics.md.
    * weighted: numeric vector of weights named by target, duplicates KEPT
      (R named-vector semantics; viper counts duplicated edges twice).
    """
    net = net.copy()
    net.columns = [str(c).lower() for c in net.columns]
    if not {"tf", "target"} <= set(net.columns):
        raise ValueError("net must contain columns 'tf' and 'target'")
    if weighted:
        if "weight" not in net.columns:
            raise ValueError("weighted=True needs a 'weight' column")
        out: Dict = {}
        for tf, d in net.groupby("tf", sort=True):   # R split: alphabetical keys
            out[str(tf)] = pd.Series(d["weight"].to_numpy(float),
                                     index=d["target"].astype(str).to_numpy())
        return out
    regulons = {}
    for tf, d in net.groupby("tf", sort=True):
        seen, uniq = set(), []
        for t in d["target"].astype(str):
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        regulons[str(tf)] = uniq
    return regulons


def regulons_to_net(regulons: Dict) -> pd.DataFrame:
    """Inverse of df_to_regulons(weighted=False): regulon dict -> edge table."""
    rows = []
    for tf, targets in regulons.items():
        if isinstance(targets, dict) or isinstance(targets, pd.Series):
            items = list(zip(np.asarray(list(targets.index) if hasattr(targets, "index")
                                        else list(targets.keys())).astype(str),
                             np.asarray(list(targets.values), dtype=float)))
        else:
            items = [(str(g), 1.0) for g in targets]
        rows.extend((str(tf), g, w) for g, w in items)
    return pd.DataFrame(rows, columns=["tf", "target", "weight"])
