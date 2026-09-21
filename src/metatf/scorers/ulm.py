"""
py_ulm_dual — ULM 双语义（round4）：round2/py_ulm 的超集。

semantics="subset"    —— round2/py_ulm 现行为 = decoupler v1.9.2 文档公式
                        （ISSUE-FACTS/literature-anchors.md anchor ulm-sem-1 /
                        ulm-impl-1）：r 在 TF 匹配目标子集上计算，
                        df = n_targets − 2，常数权重/零表达有 0/1 守卫。
semantics="full_axis" —— decoupler v2.2.0 行为
                        （https://raw.githubusercontent.com/scverse/decoupler/
                        v2.2.0/src/decoupler/mt/_ulm.py 与 pp/net.py，
                        逐字取回存于本目录 upstream/）：网络权重零填充到
                        表达矩阵全基因轴（net_to_mat/_order：madjmat =
                        zeros(len(features), n_src)），Pearson r 在全轴上
                        计算（grand-mean 居中 + ddof=1，_cov/_cor 逐字），
                        df = n_genes − 2（"fitting {n_src} univariate models
                        of {n_var} observations (targets) with {df} degrees
                        of freedom"），sd=0（全轴常数权重）→ 0/0 = NaN 自然
                        传播（decoupler 不加守卫）。

两个模式共享 round2 的 metaTF 侧约定：min_target_num=5 按基因交集后的
唯一靶基因数、重复 (tf,target) 边取均值、weight==0 边剔除、输出 TF×cells、
行序按 TF 名排序。差异仅限 r 的计算轴与 df（见 MANIFEST）。

subset 路径 = round2/py_ulm/ulm.py 逐字复制（保证位一致）；full_axis 路径
仅新增。smoke_test 验证 subset 与 round2 原模块逐位一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats

# decoupler ulm epsilon — copied verbatim (do not "fix")
EPS = 2.2e-16

__all__ = ["run_ulm", "run_ulm_full_axis", "EPS"]


# ===========================================================================
# subset 语义（round2/py_ulm 逐字）
# ===========================================================================
def _prepare_net(net, gene_index, min_target_num):
    """Group (tf,target,weight) rows; intersect targets with genes; keep TFs
    with >= min_target_num matched targets. Duplicate (tf,target) pairs are
    averaged. Entries with weight == 0 are dropped."""
    if not isinstance(net, pd.DataFrame):
        raise TypeError("net must be a pandas DataFrame with tf/target/weight columns")
    net = net.copy()
    net.columns = [str(c).lower() for c in net.columns]
    for c in ("tf", "target", "weight"):
        if c not in net.columns:
            raise ValueError(f"net is missing column {c!r}")
    net = net[net["weight"] != 0]
    net = net.groupby(["tf", "target"], as_index=False)["weight"].mean()

    genes = pd.Series(np.arange(len(gene_index)), index=np.asarray(gene_index))
    m = net["target"].isin(genes.index)
    net = net[m]
    net["row"] = genes.loc[net["target"]].to_numpy()

    kept = net.groupby("tf")["row"].transform("size") >= min_target_num
    net = net[kept]
    tfs = np.sort(net["tf"].unique())
    tf_codes = pd.Series(np.arange(len(tfs)), index=tfs)
    cols = tf_codes.loc[net["tf"]].to_numpy()
    return tfs, net["row"].to_numpy(), cols.astype(np.int64), net["weight"].to_numpy(float)


def _run_ulm_subset(X, tfs, rows, cols, vals, exp_columns, cell_chunk, eps):
    """subset 语义主路径（round2 run_ulm 计算核逐字）。X: genes × cells。"""
    n_cells = X.shape[1]
    n_tf = len(tfs)
    n_t = np.bincount(cols, minlength=n_tf).astype(np.float64)          # targets per TF
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(X.shape[0], n_tf))
    W0 = sparse.csr_matrix((np.ones_like(vals), (rows, cols)), shape=(X.shape[0], n_tf))
    sw = np.asarray(W.sum(axis=0)).ravel()                              # sum w
    sww = np.asarray(W.multiply(W).sum(axis=0)).ravel()                 # sum w^2

    df = n_t - 2.0
    if np.any(df <= 0):
        raise ValueError("a kept TF has < 3 targets; t-statistic undefined (min_target_num too small)")

    act_blocks, p_blocks = [], []
    for s in range(0, n_cells, cell_chunk):
        Ec = X[:, s:s + cell_chunk]                                     # genes x cb
        swe = (W.T @ Ec).astype(np.float64)                             # TF x cb, sum w*e
        se = np.asarray(W0.T @ Ec, dtype=np.float64)                    # sum e
        see = np.asarray(W0.T @ (Ec * Ec), dtype=np.float64)            # sum e^2

        num = n_t[:, None] * swe - sw[:, None] * se
        den = np.sqrt(
            np.maximum(n_t * sww - sw ** 2, 0.0)[:, None] *
            np.maximum(n_t[:, None] * see - se ** 2, 0.0)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
        r = np.clip(r, -1.0, 1.0)

        t = r * np.sqrt(df[:, None] / ((1.0 - r + eps) * (1.0 + r + eps)))
        p = 2.0 * stats.t.sf(np.abs(t), df[:, None])
        act_blocks.append(t)
        p_blocks.append(p)

    activity = pd.DataFrame(np.hstack(act_blocks), index=tfs, columns=exp_columns)
    pval = pd.DataFrame(np.hstack(p_blocks), index=tfs, columns=exp_columns)
    return activity, pval


# ===========================================================================
# full_axis 语义（decoupler v2.2.0 mt/_ulm.py + pp/net.py 逐字公式）
# ===========================================================================
def _cov_decoupler(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """decoupler v2.2.0 `_cov` 逐字：b.grand-mean 居中（b.mean() 是全矩阵
    标量均值——代数上等价于逐列居中的 Pearson 分子，浮点上保留原样）。"""
    return np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0] - 1)


def _cor_decoupler(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """decoupler v2.2.0 `_cor` 逐字。"""
    cov = _cov_decoupler(A, b)
    ssd = np.std(A, axis=0, ddof=1) * np.std(b, axis=0, ddof=1).reshape(-1, 1)
    return cov / ssd


def _tval_decoupler(r: np.ndarray, df: float) -> np.ndarray:
    """decoupler v2.2.0 `_tval` 逐字（epsilon 2.2e-16 照抄）。"""
    return r * np.sqrt(df / ((1.0 - r + 2.2e-16) * (1.0 + r + 2.2e-16)))


def run_ulm_full_axis(exp, net, min_target_num: int = 5, eps: float = EPS):
    """full_axis 语义：全基因轴零填充权重 → decoupler 2.2.0 逐字公式，
    df = n_genes − 2；全轴常数权重（sd=0 → 0/0）自然 NaN。

    默认不分块（decoupler 2.2.0 对整块 mat 一次计算；分块会改变 grand-mean
    居中的浮点路径）。内存 O(n_genes × n_cells + n_genes × n_tf + n_cells × n_tf)。
    """
    if not isinstance(exp, pd.DataFrame):
        raise TypeError("exp must be a genes x cells pandas DataFrame")
    X = np.ascontiguousarray(exp.to_numpy(dtype=np.float64))
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains NaN/Inf")

    tfs, rows, cols, vals = _prepare_net(net, exp.index, min_target_num)
    n_tf = len(tfs)
    if n_tf == 0:
        raise ValueError(f"no TF has >= {min_target_num} targets in the gene intersection")
    if X.shape[0] < 3:
        raise ValueError("full_axis semantics needs n_genes >= 3 (df = n_genes - 2)")

    # pp/net.py `_order`：madjmat = zeros((len(features), n_src))，命中行填权重
    adj = np.zeros((X.shape[0], n_tf), dtype=np.float64)
    adj[rows, cols] = vals

    n_var, _ = adj.shape
    df = n_var - 2

    b = X  # decoupler 传 mat.T 使 b 形如 (n_var × n_obs) —— 这里 X 已是 genes × cells
    with np.errstate(invalid="ignore", divide="ignore"):
        r = _cor_decoupler(adj, b)             # (n_obs × n_src)
        t = _tval_decoupler(r, df)
        pv = stats.t.sf(np.abs(t), df) * 2     # decoupler 写法：sf(...)*2

    activity = pd.DataFrame(np.asarray(t).T, index=tfs, columns=exp.columns)
    pval = pd.DataFrame(np.asarray(pv).T, index=tfs, columns=exp.columns)
    return activity, pval


# ===========================================================================
# 统一入口
# ===========================================================================
def run_ulm(exp, net, min_target_num: int = 5, cell_chunk: int = 256,
            eps: float = EPS, semantics: str = "subset"):
    """Run ULM under either semantics.

    Parameters
    ----------
    exp : genes × cells DataFrame.
    net : DataFrame with columns tf/target/weight.
    min_target_num : minimum matched unique targets to keep a TF (metaTF: 5).
    cell_chunk : (subset only) cells per chunk.
    semantics : "subset"（round2/v1.9.2 公式，df = n_targets − 2）或
                "full_axis"（decoupler 2.2.0 公式，df = n_genes − 2，
                全轴零填充，NaN 不守卫）。

    Returns
    -------
    (activity, pval) : DataFrames TF × cells；activity = t statistic。
    """
    if semantics not in ("subset", "full_axis"):
        raise ValueError('semantics must be "subset" or "full_axis"')
    if semantics == "full_axis":
        return run_ulm_full_axis(exp, net, min_target_num=min_target_num, eps=eps)
    if not isinstance(exp, pd.DataFrame):
        raise TypeError("exp must be a genes x cells pandas DataFrame")
    X = np.ascontiguousarray(exp.to_numpy(dtype=np.float64))
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains NaN/Inf")
    tfs, rows, cols, vals = _prepare_net(net, exp.index, min_target_num)
    if len(tfs) == 0:
        raise ValueError(f"no TF has >= {min_target_num} targets in the gene intersection")
    return _run_ulm_subset(X, tfs, rows, cols, vals, exp.columns, cell_chunk, eps)
