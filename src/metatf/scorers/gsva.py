"""
py_gsva — GSVA::gsva(method="gsva", kcdf="Poisson") 的 numpy 移植（round4）。

语义依据（全部可审计）：
  (a) R 层编排：GSVA 1.50.5 Bioconductor 3.18 源码 R/gsva.R
      （candidates/round4/py_gsva/upstream_c/_extracted/GSVA/R/gsva.R）：
        - 旧 API gsva(expr, gset.idx.list, ..., kcdf, min.sz, max.sz, ...)：
          .filterFeatures（常数表达基因剔除）→ .mapGeneSetsToFeatures
          （match 到 rownames，NA 丢）→ filterGeneSets(min.sz=max(1,min.sz),
          max.sz)（按映射后大小）→ kcdf="Poisson" ⇒ rnaseq=TRUE, kernel=TRUE →
          .gsva → compute.geneset.es(sample.idxs=1:n.samples)。
        - compute.geneset.es：compute.gene.density（C 层 Poisson 核 ECDF，
          逐基因逐样本）→ sort.sgn.idxs = apply(gene.density, 2, order,
          decreasing=TRUE) → rank.scores：rank r 位基因得
          |n − r + 1 − n/2| → 逐集合 .Call("ks_matrix_R")。
        - ks_matrix_R（C 层 ks_sample）：walk 全基因（rank 1 在前）；
          hit: cum += score^τ / Σ_gset score^τ；miss: cum −= 1/(n−|Γ|)；
          mx_pos = max(0, max cum)；mx_neg = min(0, min cum)；
          mx.diff=TRUE（默认）: ES = mx_pos + mx_neg；
          abs.ranking=TRUE: ES = mx_pos − mx_neg（Kuiper 型）；
          mx.diff=FALSE: ES = (mx_pos > |mx_neg|) ? mx_pos : mx_neg。
  (b) C 层核公式：src/kernel_estimation.c（GSVA 1.50.5）逐行核对——
      Poisson（rnaseq=TRUE）：bw = 0.5；
      left_tail = mean_s ppois(y_test, x_dens + 0.5)；
      r = −log((1 − left_tail) / left_tail)。
      Rmath ppois(q, λ) = P(Pois(λ) ≤ q)，C 内 floor(q + 1e-7) 后取
      整数点 —— 这里用 scipy.special.gammaincc(floor(q+1e-7)+1, λ)
      等价实现（整数恒等式 P(Pois(λ)≤k) = Q(k+1, λ)，Q 为正则上不完全
      gamma；q<0 → 0）。
  (c) 论文交叉验证：Hänzelmann/Castelo/Guinney 2013 (BMC Bioinformatics
      14:7) Eq.(2) Poisson 核 F̂_r(x_ij) = (1/n)Σ_k P(Pois(x_ik+0.5) ≤ x_ij)，
      r=0.5 —— 与 C 源一致；Eq.(3) 随机游走权重 |r_ij|^τ（r_ij=秩分
      |p/2−rank|，最高 z 基因得 p/2 —— 与 C 的 compute_rank_score 一致）；
      Eq.(4)/(5) 的 max-deviation / difference 选项对应 mx.diff（ES+ −ES−
      中 ES− 取幅值时与 C 的 mx_pos+mx_neg 相等，见 MANIFEST）。

Gaussian/none 核与 ssgsea/zscore/plage 方法不在本候选范围（metaTF 只调
method="gsva", kcdf="Poisson", min.sz=5, max.sz=500 —— 见 ISSUE-FACTS/
code-anatomy.md §5(d)）。

向量化：逐基因 Poisson 核（基因块 × 全样本成对）；KS 行走按样本循环、
集合维全部向量化（无逐集合 python 循环）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaincc

__all__ = ["run_gsva", "gsva_poisson"]

# Rmath ppois 的 floor 容差（R src/main/qbinom.c/ppois 路径逐字）
_PPOIS_FLOOR_EPS = 1e-7


def _ppois_lower(q: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Rmath ppois(q, λ, lower.tail=TRUE, log.p=FALSE) 的等价实现。

    P(Pois(λ) ≤ q) = Q(k+1, λ)（正则上不完全 gamma，k = floor(q+1e-7)；
    整数恒等式 P(Pois(λ)≤k) = 1 − P(k+1, λ) = Γ_upper_reg(k+1, λ)），
    用 scipy.special.gammaincc(k+1, λ)。q < 0 → 0；λ == 0 → 1（R 行为；
    本路径 λ = x + 0.5 ≥ 0.5 恒 > 0）。
    """
    q = np.asarray(q, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    kq = np.floor(q + _PPOIS_FLOOR_EPS) + 1.0
    out = np.broadcast_to(gammaincc(kq, lam), q.shape if q.shape else ())
    out = np.where(q < 0.0, 0.0, out)
    return out


def _filter_features(expr: np.ndarray, gene_names: Sequence[str]):
    """.filterFeatures（GSVA 1.50.5 R/utils.R）：per-gene 样本 sd（ddof=1），
    sd < 1e-10 → 0；method!="ssgsea" 时 sd==0 或 NA 的基因剔除。"""
    with np.errstate(invalid="ignore"):
        sd = np.std(expr, axis=1, ddof=1)
    sd = np.where(sd < 1e-10, 0.0, sd)
    keep = (sd > 0) & ~np.isnan(sd)
    return expr[keep, :], [g for g, k in zip(gene_names, keep) if k], int((~keep).sum())


def _map_gene_sets(gene_sets: Dict[str, Sequence[str]],
                   features: Sequence[str]) -> Dict[str, np.ndarray]:
    """.mapGeneSetsToFeatures + filterGeneSets：match 到 features（行名）的
    1-based 索引；未匹配丢弃；按映射后大小过滤 min.sz..max.sz。"""
    fpos: Dict[str, int] = {}
    for i, g in enumerate(features):
        fpos.setdefault(g, i)  # R match()：首个匹配胜出
    out: Dict[str, np.ndarray] = {}
    for name, genes in gene_sets.items():
        idx = [fpos[g] for g in genes if g in fpos]  # match()：首个匹配
        if idx:
            out[name] = np.asarray(idx, dtype=np.int64)
    return out


def _filter_gene_sets(mapped: Dict[str, np.ndarray], min_sz: int,
                      max_sz: float) -> Dict[str, np.ndarray]:
    lo = max(1, min_sz)
    return {k: v for k, v in mapped.items() if lo <= len(v) <= max_sz}


def gsva_poisson(expr: np.ndarray, gene_chunk: int = 512) -> np.ndarray:
    """compute.gene.density（kernel=TRUE, rnaseq=TRUE 路径）。

    expr: genes × samples。返回 z（genes × samples）：
      z[g,t] = −log((1 − F_gt) / F_gt)，F_gt = (1/S) Σ_s ppois(x_gt, x_gs + 0.5)。
    F_gt == 1（浮点上）时 z = +inf —— 与 C 源同样产生（不截断，保持与 R 一致）。
    """
    n_genes, n_samples = expr.shape
    z = np.empty((n_genes, n_samples), dtype=np.float64)
    for g0 in range(0, n_genes, gene_chunk):
        g1 = min(g0 + gene_chunk, n_genes)
        block = expr[g0:g1, :]                     # (b, S)
        lam = block + 0.5                          # λ = x_gs + 0.5
        kq = np.floor(block + _PPOIS_FLOOR_EPS) + 1.0  # floor(x_gt + 1e-7) + 1（形状参数）
        # F[t, s] = Q(kq[t]+? , λ[s]) —— P(Pois(λ_s) ≤ x_t) = gammaincc(kq_t, λ_s)
        # → (S_t, S_s) 分块避免超内存
        for j in range(g0, g1):
            F = gammaincc(kq[j - g0, :][:, None], lam[j - g0, :][None, :])  # (S_t, S_s)
            meanF = F.mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                z[j, :] = -np.log((1.0 - meanF) / meanF)
    return z


def _rank_scores(z: np.ndarray) -> np.ndarray:
    """compute.geneset.es 的秩分步：每样本按 z 降序 order，rank r 位得
    |n − r + 1 − n/2|（compute_rank_score 逐字）。返回 w（genes × samples）。
    并列：numpy 稳定 argsort（R order 对 double 用 shell 排序，不稳定——
    差异记录于 MANIFEST uncertainties）。"""
    n_genes, n_samples = z.shape
    pos = np.abs(np.arange(n_genes, 0, -1, dtype=np.float64) - n_genes / 2.0)
    order = np.argsort(-z, axis=0, kind="stable")   # 列独立降序；rank 1 在前
    w = np.empty_like(z)
    rows = np.arange(n_samples)[None, :]
    w[order, rows] = pos[:, None]                   # rank r 位 ← pos[r-1]
    return w, order


def _ks_walk(w: np.ndarray, order: np.ndarray, memberships: np.ndarray,
             set_sizes: np.ndarray, tau: float = 1.0, mx_diff: bool = True,
             abs_rnk: bool = False, set_chunk: int = 512) -> np.ndarray:
    """ks_matrix_R 的集合维向量化：逐样本循环（R/C 逐样本），集合维全向量化。

    memberships: (n_sets × n_genes) bool。返回 ES (n_sets × n_samples)。
    increments[s, r] = hit ? w_sorted[r]^τ / Σ_gset w^τ : −1/(n − |Γ_s|)；
    ES = max(0, max cum) + min(0, min cum)   (mx.diff=TRUE, abs.rnk=FALSE)
       = max(0, max cum) − min(0, min cum)   (abs.rnk=TRUE)
       = (mx_pos > |mx_neg|) ? mx_pos : mx_neg  (mx.diff=FALSE)
    """
    n_genes, n_samples = w.shape
    n_sets = memberships.shape[0]
    w_pos = np.abs(np.arange(n_genes, 0, -1, dtype=np.float64) - n_genes / 2.0)
    w_sorted = np.power(np.abs(w_pos), tau)         # rank 位置权重 ^ τ
    ES = np.empty((n_sets, n_samples), dtype=np.float64)
    dec = 1.0 / (n_genes - set_sizes)               # (n_sets,)
    for t in range(n_samples):
        wt = w[:, t]
        hits = memberships[:, order[:, t]]          # (n_sets × n_genes) 按 rank 排
        with np.errstate(invalid="ignore", divide="ignore"):
            sumw = memberships @ wt                 # Σ_gset w_g^τ（τ=1 时 w 非负）
            if tau != 1.0:
                sumw = memberships @ np.power(np.abs(wt), tau)
            inc = np.where(hits,
                           w_sorted[None, :] / sumw[:, None],
                           -dec[:, None])           # (n_sets × n_genes)
            cum = np.cumsum(inc, axis=1)            # 顺序累加 == C 逐基因累加
            mx_pos = np.maximum(cum.max(axis=1), 0.0)
            mx_neg = np.minimum(cum.min(axis=1), 0.0)
            if mx_diff and not abs_rnk:
                es = mx_pos + mx_neg
            elif mx_diff and abs_rnk:
                es = mx_pos - mx_neg
            else:
                es = np.where(mx_pos > np.abs(mx_neg), mx_pos, mx_neg)
        ES[:, t] = es
    return ES


def run_gsva(expr: pd.DataFrame, gene_sets: Dict[str, Sequence[str]],
             min_sz: int = 5, max_sz: float = 500, tau: float = 1.0,
             mx_diff: bool = True, abs_ranking: bool = False,
             verbose: bool = False) -> pd.DataFrame:
    """GSVA::gsva(expr, gset.idx.list, method="gsva", kcdf="Poisson",
    min.sz=min_sz, max.sz=max_sz, mx.diff=mx_diff, tau=tau,
    abs.ranking=abs_ranking) 的 numpy 移植。

    Parameters
    ----------
    expr : genes × cells 表达矩阵（计数尺度；kcdf="Poisson" 对应 rnaseq 路径）。
    gene_sets : dict 集合名 → 基因名列表。
    Returns
    -------
    DataFrame sets × cells（= R 端 gsva() 返回矩阵）。
    """
    if not isinstance(expr, pd.DataFrame):
        raise TypeError("expr must be a genes x cells pandas DataFrame")
    gene_names = [str(g) for g in expr.index]
    X = np.ascontiguousarray(expr.to_numpy(dtype=np.float64))
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains NaN/Inf")

    if len(gene_sets) == 0:
        raise ValueError("The gene set list is empty! Filter may be too stringent.")

    # .filterFeatures：常数表达基因剔除（R 端 warning 于此发出）
    Xf, kept_names, n_const = _filter_features(X, gene_names)
    if n_const and verbose:
        print(f"GSVA: {n_const} genes with constant expression values "
              f"throughout the samples are discarded.")
    if Xf.shape[0] < 2:
        raise ValueError("Less than two genes in the input assay object")

    # 映射 + 大小过滤
    mapped = _map_gene_sets(gene_sets, kept_names)
    if len(mapped) == 0:
        raise ValueError("No identifiers in the gene sets could be matched to the "
                         "identifiers in the expression data.")
    mapped = _filter_gene_sets(mapped, min_sz, max_sz)
    if len(mapped) == 0:
        raise ValueError("No gene set meets the minimum and maximum size filter")
    if verbose:
        print(f"Estimating GSVA scores for {len(mapped)} gene sets.")

    # Poisson 核 ECDF → 秩分
    z = gsva_poisson(Xf)
    w, order = _rank_scores(z)

    set_names = list(mapped.keys())
    n_genes = Xf.shape[0]
    memberships = np.zeros((len(set_names), n_genes), dtype=bool)
    for i, k in enumerate(set_names):
        memberships[i, mapped[k]] = True
    set_sizes = memberships.sum(axis=1).astype(np.float64)

    ES = _ks_walk(w, order, memberships, set_sizes, tau=tau,
                  mx_diff=mx_diff, abs_rnk=abs_ranking)

    return pd.DataFrame(ES, index=set_names, columns=expr.columns)
