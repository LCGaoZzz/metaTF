"""
py_sincerities_fast — round3/py_sincerities 的 numba 化（round4）。

算法与调用顺序与 candidates/round3/py_sincerities/sincerities.py 完全一致——
只是编译，不做语义修改。逐位一致由 selfcheck/ 审计（同一输入下两版输出
np.array_equal 全等）。

三处加速（均为纯编译/纯调度，不改任何浮点运算序列）：
  1) KS 距离：scipy.stats.ks_2samp 不可 njit（scipy 分发对象）→ 手写
     两样本双侧 D 统计量内循环，逐语句复刻 scipy 1.16 `_stats_py.ks_2samp`
     的 statistic 路径（sort→searchsorted(right)/n1→cdf1−cdf2→
     max(clip(−min,0,1), max)）；D 只含比较/IEEE 除法/加减，无求和次序
     问题 → 与 scipy 逐位一致（selfcheck 300 组随机输入含重并列验证）。
  2) 坐标下降：沿用 round3 同一 `_cd_path_core`（逐字复制），njit(cache=True)。
  3) SINCERITIES 主版本 LOOCV：round3 逐 (gene,alpha) 串行 python 编排
     （每折 _fit_path = numpy 标准化 + njit CD + numpy 反变换/预测）。
     本版把 numpy 预处理/收尾原样保留在串行段，把 (gene,alpha,fold) 的
     CD 求解摊平成任务数组，njit prange 并行执行；每个任务内部的
     浮点序列与 round3 完全相同（同一 njit `_cd_path`、同一输入数组、
     逐折 numpy 标准化结果缓存复用——确定性 numpy 计算，位相同），
     故输出逐位一致；并行只改变任务完成顺序，基因间无数据流、无 RNG。

distance=2/4、SINCERITIES_PLUS、pcor、cmtest2、final_ranked_predictions、
run_sincerities 编排：与 round3 相同（PLUS 的距离矩阵同样走 numba KS 行核）。
"""

from __future__ import annotations

import sys
from math import gcd as _gcd
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from numba import njit as _njit, prange as _prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
    _njit = None
    _prange = None

__all__ = [
    "run_sincerities",
    "SINCERITIES",
    "SINCERITIES_PLUS",
    "cmtest2",
    "final_ranked_predictions",
    "pcor_spearman_estimate",
    "ks_distance",
    "HAVE_NUMBA",
]


# ===========================================================================
# 1) 非负弹性网坐标下降核心（与 round3 逐字相同；只是 njit(cache=True)）
# ===========================================================================
def _cd_path_core(Z, yc, lam, alpha, beta_init, max_sweeps, tol):
    """对标准化后的 Z（n×p）、中心化 yc，沿降序 lambda 网格求解，warm start。

    返回 B (p×L)。Z 必须 C 连续 float64。
    """
    n = Z.shape[0]
    p = Z.shape[1]
    L = lam.shape[0]
    B = np.empty((p, L))
    beta = beta_init.copy()
    r = np.empty(n)
    for i in range(n):
        acc = 0.0
        for j in range(p):
            acc += Z[i, j] * beta[j]
        r[i] = yc[i] - acc
    g = np.empty(p)  # (1/n) z_j' z_j
    for j in range(p):
        acc = 0.0
        for i in range(n):
            acc += Z[i, j] * Z[i, j]
        g[j] = acc / n
    for l in range(L):
        l1 = lam[l] * alpha
        l2 = lam[l] * (1.0 - alpha)
        for _sweep in range(max_sweeps):
            maxdel = 0.0
            for j in range(p):
                old = beta[j]
                zjr = 0.0
                for i in range(n):
                    zjr += Z[i, j] * (r[i] + Z[i, j] * old)
                num = zjr / n - l1
                den = g[j] + l2
                new = num / den
                if new < 0.0:
                    new = 0.0
                if new != old:
                    delta = old - new
                    for i in range(n):
                        r[i] += Z[i, j] * delta
                    d = new - old
                    if d < 0.0:
                        d = -d
                    if d > maxdel:
                        maxdel = d
                    beta[j] = new
            if maxdel < tol:
                break
        for j in range(p):
            B[j, l] = beta[j]
    return B


if HAVE_NUMBA:
    _cd_path = _njit(cache=True, fastmath=False)(_cd_path_core)
else:  # pragma: no cover
    _cd_path = _cd_path_core


def _standardize_train(X: np.ndarray):
    """glmnet 式标准化：中心化 + sd 用 1/n 分母（总体 sd）；零方差列 sd 置 1。"""
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    sd = np.sqrt((Xc ** 2).sum(axis=0) / n)
    sd = np.where(sd > 0, sd, 1.0)
    return np.ascontiguousarray(Xc / sd), mu, sd


def _lambda_path(Z: np.ndarray, yc: np.ndarray, alpha: float, nlambda: int = 100,
                 min_ratio: Optional[float] = None) -> np.ndarray:
    """glmnet 默认路径复刻：lambda_max = max|z'yc|/n / max(alpha, 0.01)；
    n < p 时 lambda.min.ratio = 0.01，否则 1e-4；几何降序 nlambda 个。"""
    n, p = Z.shape
    if min_ratio is None:
        min_ratio = 0.01 if n < p else 1e-4
    alfa = max(alpha, 0.01)
    eqf = np.abs(Z.T @ yc) / n
    lmax = float(eqf.max()) / alfa if eqf.size else 0.0
    if not np.isfinite(lmax) or lmax <= 0:
        lmax = 1e-3  # 退化 y（全常数）时的护栏，MANIFEST 记录
    return np.geomspace(lmax, lmax * min_ratio, nlambda)


def _fit_path(X: np.ndarray, y: np.ndarray, alpha: float, lambdas: np.ndarray,
              tol: float = 1e-10, max_sweeps: int = 10000):
    """完整路径拟合；返回 (Braw (p×L, 原始尺度), intercept (L,))。"""
    Z, mu, sd = _standardize_train(X)
    yc = y - y.mean()
    B = _cd_path(Z, yc, np.ascontiguousarray(lambdas, dtype=np.float64), float(alpha),
                 np.zeros(Z.shape[1]), int(max_sweeps), float(tol))
    Braw = B / sd[:, None]
    a = y.mean() - mu @ Braw
    return Braw, a


def _cv_glmnet_nn_loocv(X: np.ndarray, y: np.ndarray, alpha: float,
                        exclude: Optional[int] = None, nlambda: int = 100,
                        tol: float = 1e-10):
    """round3 的串行 LOOCV（保留为参考实现 / 无 numba 回退路径）。

    批量并行路径见 _loocv_prepare / _loocv_batch / _loocv_finalize——
    三段合起来与该函数逐位等价（selfcheck 验证）。
    """
    if exclude is not None:
        keep = [j for j in range(X.shape[1]) if j != exclude]
        X_in = np.ascontiguousarray(X[:, keep])
    else:
        X_in = X
    Z, _, _ = _standardize_train(X_in)
    yc = y - y.mean()
    lambdas = _lambda_path(Z, yc, alpha, nlambda)
    Braw_full, _ = _fit_path(X_in, y, alpha, lambdas, tol=tol)

    n = X_in.shape[0]
    L = lambdas.shape[0]
    sse = np.zeros(L)
    for k in range(n):  # foldid = 1:n → LOOCV
        tr = np.arange(n) != k
        Bk, ak = _fit_path(X_in[tr], y[tr], alpha, lambdas, tol=tol)
        pred = X_in[~tr] @ Bk + ak  # (1, L)
        sse += (y[~tr] - pred.ravel()) ** 2
    cvm = sse / n
    imin = int(np.argmin(cvm))  # R which.min：第一个最小
    return float(lambdas[imin]), float(cvm[imin]), Braw_full[:, imin], lambdas, cvm


# ===========================================================================
# 2) 两样本 KS D 统计量：scipy statistic 路径的手写复刻（可 njit）
# ===========================================================================
def _ks2_d_core(p1: np.ndarray, p2: np.ndarray) -> float:
    """逐语句复刻 scipy.stats.ks_2samp(...).statistic 的计算路径：
    data1=sort(p1); data2=sort(p2); data_all=concat;
    cdf1=searchsorted(data1,data_all,'right')/n1; cdf2=.../n2;
    cddiffs=cdf1-cdf2; d=max(clip(-min(cddiffs),0,1), max(cddiffs))。
    只含排序/比较/IEEE 除法/加减 → 与 scipy 逐位一致。
    """
    data1 = np.sort(p1)
    data2 = np.sort(p2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m = n1 + n2
    data_all = np.empty(m)
    for i in range(n1):
        data_all[i] = data1[i]
    for i in range(n2):
        data_all[n1 + i] = data2[i]
    maxS = -np.inf
    minD = np.inf
    for i in range(m):
        v = data_all[i]
        lo, hi = 0, n1
        while lo < hi:
            mid = (lo + hi) // 2
            if data1[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        c1 = lo
        lo, hi = 0, n2
        while lo < hi:
            mid = (lo + hi) // 2
            if data2[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        c2 = lo
        d = c1 / n1 - c2 / n2
        if d > maxS:
            maxS = d
        if d < minD:
            minD = d
    # scipy: minS = np.clip(-np.min(cddiffs), 0, 1)
    minS = -minD
    if minS < 0.0:
        minS = 0.0
    if minS > 1.0:
        minS = 1.0
    if maxS > minS:
        d = maxS
    else:
        d = minS
    # scipy 对 statistic 的 h/lcm 取整是无条件的（exact/asymp 只影响 p 值；
    # _attempt_exact_2kssamp 中 lcm = n1/gcd*n2；h = int(np.round(d*lcm))；
    # d = h*1.0/lcm 在返回 statistic 前生效）。round3 调 method='auto'，
    # 本内核必须逐位复刻。
    g0 = _gcd(n1, n2)
    lcm = (n1 // g0) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    return float(d)


if HAVE_NUMBA:
    _ks2_d = _njit(cache=True, fastmath=False)(_ks2_d_core)
else:  # pragma: no cover
    _ks2_d = _ks2_d_core


def ks_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """stats::ks.test(p1, p2)$statistic —— 两样本双侧 KS 的 D 统计量（numba 内核，
    与 scipy.stats.ks_2samp(...).statistic 逐位一致；无 numba 时退回 scipy）。"""
    if HAVE_NUMBA:
        return float(_ks2_d(np.ascontiguousarray(p1, dtype=np.float64),
                            np.ascontiguousarray(p2, dtype=np.float64)))
    return float(stats.ks_2samp(p1, p2, alternative="two-sided", method="auto").statistic)


# ---- 时间点聚合：一次 njit 调用算完相邻两时间点全部基因的 KS 行 ----
def _ks_matrix_rows_core(A, B, out_row, n_cells_a, n_cells_b, g):
    # A/B: cells×genes（C 连续）；out_row[g] = KS(A[:,g], B[:,g])
    a = np.empty(n_cells_a)
    b = np.empty(n_cells_b)
    for gi in range(g):
        for i in range(n_cells_a):
            a[i] = A[i, gi]
        for i in range(n_cells_b):
            b[i] = B[i, gi]
        out_row[gi] = _ks2_d(a, b)


if HAVE_NUMBA:
    _ks_matrix_rows = _njit(cache=True, fastmath=False)(_ks_matrix_rows_core)
else:  # pragma: no cover
    _ks_matrix_rows = None


# ===========================================================================
# 3) 分布距离矩阵 / dt 归一（round3 结构；distance=1 走 numba 行核）
# ===========================================================================
def _dist_stat(p1, p2, distance: int) -> float:
    if distance == 1:
        return ks_distance(p1, p2)
    if distance == 2:
        return cmtest2(p1, p2)["CM_limiting_stat"]
    if distance == 3:
        raise NotImplementedError(
            "distance=3 (kSamples::ad.test) 未移植：scipy anderson_ksamp 的 AD 统计量"
            "归一化/离散校正口径与 kSamples 不同，见 MANIFEST uncertainties"
        )
    if distance == 4:
        return abs(float(np.mean(p1) - np.mean(p2)))
    raise ValueError("distance must be 1 (KS), 2 (CM) or 4 (mean diff); 3 (AD) not ported")


def _distance_matrix(single_cell_data: List[np.ndarray], time: Sequence[float],
                     num_genes: int, distance: int = 1) -> np.ndarray:
    """DISTANCE_matrix：(T-1) × G。single_cell_data[k] = 细胞 × 基因。"""
    T = len(time)
    D = np.zeros((T - 1, num_genes))
    if distance == 1 and HAVE_NUMBA and _ks_matrix_rows is not None:
        for ti in range(T - 1):  # 时间点聚合：逐相邻对一次内核调用
            A = np.ascontiguousarray(single_cell_data[ti])
            B = np.ascontiguousarray(single_cell_data[ti + 1])
            _ks_matrix_rows(A, B, D[ti], A.shape[0], B.shape[0], num_genes)
        return D
    for ti in range(T - 1):
        a = single_cell_data[ti].T        # 基因 × 细胞
        b = single_cell_data[ti + 1].T
        for gi in range(num_genes):
            D[ti, gi] = _dist_stat(a[gi], b[gi], distance)
    return D


def _normalize_by_dt(D: np.ndarray, time: Sequence[float]) -> np.ndarray:
    deltaT = np.asarray(time[1:], dtype=np.float64)[:, None] - np.asarray(time[:-1], dtype=np.float64)[:, None]
    return D / deltaT


# ===========================================================================
# 4) ppcor::pcor(spearman) 的 estimate（round3 逐字）
# ===========================================================================
def pcor_spearman_estimate(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    ranks = stats.rankdata(mat, axis=0, method="average")
    cv = np.cov(ranks, rowvar=False)
    cv = np.atleast_2d(cv)
    eps = np.finfo(np.float64).eps
    if np.linalg.det(cv) < eps:
        U, s, Vt = np.linalg.svd(cv)
        tol = np.sqrt(eps)
        keep = s > tol * s[0]
        sinv = np.zeros_like(s)
        sinv[keep] = 1.0 / s[keep]
        icv = (U * sinv) @ Vt
    else:
        icv = np.linalg.inv(cv)
    d = np.sqrt(np.diag(icv))
    p = -icv / np.outer(d, d)
    np.fill_diagonal(p, 1.0)
    return p


# ===========================================================================
# 5) SINCERITIES 主版本（T >= 5）：三段式（串行 numpy 预处理 → prange CD 批 →
#    串行 numpy 收尾），每基因浮点序列与 round3 完全一致
# ===========================================================================
def _one_prep(X, y, alpha, exclude, nlambda, tol, cache):
    """单 (gene, alpha) 的预处理；与 round3 _cv_glmnet_nn_loocv 前半逐语句相同。

    标准化数组按 (输入列集, fold) 缓存——_standardize_train 是确定性 numpy
    计算，重复调用位相同，缓存复用不改变任何一位。
    """
    n = X.shape[0]
    if exclude is not None:
        key_cols = ("drop", int(exclude))
        keep = [j for j in range(X.shape[1]) if j != exclude]
        X_in = np.ascontiguousarray(X[:, keep])
    else:
        key_cols = ("full",)
        X_in = X
    key_full = ("std", key_cols, None)
    if key_full not in cache:
        cache[key_full] = _standardize_train(X_in)
    Z, _, _ = cache[key_full]
    yc = y - y.mean()
    lambdas = _lambda_path(Z, yc, alpha, nlambda)
    folds = []
    for k in range(n):
        key_k = ("std", key_cols, k)
        tr = np.arange(n) != k
        if key_k not in cache:
            cache[key_k] = _standardize_train(X_in[tr])
        Zk, muk, sdk = cache[key_k]
        yk = y[tr]
        folds.append((Zk, muk, sdk, yk - yk.mean(), float(yk.mean()), X_in[~tr]))
    return {"alpha": alpha, "lambdas": lambdas, "yc_full": yc,
            "Z_full": Z, "mu_full": cache[key_full][1], "sd_full": cache[key_full][2],
            "y": y, "X_in": X_in, "folds": folds}


if HAVE_NUMBA:

    def _cd_batch_core(Zbuf, ycbuf, lambdas_buf, alpha_buf, Bbuf, n_tasks):
        """prange 摊平任务批：每个任务调与 round3 逐字相同的 `_cd_path_core`。

        Zbuf: (T,n,p)；ycbuf: (T,n)；lambdas_buf: (T,L)；Bbuf: (T,p,L)。
        任务内浮点序列与 round3 单次 _fit_path 内核完全一致；并行只改变
        任务完成顺序（任务间无数据流、无共享可变状态）。
        """
        for t in _prange(n_tasks):
            B = _cd_path(Zbuf[t], ycbuf[t], lambdas_buf[t],
                         alpha_buf[t], np.zeros(Zbuf[t].shape[1]), 10000, 1e-10)
            for j in range(B.shape[0]):
                for l in range(B.shape[1]):
                    Bbuf[t, j, l] = B[j, l]

    _cd_batch = _njit(cache=True, fastmath=False, parallel=True)(_cd_batch_core)
else:  # pragma: no cover
    _cd_batch = None


def _run_flat(flat, parallel: bool):
    """执行 (gene, alpha, fit) 元任务的 CD 求解，返回与 flat 对齐的 B_std 列表。

    按形状分组（full 拟合 n 行 / LOOCV 各折 n−1 行），每组一个
    `_cd_batch` prange 批内核；组内每个任务调与 round3 逐字相同的
    `_cd_path`（浮点序列位一致）。parallel=False / 无 numba 时串行
    逐任务（同样位一致）。
    """
    n_tasks = len(flat)
    out: List[Optional[np.ndarray]] = [None] * n_tasks
    groups: Dict[tuple, List[int]] = {}
    for t, (Z, yc, lam, alpha) in enumerate(flat):
        groups.setdefault((Z.shape, lam.shape[0]), []).append(t)
    for key, ts in groups.items():
        Zs, L = key
        if HAVE_NUMBA and _cd_batch is not None and parallel:
            nt = len(ts)
            Zbuf = np.empty((nt,) + Zs, dtype=np.float64)
            ycbuf = np.empty((nt, Zs[0]), dtype=np.float64)
            lambdas_buf = np.empty((nt, L), dtype=np.float64)
            alpha_buf = np.empty(nt, dtype=np.float64)
            Bbuf = np.empty((nt, Zs[1], L), dtype=np.float64)
            for u, t in enumerate(ts):
                Z, yc, lam, alpha = flat[t]
                Zbuf[u] = Z
                ycbuf[u] = yc
                lambdas_buf[u] = lam
                alpha_buf[u] = alpha
            _cd_batch(Zbuf, ycbuf, lambdas_buf, alpha_buf, Bbuf, nt)
            for u, t in enumerate(ts):
                out[t] = np.array(Bbuf[u])
            continue
        for t in ts:
            Z, yc, lam, alpha = flat[t]
            out[t] = _cd_path(np.ascontiguousarray(Z), np.ascontiguousarray(yc),
                              np.ascontiguousarray(lam, dtype=np.float64),
                              float(alpha), np.zeros(Z.shape[1]), 10000, 1e-10)
    return out


def SINCERITIES(DATA: Dict, distance: int = 1, method: int = 1, noDIAG: int = 0,
                SIGN: int = 1, nlambda: int = 100, parallel: bool = True,
                gene_block: Optional[int] = None):
    if distance not in (1, 2, 3, 4):
        raise ValueError("distance must be 1/2/3/4")
    if method not in (1, 2, 3, 4):
        raise ValueError("method must be 1/2/3/4")
    if method == 4:
        raise NotImplementedError("method=4 需要交互式 readline 输入 alpha，headless 不支持")
    if noDIAG not in (0, 1):
        raise ValueError("noDIAG must be 0 or 1")
    if SIGN not in (0, 1):
        raise ValueError("SIGN must be 0 or 1")

    single_cell_data = DATA["singleCELLdata"]
    time = DATA["time"]
    num_genes = DATA["numGENES"]
    gene_names = DATA["genes"]
    num_time_points = len(time)
    if num_time_points < 5:
        raise ValueError("DATA with a number of time points < 5. Use SINCERITIES_PLUS.")

    # Distribution Distance
    D = _distance_matrix(single_cell_data, time, num_genes, distance)
    D = _normalize_by_dt(D, time)

    # alphas：1→RIDGE；2→ELASTIC-NET 自动 α 网格；3→LASSO
    if method == 1:
        alphas = np.array([0.0])
    elif method == 2:
        alphas = np.linspace(0.0, 1.0, 11)  # R seq(0,1,0.1)：11 个值含端点
    else:
        alphas = np.array([1.0])

    X_matrix = D[0: num_time_points - 2, :]
    Y_all = D[1: num_time_points - 1, :]

    pred = np.zeros((num_genes, num_genes))  # 行=regulator，列=target
    lambda_res = np.zeros(num_genes)
    alpha_res = np.zeros(num_genes)
    n_alpha = len(alphas)
    tol = 1e-10

    # ---- 分基因块：串行 numpy 预处理 → (gene,alpha,fit) 摊平 prange CD →
    #      串行 numpy 收尾（收尾与 round3 _cv_glmnet_nn_loocv 后半逐语句相同）。
    #      块越大 prange 负载越均衡；Bbuf 内存 ≈ 5·g·p·L·8B，默认上限 4e8B。
    if gene_block is None:
        n_fits_per_gene = len(alphas) * (1 + num_time_points - 2)
        per_gene_bytes = n_fits_per_gene * X_matrix.shape[1] * nlambda * 8
        gene_block = int(max(64, min(num_genes, 4e8 // max(per_gene_bytes, 1))))
    for g0 in range(0, num_genes, gene_block):
        g1 = min(g0 + gene_block, num_genes)
        cache: Dict = {}
        prepped = []           # (gi, ai, meta)，块内顺序与 round3 基因循环一致
        flat_meta = []         # (meta, kind)  kind="full" 或折号
        for gi in range(g0, g1):
            y = Y_all[:, gi]
            exclude = gi if noDIAG == 1 else None  # round3：目标基因自身回归剔除对角
            for ai in range(n_alpha):
                meta = _one_prep(X_matrix, y, float(alphas[ai]), exclude,
                                 nlambda, tol, cache)
                prepped.append((gi, ai, meta))
                flat_meta.append((meta, "full"))
                for k in range(len(meta["folds"])):
                    flat_meta.append((meta, k))
        flat = []
        for meta, kind in flat_meta:
            if kind == "full":
                flat.append((meta["Z_full"], meta["yc_full"], meta["lambdas"],
                             meta["alpha"]))
            else:
                Zk, _, _, yck, _, _ = meta["folds"][kind]
                flat.append((Zk, yck, meta["lambdas"], meta["alpha"]))
        Bres = _run_flat(flat, parallel)

        # 收尾 + alpha 选择（与 round3 逐语句相同）
        idx = 0
        sel_by_gene: Dict[int, Dict] = {}
        for gi, ai, meta in prepped:
            B_full = Bres[idx]; idx += 1
            n = meta["X_in"].shape[0]
            L = meta["lambdas"].shape[0]
            sse = np.zeros(L)
            for k in range(n):
                Bk_std = Bres[idx]; idx += 1
                Zk, muk, sdk, yck, yk_mean, x_held = meta["folds"][k]
                Bk = Bk_std / sdk[:, None]
                ak = yk_mean - muk @ Bk
                pred_k = x_held @ Bk + ak  # (1, L)
                sse += (meta["y"][np.arange(n) == k] - pred_k.ravel()) ** 2
            cvm = sse / n
            imin = int(np.argmin(cvm))  # R which.min：第一个最小
            lam_min = float(meta["lambdas"][imin])
            cvm_min = float(cvm[imin])
            Braw_full = B_full / meta["sd_full"][:, None]
            b_raw = Braw_full[:, imin]
            if noDIAG == 1 and b_raw.shape[0] == X_matrix.shape[1] - 1:
                full = np.zeros(X_matrix.shape[1])
                keep = [j for j in range(X_matrix.shape[1]) if j != gi]
                full[keep] = b_raw
                b_raw = full
            sel = sel_by_gene.setdefault(gi, {
                "lam_seq": np.zeros(n_alpha), "cv_err": np.zeros(n_alpha),
                "beta_cols": np.zeros((X_matrix.shape[1], n_alpha))})
            sel["lam_seq"][ai] = lam_min
            sel["cv_err"][ai] = cvm_min
            sel["beta_cols"][:, ai] = b_raw
        for gi in range(g0, g1):
            sel = sel_by_gene[gi]
            cv_err = sel["cv_err"]
            min_idx = int(np.max(np.where(cv_err == cv_err.min())[0]))  # R max(which(...))
            lambda_res[gi] = sel["lam_seq"][min_idx]
            alpha_res[gi] = alphas[min_idx]
            pred[:, gi] = sel["beta_cols"][:, min_idx]

    if SIGN == 1:
        parcorr = pcor_spearman_estimate(DATA["totDATA"])
        pred = pred * np.sign(parcorr)

    return {
        "DISTANCE_matrix": D,
        "adj_matrix": pred,
        "genes": gene_names,
        "lambda": lambda_res,
        "alpha": alpha_res,
    }


# ===========================================================================
# 6) SINCERITIES_PLUS（3-4 时间点）+ SINCERITIES_final —— round3 结构不变，
#    仅距离矩阵走 numba KS 行核
# ===========================================================================
def _cvfolds_indices(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int64)
    base = n // K
    rem = n % K
    start = 0
    for k in range(K):
        size = base + (1 if k < rem else 0)
        fold_id[perm[start: start + size]] = k
        start += size
    return fold_id


def _nn_ridge_fit_fixed_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray,
                             exclude: Optional[int] = None):
    if exclude is not None:
        keep = [j for j in range(X.shape[1]) if j != exclude]
        X_in = np.ascontiguousarray(X[:, keep])
    else:
        X_in = X
    B, a = _fit_path(X_in, y, 0.0, lambdas)
    return B, a, X_in.shape[1], X.shape[1]


def _expand_beta(B: np.ndarray, p_full: int, exclude: Optional[int] = None) -> np.ndarray:
    if exclude is None:
        return B
    out = np.zeros((p_full, B.shape[1]))
    out[[j for j in range(p_full) if j != exclude], :] = B
    return out


def SINCERITIES_PLUS(DATA: Dict, noDIAG: int = 0, SIGN: int = 1, CV_nfolds: int = 5,
                     seed: int = 0):
    single_cell_data = DATA["singleCELLdata"]
    time = DATA["time"]
    num_genes = DATA["numGENES"]
    gene_names = DATA["genes"]
    num_time_points = len(time)
    if num_time_points < 3:
        raise ValueError("The data must contain at least 3 time points")

    rng = np.random.default_rng(seed)
    K = CV_nfolds
    fold_ids = [_cvfolds_indices(sc.shape[0], K, rng) for sc in single_cell_data]

    lambdas = 10 ** np.linspace(-2, 2, 100)  # R: 10^seq(-2,2,length.out=100)
    nL = len(lambdas)
    error_CV = np.zeros((K, num_genes, nL))

    for cross in range(K):
        data4train = [sc[fold_ids[t] != cross] for t, sc in enumerate(single_cell_data)]
        data4test = [sc[fold_ids[t] == cross] for t, sc in enumerate(single_cell_data)]

        D_tr = _normalize_by_dt(_distance_matrix(data4train, time, num_genes, 1), time)
        D_te = _normalize_by_dt(_distance_matrix(data4test, time, num_genes, 1), time)
        X_tr = D_tr[0: num_time_points - 2, :]
        X_te = D_te[0: num_time_points - 2, :]

        for gi in range(num_genes):
            Y_tr = D_tr[1: num_time_points - 1, gi]
            Y_te = D_te[1: num_time_points - 1, gi]
            B, a, _, p_full = _nn_ridge_fit_fixed_path(
                X_tr, Y_tr, lambdas, exclude=(gi if noDIAG == 1 else None))
            Bx = _expand_beta(B, p_full, (gi if noDIAG == 1 else None))
            pred = X_te @ Bx + a  # (T-2, L)
            error_CV[cross, gi, :] = ((Y_te[:, None] - pred) ** 2).sum(axis=0)

    mean_error = error_CV.mean(axis=0)
    se_mean = error_CV.std(axis=0, ddof=1) / np.sqrt(K)

    idx_lambda_min = np.argmin(mean_error, axis=1)
    idx_lambda_1se = np.zeros(num_genes, dtype=np.int64)
    for gi in range(num_genes):
        thr = mean_error[gi, idx_lambda_min[gi]] + se_mean[gi, idx_lambda_min[gi]]
        idx_lambda_1se[gi] = int(np.where(mean_error[gi, :] <= thr)[0][0])

    D_full = _normalize_by_dt(_distance_matrix(single_cell_data, time, num_genes, 1), time)
    X_full = D_full[0: num_time_points - 2, :]
    pred = np.zeros((num_genes, num_genes))
    for gi in range(num_genes):
        Y = D_full[1: num_time_points - 1, gi]
        B, _, _, p_full = _nn_ridge_fit_fixed_path(
            X_full, Y, lambdas, exclude=(gi if noDIAG == 1 else None))
        Bx = _expand_beta(B, p_full, (gi if noDIAG == 1 else None))
        pred[:, gi] = Bx[:, idx_lambda_min[gi]]

    if SIGN == 1:
        parcorr = pcor_spearman_estimate(DATA["totDATA"])
        pred = pred * np.sign(parcorr)

    return {
        "DISTANCE_matrix": D_full,
        "adj_matrix": pred,
        "genes": gene_names,
        "idx_lambda_min": idx_lambda_min,
        "idx_lambda_1se": idx_lambda_1se,
        "lambdas": lambdas,
    }


# ===========================================================================
# 7) final_ranked_predictions（round3 逐字）
# ===========================================================================
def final_ranked_predictions(connectivity: np.ndarray, genes: Sequence[str], SIGN: int = 1,
                             norm_interaction: bool = False) -> pd.DataFrame:
    num_genes = len(genes)
    interactions = np.asarray(connectivity, dtype=np.float64).flatten(order="F")
    if SIGN == 1:
        edges = np.where(interactions < 0, "repression",
                         np.where(interactions > 0, "activation", "no regulation"))
    else:
        edges = np.where(interactions == 0, "no regulation", "activation/repression")
    if norm_interaction:
        interactions = np.abs(interactions)

    source = np.tile(np.asarray(genes, dtype=object), num_genes)
    target = np.repeat(np.asarray(genes, dtype=object), num_genes)

    df = pd.DataFrame(
        {"SourceGENES": source, "TargetGENES": target,
         "Interaction": interactions, "Edges": edges})
    df = df.sort_values("Interaction", ascending=False, kind="mergesort").reset_index(drop=True)
    df.index = df.index + 1
    return df


# ===========================================================================
# 8) 编排层 run_sincerities（round3 逐字 + parallel 透传）
# ===========================================================================
def run_sincerities(exp_mat: pd.DataFrame, col_data: pd.DataFrame, distance: int = 1,
                    method: int = 1, noDIAG: int = 0, SIGN: int = 1, CV_nfolds: int = 5,
                    seed: int = 0, nlambda: int = 100, parallel: bool = True) -> Dict:
    """exp_mat: 基因×细胞；col_data: 两列 (sample, time)。返回
    dict(adj_matrix(归一), final_table, DISTANCE_matrix, genes)。"""
    exp_mat = exp_mat.copy()
    if sorted(map(str, exp_mat.columns)) != sorted(map(str, col_data.iloc[:, 0].astype(str))):
        raise ValueError("The samples in exp_mat is not identical in col_data!")
    if not np.issubdtype(np.asarray(col_data.iloc[:, 1]).dtype, np.number):
        raise ValueError("The second column of 'col_data' must be numeric!")

    M = exp_mat.T.astype(np.float64)              # 细胞 × 基因
    gene_names = list(exp_mat.index)
    cd = col_data.iloc[:, [0, 1]].copy()
    cd.iloc[:, 0] = cd.iloc[:, 0].astype(str)
    cd = cd.sort_values(cd.columns[1], kind="mergesort")
    time_line = cd.iloc[:, 1].to_numpy(dtype=np.float64)
    time = np.unique(time_line)                    # sort(unique())
    M = M.loc[cd.iloc[:, 0].to_list()]             # 按样本名重排

    single_cell_data = [M.to_numpy()[time_line == t, :] for t in time]

    DATA = {
        "time": time,
        "num_time_points": len(time),
        "totDATA": M.to_numpy(),
        "timeline": time_line,
        "numGENES": M.shape[1],
        "genes": gene_names,
        "singleCELLdata": single_cell_data,
    }

    if len(time) >= 5:
        res = SINCERITIES(DATA, distance=distance, method=method, noDIAG=noDIAG,
                          SIGN=SIGN, nlambda=nlambda, parallel=parallel)
    elif 2 < len(time) < 5:
        res = SINCERITIES_PLUS(DATA, noDIAG=noDIAG, SIGN=SIGN, CV_nfolds=CV_nfolds, seed=seed)
    else:
        raise ValueError("SINCERITIES method must have at least 3 time points.")

    adj = res["adj_matrix"]
    mx = adj.max()
    adj_norm = adj / mx  # R: adj_matrix/max(adj_matrix)（含符号矩阵上的最大值）
    final_table = final_ranked_predictions(adj_norm, DATA["genes"], SIGN=1)

    return {
        "adj_matrix": adj_norm,
        "final_table": final_table,
        "DISTANCE_matrix": res["DISTANCE_matrix"],
        "genes": DATA["genes"],
    }
