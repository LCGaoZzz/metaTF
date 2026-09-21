"""Plotting helpers mirroring the main visual idioms of the R metaTF package.

The upstream package uses pheatmap/scater/ggplot2 for recurring views:
scaled regulon heatmaps, reduced-dimension cell maps, expression distributions,
and Radviz projections.  Matplotlib is imported lazily so the numerical core
stays lightweight when plotting is not requested.
"""

from __future__ import annotations

from math import ceil, pi
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "plot_heatmap",
    "plot_reduced_dim",
    "plot_regulon_reduced_dim",
    "plot_expression",
    "plot_radviz",
]

# Stage colours used in the upstream HSC/EHT vignettes.
STAGE_COLORS = ("#729ece", "#ff9e4a", "#67bf5c", "#ed665d", "#ad8bc9", "#a8786e")


def _vector(values, index, name="labels"):
    if isinstance(values, pd.Series):
        if not values.index.is_unique or not index.isin(values.index).all():
            raise ValueError(f"{name} must contain one label for each cell")
        out = values.reindex(index)
    else:
        if len(values) != len(index):
            raise ValueError(f"{name} must have one value per cell")
        out = pd.Series(values, index=index)
    if out.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    return out


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotting requires matplotlib; install metatf[plot]") from exc
    return plt


def _frame(x, *, name: str = "matrix") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        out = x.copy()
    else:
        arr = np.asarray(x)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        out = pd.DataFrame(arr)
    if out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    return ax


def _scale_rows(mat: pd.DataFrame) -> pd.DataFrame:
    x = mat.astype(float)
    mean = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1).replace(0, np.nan)
    return x.sub(mean, axis=0).div(sd, axis=0).fillna(0.0)


def plot_heatmap(
    activity,
    *,
    features: Optional[Sequence[str]] = None,
    cells: Optional[Sequence[str]] = None,
    annotations: Optional[pd.DataFrame | Mapping[str, Sequence]] = None,
    scale: str = "row",
    cmap: str = "RdYlBu_r",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    ax=None,
    colorbar: bool = True,
    show: bool = False,
):
    """Draw a pheatmap-like regulon heatmap and return its ``Axes``.

    ``activity`` follows metaTF's convention (features/regulons x cells).
    ``scale='row'`` is the upstream default.  Categorical column annotations
    are rendered as compact color strips above the matrix.
    """
    plt = _mpl()
    mat = _frame(activity, name="activity")
    if features is not None:
        missing = [x for x in features if x not in mat.index]
        if missing:
            raise KeyError(f"features not found: {missing[:5]}")
        mat = mat.loc[list(features)]
    if cells is not None:
        missing = [x for x in cells if x not in mat.columns]
        if missing:
            raise KeyError(f"cells not found: {missing[:5]}")
        mat = mat.loc[:, list(cells)]
    if scale not in ("row", "none"):
        raise ValueError("scale must be 'row' or 'none'")
    values = _scale_rows(mat) if scale == "row" else mat.astype(float)
    if min(values.shape) == 0 or not np.isfinite(mat.to_numpy(dtype=float)).all():
        raise ValueError("heatmap requires a nonempty finite matrix")
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
    row_tree = col_tree = None
    if cluster_rows and len(values) > 1:
        row_tree = linkage(values.to_numpy(), method="complete", metric="euclidean")
        values = values.iloc[leaves_list(row_tree)]
    if cluster_cols and values.shape[1] > 1:
        col_tree = linkage(values.to_numpy().T, method="complete", metric="euclidean")
        values = values.iloc[:, leaves_list(col_tree)]
    if ax is None:
        if figsize is None:
            figsize = (max(5.0, min(16.0, 0.24 * values.shape[1])),
                       max(3.0, min(16.0, 0.28 * values.shape[0])))
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values.to_numpy(), aspect="auto", interpolation="nearest",
                   cmap=cmap)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(values.index, fontsize=8)
    step = max(1, ceil(values.shape[1] / 40))
    xt = np.arange(0, values.shape[1], step)
    ax.set_xticks(xt)
    ax.set_xticklabels(values.columns[xt], rotation=90, fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("Cells")
    ax.set_ylabel("Regulons")
    _style(ax)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    n_ann = 0
    if annotations is not None:
        ann = (annotations.copy() if isinstance(annotations, pd.DataFrame)
               else pd.DataFrame(annotations, index=_frame(activity).columns))
        if not ann.index.is_unique or not values.columns.isin(ann.index).all():
            raise ValueError("annotation index must cover the plotted cells")
        ann = ann.reindex(values.columns)
        if ann.isna().any().any():
            raise ValueError("annotations must not contain missing labels")
        n_ann = len(ann.columns)
        handles = []
        for j, col in enumerate(ann.columns):
            codes, levels = pd.factorize(ann[col], sort=False)
            _, colors = _categorical_colors(ann[col])
            strip_ax = ax.inset_axes([0, 1.01 + 0.045 * j, 1, 0.035])
            strip_ax.imshow(codes[None, :], aspect="auto", interpolation="nearest",
                            cmap=ListedColormap([colors[x] for x in levels]),
                            vmin=-0.5, vmax=len(levels) - 0.5)
            strip_ax.set(xticks=[], yticks=[])
            strip_ax.set_ylabel(str(col), rotation=0, ha="right", va="center", fontsize=7)
            handles.extend(Patch(color=colors[x], label=f"{col}: {x}") for x in levels)
        ax.legend(handles=handles, frameon=False, fontsize=7,
                  bbox_to_anchor=(1.18, 1), loc="upper left")
    if row_tree is not None and np.any(row_tree[:, 2] > 0):
        tree_ax = ax.inset_axes([-0.16, 0, 0.10, 1])
        dendrogram(row_tree, orientation="left", ax=tree_ax,
                   no_labels=True, link_color_func=lambda _: "#333333")
        tree_ax.invert_yaxis()
        tree_ax.axis("off")
    if col_tree is not None and np.any(col_tree[:, 2] > 0):
        tree_ax = ax.inset_axes([0, 1.03 + 0.045 * n_ann, 1, 0.15])
        dendrogram(col_tree, ax=tree_ax, no_labels=True,
                   link_color_func=lambda _: "#333333")
        tree_ax.axis("off")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                           label="Scaled activity" if scale == "row" else "Activity")
    if show:
        plt.show()
    return ax


def _categorical_colors(values, palette=None):
    plt = _mpl()
    s = pd.Series(values)
    cats = list(pd.unique(s))
    if isinstance(palette, Mapping):
        colors = {v: palette[v] for v in cats}
    elif palette is None:
        colors = {v: STAGE_COLORS[i % len(STAGE_COLORS)] for i, v in enumerate(cats)}
    else:
        cmap = plt.get_cmap(palette)
        colors = {v: cmap(i % cmap.N) for i, v in enumerate(cats)}
    return [colors[v] for v in s], colors


def plot_reduced_dim(
    embedding,
    *,
    color=None,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    title: Optional[str] = None,
    point_size: float = 10,
    alpha: float = 0.9,
    palette: Optional[str] = None,
    legend: bool = True,
    show: bool = False,
):
    """Draw a scater/ggplot-like two-dimensional cell embedding."""
    plt = _mpl()
    emb = _frame(embedding, name="embedding")
    if emb.shape[1] < 2:
        raise ValueError("embedding needs at least two columns")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    xy = emb.iloc[:, :2].to_numpy(dtype=float)
    if color is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c="#4C78A8", alpha=alpha,
                   linewidths=0)
    else:
        c = _vector(color, emb.index, "color")
        if pd.api.types.is_numeric_dtype(c):
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=c.to_numpy(float),
                            cmap=palette or "viridis", alpha=alpha, linewidths=0)
            ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=c.name or "value")
        else:
            _, color_map = _categorical_colors(c, palette)
            for key, col in color_map.items():
                mask = c.to_numpy() == key
                ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, c=[col],
                           alpha=alpha, linewidths=0, label=str(key))
            if legend:
                ax.legend(frameon=False, title=c.name, loc="best")
    if labels is not None:
        if len(labels) != len(emb):
            raise ValueError("labels must have one value per embedding row")
        for x, y, label in zip(xy[:, 0], xy[:, 1], labels):
            ax.text(x, y, str(label), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(str(emb.columns[0]))
    ax.set_ylabel(str(emb.columns[1]))
    ax.set_title(title or "")
    _style(ax)
    if show:
        plt.show()
    return ax


def plot_regulon_reduced_dim(
    activity,
    embedding=None,
    *,
    feature: Optional[str] = None,
    color=None,
    method: str = "PCA",
    point_size: float = 10,
    ax=None,
    show: bool = False,
):
    """Plot activity on an embedding or a deterministic PCA embedding.

    This is the Python counterpart of ``plotRegulonReducedDim``.  Activity is
    regulons x cells; a supplied embedding must be cells x 2+.
    """
    mat = _frame(activity, name="activity")
    if embedding is None:
        if method.upper() != "PCA":
            raise ValueError("automatic embedding is PCA; supply coordinates for UMAP/TSNE")
        if min(mat.shape) < 2:
            raise ValueError("PCA needs at least two features and two cells")
        x = mat.to_numpy(dtype=float).T
        x = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        xy = x @ vt[:2].T
        embedding = pd.DataFrame(xy, index=mat.columns,
                                 columns=[f"{method}1", f"{method}2"])
    else:
        labeled = isinstance(embedding, pd.DataFrame)
        embedding = _frame(embedding, name="embedding")
        if len(embedding) != mat.shape[1]:
            raise ValueError("embedding rows must match activity columns")
        if labeled:
            if not embedding.index.is_unique or not mat.columns.isin(embedding.index).all():
                raise ValueError("embedding index must match activity cell names")
            embedding = embedding.reindex(mat.columns)
        else:
            embedding.index = mat.columns
    if color is None and feature is not None:
        if feature not in mat.index:
            raise KeyError(feature)
        color = mat.loc[feature].to_numpy()
    return plot_reduced_dim(embedding, color=color, point_size=point_size,
                            ax=ax, title=feature, show=show)


def plot_expression(
    expression,
    groups,
    *,
    features: Optional[Sequence[str]] = None,
    kind: str = "violin",
    ncols: Optional[int] = None,
    figsize: Optional[tuple[float, float]] = None,
    point_size: float = 3,
    show: bool = False,
):
    """Draw grouped expression/activity distributions like scater plots.

    ``expression`` is genes/features x cells and ``groups`` has one label per
    cell.  ``kind`` accepts ``violin`` (the usual metaTF vignette style) or
    ``box``.  Returns ``(figure, axes)``.
    """
    plt = _mpl()
    mat = _frame(expression, name="expression")
    if features is None:
        features = list(mat.index[: min(12, len(mat.index))])
    features = list(features)
    if not features:
        raise ValueError("features must not be empty")
    if kind not in ("violin", "box"):
        raise ValueError("kind must be 'violin' or 'box'")
    group = _vector(groups, mat.columns, "groups")
    levels = list(pd.unique(group))
    ncols = ncols or max(1, int(np.ceil(np.sqrt(len(features)))))
    nrows = int(np.ceil(len(features) / ncols))
    if figsize is None:
        figsize = (3.0 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, feature in enumerate(features):
        if feature not in mat.index:
            raise KeyError(feature)
        ax = axes.flat[i]
        vals = [mat.loc[feature, group.to_numpy() == level].to_numpy(float)
                for level in levels]
        positions = np.arange(1, len(levels) + 1)
        if kind == "violin":
            for j, arr in enumerate(vals, start=1):
                color = STAGE_COLORS[(j - 1) % len(STAGE_COLORS)]
                if len(arr) > 1 and np.ptp(arr) > 0:
                    parts = ax.violinplot([arr], positions=[j], showmeans=False,
                                         showmedians=True, widths=0.85)
                    parts["bodies"][0].set_facecolor(color)
                    parts["bodies"][0].set_alpha(0.6)
                else:
                    ax.plot([j - 0.25, j + 0.25], [arr[0]] * 2, color=color)
        else:
            ax.boxplot(vals, positions=positions, widths=0.65, patch_artist=True,
                       boxprops={"facecolor": "white", "linewidth": 0.8})
        for j, arr in enumerate(vals, start=1):
            if len(arr):
                rng = np.random.default_rng(0)
                xx = j + rng.uniform(-0.08, 0.08, size=len(arr))
                ax.scatter(xx, arr, s=point_size, c=STAGE_COLORS[(j - 1) % 6], alpha=0.35,
                           linewidths=0)
        ax.set_title(str(feature), fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(x) for x in levels], rotation=45, ha="right")
        _style(ax)
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_radviz(
    data,
    *,
    anchors: Optional[Sequence[str]] = None,
    color=None,
    ax=None,
    point_size: float = 10,
    alpha: float = 0.8,
    outline_circle: bool = True,
    plot_type: str = "point",
    palette=None,
    show: bool = False,
):
    """Draw a lightweight Radviz projection with ggplot-like styling."""
    plt = _mpl()
    df = _frame(data, name="data")
    anchors = list(df.columns if anchors is None else anchors)
    if plot_type not in ("point", "hexagonal", "density", "bubble"):
        raise ValueError("plot_type must be point, hexagonal, density or bubble")
    if plot_type != "point" and color is not None:
        raise ValueError("color is supported for point plots; aggregate views show cell density")
    if len(anchors) < 3:
        raise ValueError("Radviz needs at least three anchors")
    missing = [a for a in anchors if a not in df.columns]
    if missing:
        raise KeyError(f"anchors not found: {missing[:5]}")
    z = df.loc[:, anchors].astype(float)
    lo, hi = z.min(axis=0), z.max(axis=0)
    z = z.sub(lo).div((hi - lo).replace(0, 1), axis=1).fillna(0.0)
    angles = np.linspace(0, 2 * pi, len(anchors), endpoint=False) + pi / 2
    xy_anchor = np.column_stack([np.cos(angles), np.sin(angles)])
    weight = z.to_numpy()
    den = weight.sum(axis=1, keepdims=True)
    xy = np.divide(weight @ xy_anchor, den, out=np.zeros((len(df), 2)), where=den != 0)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))
    if plot_type == "hexagonal":
        h = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=25, mincnt=1, cmap="Blues")
        ax.figure.colorbar(h, ax=ax, fraction=0.046, pad=0.04, label="Cells")
    elif plot_type == "density":
        from scipy.stats import gaussian_kde
        if len(xy) < 3 or np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
            raise ValueError("density needs at least three non-collinear projected cells")
        grid = np.linspace(-1, 1, 120)
        xx, yy = np.meshgrid(grid, grid)
        density = gaussian_kde(xy.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        density[xx * xx + yy * yy > 1] = np.nan
        h = ax.contourf(xx, yy, density, levels=12, cmap="Blues")
        ax.figure.colorbar(h, ax=ax, fraction=0.046, pad=0.04, label="Density")
    elif plot_type == "bubble":
        counts, edges_x, edges_y = np.histogram2d(xy[:, 0], xy[:, 1], bins=20,
                                                 range=[[-1, 1], [-1, 1]])
        ix, iy = np.nonzero(counts)
        h = ax.scatter((edges_x[ix] + edges_x[ix + 1]) / 2,
                       (edges_y[iy] + edges_y[iy + 1]) / 2,
                       s=100 * counts[ix, iy] / counts.max(), c=counts[ix, iy], cmap="Blues")
        ax.figure.colorbar(h, ax=ax, fraction=0.046, pad=0.04, label="Cells")
    elif color is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c="#4C78A8", alpha=alpha,
                   linewidths=0)
    else:
        c = _vector(color, df.index, "color")
        if pd.api.types.is_numeric_dtype(c):
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=c.to_numpy(float),
                            cmap="viridis", alpha=alpha, linewidths=0)
            ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            _, color_map = _categorical_colors(c, palette)
            for key, col in color_map.items():
                mask = c.to_numpy() == key
                ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, c=[col],
                           alpha=alpha, linewidths=0, label=str(key))
            ax.legend(frameon=False, loc="best")
    if outline_circle:
        t = np.linspace(0, 2 * pi, 300)
        ax.plot(np.cos(t), np.sin(t), color="#B0B0B0", linewidth=0.7)
    for (x, y), label in zip(xy_anchor, anchors):
        ax.plot([0, x], [0, y], color="#D9D9D9", linewidth=0.5, zorder=0)
        ax.text(1.08 * x, 1.08 * y, str(label), color="#8B3A3A", fontsize=8,
                ha="center", va="center")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    _style(ax)
    if show:
        plt.show()
    return ax
