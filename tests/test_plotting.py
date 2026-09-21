"""Smoke tests for the Python counterparts of the main metaTF plot types."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from metatf import (plot_expression, plot_heatmap, plot_radviz,
                    plot_reduced_dim, plot_regulon_reduced_dim)


@pytest.fixture
def activity():
    return pd.DataFrame(
        np.arange(30, dtype=float).reshape(5, 6),
        index=[f"TF{i}" for i in range(5)],
        columns=[f"cell{i}" for i in range(6)],
    )


@pytest.mark.fast
def test_plot_heatmap_and_embedding(activity):
    ax = plot_heatmap(activity, features=["TF0", "TF1"], colorbar=False)
    assert len(ax.images) == 1
    emb = pd.DataFrame(np.arange(12).reshape(6, 2), index=activity.columns,
                       columns=["UMAP1", "UMAP2"])
    ax = plot_reduced_dim(emb, color=["A", "A", "B", "B", "A", "B"])
    assert len(ax.collections) == 2
    ax = plot_regulon_reduced_dim(activity, feature="TF0")
    assert len(ax.collections) == 1


@pytest.mark.fast
def test_plot_expression_and_radviz(activity):
    fig, axes = plot_expression(activity, ["A", "A", "B", "B", "A", "B"],
                                 features=["TF0", "TF1"], ncols=2)
    assert axes.size == 2
    df = pd.DataFrame(np.eye(6, 3), columns=["A", "B", "C"])
    ax = plot_radviz(df, anchors=["A", "B", "C"])
    assert ax.get_aspect() == 1.0
