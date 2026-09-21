# Plotting compatibility

The upstream metaTF vignettes repeatedly use four view types: a scaled
regulon heatmap (`pheatmap`), a two-dimensional cell map (`scater`), grouped
expression distributions (`ggplot2`/`scater`), and a Radviz projection. The
optional `metatf[plot]` extra provides Matplotlib helpers with the same
orientation, labels, and restrained white-background style.

| Upstream view | Python helper |
| --- | --- |
| `heatmapPlot()` | `plot_heatmap()` |
| `plotReducedDim()` | `plot_reduced_dim()` |
| `plotRegulonReducedDim()` | `plot_regulon_reduced_dim()` |
| expression violin/box plots | `plot_expression()` |
| `doRadvizPlot()` | `plot_radviz()` |

Install the optional dependency and save publication-ready vector output:

```bash
python -m pip install -e '.[plot]'
```

```python
import matplotlib.pyplot as plt
from metatf import plot_heatmap, plot_regulon_reduced_dim

plot_heatmap(activity, annotations=cell_metadata)
plt.savefig("regulon_heatmap.pdf", bbox_inches="tight")

plot_regulon_reduced_dim(activity, feature="GATA1")
plt.savefig("gata1_umap.pdf", bbox_inches="tight")
```

`activity` follows metaTF's convention of features/regulons × cells.
`plot_regulon_reduced_dim()` computes a deterministic PCA embedding when no
embedding is supplied; pass an existing UMAP/TSNE/PCA matrix to preserve the
coordinates from another workflow. Plotting is isolated from the numerical
core, so installations without Matplotlib remain supported.
