# Inputs and outputs

## Expression

Tabular input is **genes x cells/samples**, with identifiers in the first
column/index. Native readers support CSV/TSV (including gzip), Parquet, and
AnnData. Inspect original axis labels, uniqueness, empty axes, missing/non-finite
values, species and scale before a meaningful run. Infer orientation from labels
and context, then transpose when needed; do not blindly transpose by shape.
Resolve cell versus biological-sample meaning and duplicate identifiers before
analysis. Numeric-looking IDs need care with CSV type inference; preserve their
literal values. Routine, justified preparation is authorized by the analysis
task: perform it, record it, and proceed rather than requesting approval for
each technical step.

### Assess scale using context and small checks

Use the existing workflow/history and input documentation together with the
actual matrix. In AnnData, inspect `X`, relevant `layers` and `raw.X` with their
own gene axes, plus available transformation metadata such as `uns["log1p"]`.
Names such as `counts`/`raw` and object-level metadata are clues, not guarantees:
`raw` stores a snapshot of the then-current `X`/`var`, and log metadata can be
stale or refer to a different slot. Float dtype does not rule out counts.

Choose useful, inexpensive checks: nonnegativity, integer-likeness of nonzero
values, ranges/quantiles, zeros, per-cell totals and gene-wise centering. Inspect
sparse values or bounded blocks instead of densifying the whole input just to
classify it; do not describe a sample check as an exhaustive validation.
Nonnegative integer-like values support a counts interpretation but cannot
prove its history; nonnegative fractional values may be normalized or logged.
Negative values rule out unmodified counts, but do not uniquely identify a
particular scaling/residual procedure. A maximum-value cutoff cannot settle
these distinctions. Metadata and numeric evidence should corroborate each other.

Make an informed choice when the available evidence is sufficient, even without
a formal provenance file; describe it as inferred rather than verified when
appropriate. If conflicting/insufficient evidence leaves a distinction that
would change preprocessing, inspect the prior workflow or another slot first.
Ask one targeted question only when that material ambiguity remains unresolved;
do not force a guess or require perfect provenance for every routine analysis.

### Prepare only what the selected method needs

Consult [method-specific guidance](methods.md) before transforming anything.
Prefer an already suitable matrix over reconstructing it from another slot.
Keep the original file and counts intact; use a working copy or separate output.
If confirmed counts need normalization/logging for the chosen workflow, do those
operations once, with explicit parameters. For example, for a chosen per-cell
size-normalized log-expression workflow, `normalize_total(target_sum=1e4)` then
natural-log `log1p` on a counts copy is one possible choice, not a metaTF default
or a universal rule. Match an existing justified normalization convention when
one is available. Skip normalization on suitable non-log normalized data and
skip both steps on suitable log-normalized data. Do not run normalization on
logged values. `expm1(log1p(normalized_counts))` recovers normalized values, not
original counts; never round, clip or exponentiate unknown data to fabricate
counts or to conceal incompatibility. Bulk and targeted-panel data need their
own context, not an automatic whole-transcriptome scRNA recipe.

Preserve the intended gene/cell universe, species and IDs. Do not automatically
HVG-subset, impute, batch-correct, merge duplicates or discard cells just to make
a call succeed. Check matrix shape/finite values, network overlap and memory
before the full call; a small numerical check is useful when needed, not a
mandatory extra analysis. Summarize the source slot, inferred/reported scale,
selection reason, transformations/parameters (including none), exclusions and
remaining uncertainty in the existing job/notebook or result summary. Do not
create an additional pipeline or require a separate manifest for these notes.

### Select AnnData slots explicitly

Native `.h5ad` input reads **`X` only**, transposes cells x genes to genes x cells
and densifies sparse matrices. It does not choose a counts layer or `raw.X`.
When you select another slot, use the public Python API or write the prepared
matrix to a separate supported input file before using the launcher. Merely
identifying `layers["counts"]` and then passing the original h5ad to the CLI
still reads `X`. The existing API also uses `X` when handed AnnData directly:
pass the prepared genes x cells DataFrame instead. Do not invent `--layer` or
`--use-raw` flags. Example below assumes your assessment selected counts;
change the slot for another justified choice, rather than treating it as a
counts-first default:

```python
import numpy as np
import pandas as pd

# Example only: the agent has assessed this slot as suitable for this method.
slot = "layers/counts"
if slot == "X":
    matrix, genes = adata.X, adata.var_names
elif slot == "raw.X":
    if adata.raw is None:
        raise ValueError("raw.X was requested but adata.raw is absent")
    matrix, genes = adata.raw.X, adata.raw.var_names
elif slot.startswith("layers/"):
    matrix, genes = adata.layers[slot.split("/", 1)[1]], adata.var_names
else:
    raise ValueError(f"Unsupported explicit slot: {slot}")
if not genes.is_unique or not adata.obs_names.is_unique:
    raise ValueError("Resolve duplicate gene/cell identifiers explicitly")
# Check the memory budget BEFORE densification; float64 alone costs ~8*genes*cells bytes.
values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
expression = pd.DataFrame(values.T, index=genes, columns=adata.obs_names)
```

The `raw.X` branch deliberately uses `raw.var_names`, not the current `var`.
Conversion can involve several allocations; the estimate is not peak memory.
GRN methods can additionally allocate dense gene x gene matrices. A regulator
subset is not a guarantee that upstream computation avoids those allocations.
Do not feed z-scored residuals to the default GRN filter without examining its
`rowSums > 0` semantics; it is not a variance filter.

## Networks and time metadata

Activity CLI input is a **long edge table** with `tf`, `target`, and optionally
`weight` columns, not the square GRN output. The reader lowercases column
names and fills missing weights with 1.0. Consequently, CLI VIPER defaults to
the weighted entry even for a two-column file; select `--unweighted` when the
unweighted/deduplicated entry is intended. Do not indiscriminately deduplicate
edges: weighted VIPER, unweighted scorers and ULM handle them differently.

A GRN-to-regulon conversion requires a recorded rule for TF eligibility,
self-edges, signs and edge selection. Do not invent a universal threshold,
include every dense weight or represent this step as motif validation.
Preserve the user's network source/species and report target intersection and
dropped regulons. Xenium panel coverage is not whole-transcriptome coverage;
the measured gene universe also affects scoring backgrounds.

SINCERITIES requires `--col-data` with `sample,time` columns (in that order),
where `sample` identifies the expression columns and `time` is numeric with
at least three distinct time points. Verify exact ID coverage and uniqueness,
then align by literal IDs, not row order or barcode heuristics. Keep donor and
replicate provenance separately; do not silently substitute pseudotime for
observed collection time.

## Outputs and completion

`infer-grn` writes regulator x target weights; PCOR without regulator subsetting
is a symmetric gene x gene matrix with diagonal 1. `activity` writes regulon/TF
x cell/sample scores. ULM additionally writes a p-value table: for
`--output activity.csv`, the sibling is `activity_pval.csv`. Use uncomplicated
CSV/TSV output names when predictable sibling naming matters; consult the
installed CLI for other suffix behavior.

The native writer can overwrite existing paths. Choose fresh output paths,
inspect exit status, then inspect table dimensions, labels, retained regulons,
non-finite values and relevant warnings. Preserve ULM NaNs for diagnosis rather
than silently replacing them with zero. Keep method/mode, network provenance,
input slot/scale, actual command, versions and thread report with the results.
No automatic run manifest is generated by this launcher.

Implementation anchors (reviewed base):
[adapters.py](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/src/metatf/adapters.py),
[cli.py](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/src/metatf/cli.py),
[api.py](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/src/metatf/api.py).

Input semantics references:
[AnnData raw snapshots](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.raw.html),
[Scanpy total normalization](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html),
[Scanpy log1p](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.log1p.html).
