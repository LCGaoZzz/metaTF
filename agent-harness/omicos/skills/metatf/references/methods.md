# Methods and interpretation

Choose from the question and fixed analysis specification, not a blanket
"best TF method" ranking. Use installed CLI help or Python signatures for
current options; do not fabricate unsupported flags.

| Task | Native method | Important boundary |
|---|---|---|
| GRN | `pcor` | Spearman-rank covariance / pseudoinverse; Rust optional, NumPy fallback available. Weights are partial associations, not causal validation. |
| GRN | `puic` | Requires a compatible Rust backend. Dense gene x gene allocation can dominate memory. Do not silently replace it with PCOR. |
| GRN | `genie3` | Requires Rust; preserve seed, trees, tree method, K and `fast`/`parity` choice. |
| GRN | `sincerities` | Requires measured-time metadata; no R runtime. Optional numba accelerates its Python implementation, not a guarantee of superiority to R. |
| Activity | `viper` | aREA NES by default; weighted likelihood and unweighted gene-list entries are distinct. Weighted duplicate edges are retained. |
| Activity | `aucell` | Normalized rank-based AUC; network weights are discarded and gene-set targets deduplicated. Background gene universe matters. |
| Activity | `gsva` | The implemented scope is GSVA's Poisson/RNA-seq path, not Gaussian, ssGSEA, PLAGE or z-score scoring. Select an appropriate nonnegative expression scale. |
| Activity | `ulm` | Returns t-statistics **and raw p-values**; explicitly retain the chosen `subset` or `full_axis` semantics. |

## Choose preparation for the method, not for the file suffix

These are agent decision aids, not new API requirements or a compulsory
preprocessing recipe. Preserve a specified reproduction protocol. Otherwise,
reuse the task's appropriate existing expression representation and perform
only the missing preparation; consult the installed implementation when needed.

| Method | Input assessment and preparation guidance |
|---|---|
| `pcor` | No counts-only requirement. An appropriate existing normalized/log-expression matrix can be reused. Per-cell normalization can change across-cell gene ranks, so do not claim counts and log-normalized expression are universally interchangeable. Check constant rows and the upstream `rowSums > 0` filter rather than feeding centered residuals blindly. |
| `puic` | Account for the API's optional `log_scale=True`, which performs `log2(x+1)`. Leave it false for already logged inputs; do not log externally and internally. Choose the intended scale before discretization. The native CLI has no `--log-scale` flag; use the API for that option. |
| `genie3` | No counts-only requirement. Keep a suitable existing normalized/log-expression matrix, or prepare counts for the chosen analysis once. Transforming expression changes the inference problem; do not change the reproduction input merely because `mode="parity"` exists. |
| `sincerities` | Preserve a justified expression scale consistently across time points and align the required sample/time metadata. Do not substitute pseudotime or invent collection times to complete preparation. |
| `aucell` | Scores within-cell ranks: reuse an appropriate counts or normalized/log-expression matrix without needless renormalization. Keep the intended background gene universe; gene-wise scaling, imputation or an HVG-only input can change the ranking/background. |
| `viper`, `ulm` | For routine scRNA activity, suitable existing normalized/log-expression data can be reused; this is not a counts-only interface. Preserve intentionally supplied signatures and model semantics rather than imposing a universal log step. For ULM `full_axis`, keep the intended full gene axis, not only network targets/HVGs. |
| `gsva` | This implementation is Poisson-only. For ordinary RNA-seq use, seek genuine count-scale input rather than automatically passing lognorm or residuals. Prefer an available validated counts slot; do not round or invert lognorm to manufacture counts. If none exists and no justified protocol resolves the choice, explain the mismatch rather than silently switching kernels or methods. |

For example, when `X` is known log-normalized, `layers["counts"]` contains
counts, and the task is routine ULM activity, reuse suitable `X` without another
normalization/log step. For Poisson GSVA on that same file, choose the counts
slot and explicitly feed it via the API or a prepared file. If `X` is scaled
but `raw.X` holds suitable log-expression, use `raw.X` with `raw.var_names`;
`raw` does not mean counts. These examples guide reasoning, not slot priorities.

## Modes that must not be conflated

GENIE3 `parity` uses the R-compatible single random stream; its thread argument
does not parallelize that mode. `fast` uses independent target streams. Do not
switch a fixed parity request to fast for throughput or describe both as
bit-identical to an R run.

ULM `subset` is the default: matched-target regression with degrees of freedom
`n_targets - 2` and guards for degenerate denominators. `full_axis` zero-pads
weights across the full expression gene axis and uses `n_genes - 2`; its
unguarded degenerate cases can produce NaNs. These are different statistics,
not merely different speeds. ULM averages duplicate edges and removes
zero-weight edges under this implementation's documented conventions.

AUCell drops gene sets at the missing-target boundary `>= 0.8` by default.
Coverage exclusions should be visible in interpretation, not mistaken for zero
activity. VIPER's `--unweighted` changes target deduplication as well as weights.
The CLI's `--minsize` controls VIPER/ULM; GSVA/AUCell method-specific tuning
requires the Python API rather than invented CLI flags.

## Evidence boundaries

Predicted GRN edges, regulon activity, TF RNA abundance, motif accessibility and
perturbation evidence are distinct measurements. This bundle adds no motif
pruning, binding evidence, causal intervention or differential-activity test.
Per-cell ULM p-values are not treatment comparisons across independent donors.
Use the biological replicate as the inferential unit in a separate appropriate
comparison; do not treat thousands of cells as thousands of independent mice.

The launcher does not normalize data, select HVGs, infer species, choose a TF
list, correct batches or convert a dense inferred GRN into a validated regulon.
The Agent performs justified preparation with existing tools and records it;
this is instructed LLM behavior, not an automatic feature of the launcher.
Algorithm benchmarks/parity evidence belong to the source repository and have
not been re-established by a portable-launcher smoke test.

Reviewed implementation and semantic documentation:
[api.py](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/src/metatf/api.py),
[semantics.md](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/docs/semantics.md).

PUIC transform anchor:
[grn/puic.py](https://github.com/LCGaoZzz/metaTF-py/blob/ad33b1ced1308596adfad643b870aca10402c9ef/src/metatf/grn/puic.py).
