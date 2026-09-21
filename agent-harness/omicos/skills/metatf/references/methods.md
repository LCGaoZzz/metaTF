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
Perform requested preparation with the existing tools and record it explicitly.
Algorithm benchmarks/parity evidence belong to the source repository and have
not been re-established by a portable-launcher smoke test.

Reviewed implementation and semantic documentation:
[api.py](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/src/metatf/api.py),
[semantics.md](https://github.com/LCGaoZzz/metaTF-py/blob/0ec381c9020cf62723b425ca0cf5f669d42801cf/docs/semantics.md).
