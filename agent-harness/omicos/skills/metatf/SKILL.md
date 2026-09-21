---
id: metatf
name: metatf
description: Run or interpret metaTF-py GRN inference and regulon activity in Omicos with the native Python API or a portable CLI launcher; preserve inputs, statistical semantics and existing results.
tier: community
category: general_omics_analysis
summary: metaTF-py GRN and regulon activity with portable execution and explicit scientific semantics.
execution_mode: packaged_python
runtime_entrypoint: scripts/run_metatf.py
---

# metaTF analysis

Use the prepared Python environment containing `metatf` (the distribution
and import name for metaTF-py). Reuse existing results where appropriate.
Do not install packages on every run or silently switch methods/backends.

Separate the requested task: `infer-grn` returns a regulator-by-target weight
matrix; `activity` takes an explicit TF/target edge table and returns scores.
Do not turn every inferred weight into a regulon without an explicit edge
selection rule. Establish species, identifiers, expression scale and matrix
orientation from provenance rather than filenames. Preserve user-fixed
methods, seeds, genes, cells/samples and ULM/GENIE3 semantics.

## Assess and prepare inputs yourself

You are responsible for assessing and preparing data before execution; the
launcher does not do this for you. Use Omicos's existing Python/shell tools to
inspect the file, relevant AnnData slots, metadata and useful numeric summaries.
Infer whether the candidate is counts, non-log normalized, log-normalized or
scaled/residual expression, then select and prepare it for the requested method.
Use the input and method references below for concrete guidance; do not apply
one counts-first or normalize-and-log recipe to every method.

Reuse a suitable processed matrix as-is. When transformation or slot selection
is warranted, perform it on a working copy and continue without asking for
approval of routine steps. Distinguish supported inference from recorded
provenance, avoid duplicate transformations, and preserve user-fixed scientific
choices. Resolve uncertainty from existing files/context first; ask only when
remaining ambiguity would materially alter the analysis. Record the chosen
slot, scale/evidence and actual transformations in the existing job/notebook
or result summary. No new detector, mandatory stage or report format is needed.

## Portable entrypoint

Resolve `scripts/run_metatf.py` using
`skill_resource(name="metatf", path="scripts/run_metatf.py", include_runtime_path=true)`.
Run the returned `runtime_path` with the actual analysis interpreter:

```text
<python> <runtime_path> infer-grn --method pcor --input exp.csv --output W.csv --threads 4
<python> <runtime_path> activity --method ulm --semantics full_axis --input exp.csv --network net.csv --output activity.csv --threads 4
```

The launcher delegates to `python -m metatf.cli`. `doctor --threads 4` diagnoses
the interpreter and importable backends; it is optional, not a mandatory stage.
`infer-grn --help` and `activity --help` show the installed CLI's options.
The entrypoint travels with the Skill; it does not need this repository checkout.

The launcher emits the actual command and thread/affinity report on stderr,
then preserves native outputs and exit status. Use fresh task output paths:
the native writer can overwrite existing files. Check the actual tables,
labels, coverage, non-finite values and warnings before declaring completion.
ULM writes both activity and a sibling p-value table. No manifest, retry daemon,
mandatory notebook or production catalog deployment is implied by this bundle.

## References on demand

- [Inputs and outputs](references/inputs-and-outputs.md): LLM-led scale assessment/preparation, explicit AnnData slots, metadata alignment and results.
- [Methods and interpretation](references/methods.md): method-specific input choices, the eight methods, ULM/GENIE3 modes and statistical boundaries.
- [Runtime](references/runtime.md): installation, CPU allocation, diagnostics and custom Python API calls.
