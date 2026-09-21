# Runtime and resources

## Existing environment first

Use the actual analysis interpreter, not an assumed `python` executable or
hard-coded server path. Distribution/import name: **`metatf`**, not `metaTF-py`.
The Agent/Skill copy does not install that package. In an authorized checkout,
prepare the environment only when necessary:

```bash
python -m pip install -e .
# Select extras only for the task: anndata, parquet, numba, plot, test.
python -m pip install -e '.[anndata]'
# Optional Rust accelerators, with the required build tools:
PYTHON="$(command -v python)" bash build.sh
```

PCOR has a NumPy fallback. PUIC and GENIE3 require compatible Rust extensions;
missing backends must remain visible errors, not silently substituted methods.
SINCERITIES can use optional numba. Install/build operations use the host's
normal package tools and permissions; the launcher never downloads or installs.

## CPU policy before computation

`run_metatf.py <command> --threads N` derives its allocation from the process's
inherited CPU affinity where supported, with default 8 and a maximum of 63.
The effective count is bounded by the available allocation. It selects actual
allowed CPU IDs, never a hard-coded `0-63` range. On affinity-capable systems,
failure to apply/verify affinity aborts before computation. Else the report
explicitly says `affinity_enforced: false`; environment limits are not a claim
of OS-enforced CPU isolation.

The launcher sets BLAS/OpenMP/NumExpr/numba/Rayon thread environments before
loading numerical libraries. It replaces itself with a fresh interpreter
running `-m metatf.cli` and passes the effective native `--threads` value.
`PYTHON_CPU_COUNT` is also supplied for interpreters that honor it at startup;
older Python may still report host-wide `os.cpu_count()`. This affects the
launched process, not global host settings. It does not impose a RAM limit,
guarantee CPU-seconds quotas or claim all scoring methods use multiple threads.
Use the host's existing scheduler/resource controls when stronger limits matter.

The JSON startup record is written to **stderr** with event `metatf_resources`.
It contains the actual native arguments, interpreter, requested/effective
threads, affinity and applied environment. The native CLI then owns stdout,
output files, error handling, signals and exit status. Preserve this record in
the existing Omicos job log. There is no polling daemon or second job registry.

## Diagnostics and custom API work

`doctor [--threads N]` prints JSON with interpreter, versions, core importability,
importable Rust backends and resource policy. It exits nonzero when core import
fails. A zero exit means the core imported, **not** that every optional backend
exists, the data is valid, a numerical kernel passed or an analysis ran.
A missing module calls for checking the interpreter/environment; an ABI/import
error calls for checking the installed wheel, not changing the requested method.

The portable CLI deliberately preserves the native CLI's scope. Use
`metatf.infer_grn` / `metatf.regulon_activity` directly for a regulator subset,
gene-set dictionaries, explicit AnnData slots or method-specific keyword
arguments unavailable in CLI help. Set resources before imports in that fresh
analysis process using the host's execution tools; changing environment
variables after NumPy/numba/Rayon initialization is not equivalent. Inspect the
installed callable signature and method implementation when selecting kwargs;
this launcher does not add kwargs to the scientific API.

For long jobs, reuse Omicos's existing shell/session/job facilities. Inspect
existing artifacts after interruption; the native commands have no resumable
checkpoint contract. Do not label a partial output as complete or automatically
repeat a costly run merely because the interactive request timed out.
