"""Portable Omicos launcher; numerical work stays in the installed metatf CLI."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys

_THREAD_ENV = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "PYTHON_CPU_COUNT",
)


def positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threads must be a positive integer") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("threads must be a positive integer")
    return n


def configure_threads(requested: int | None) -> dict:
    """Constrain this launcher and its replacement process, never the host."""
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError("threads must be a positive integer")
    have_affinity = hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
    before = sorted(os.sched_getaffinity(0)) if have_affinity else None
    available = len(before) if before is not None else (os.cpu_count() or 1)
    if available < 1:
        raise RuntimeError("The process has no available CPUs")
    effective = min(8 if requested is None else requested, available, 63)
    after = None
    if have_affinity:
        os.sched_setaffinity(0, before[:effective])
        after = sorted(os.sched_getaffinity(0))
        if not after or len(after) > effective or not set(after).issubset(before):
            raise RuntimeError("Could not enforce the requested CPU affinity")
        effective = min(effective, len(after))
    for key in _THREAD_ENV:
        os.environ[key] = str(effective)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    return {
        "requested_threads": requested, "effective_threads": effective,
        "affinity_enforced": have_affinity, "affinity_before": before,
        "affinity_after": after,
        "environment": {key: os.environ[key] for key in _THREAD_ENV},
    }


def doctor(resources: dict) -> int:
    report = {
        "python": sys.executable, "python_version": sys.version,
        "resources": resources, "core_importable": False,
        "analysis_executed": False,
    }
    try:
        import metatf
        report.update(core_importable=True, metatf_version=metatf.__version__,
                      metatf_module=metatf.__file__,
                      rust_importable=metatf.rust_available())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["distributions"] = {}
    for name in ("metatf", "metatf-rust", "numpy", "pandas", "scipy", "numba", "anndata", "pyarrow"):
        try:
            report["distributions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["distributions"][name] = None
    print(json.dumps(report, indent=2))
    return 0 if report["core_importable"] else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    usage = (
        "%(prog)s {doctor|infer-grn|activity} [native metatf options]\n"
        "\nExamples:\n"
        "  %(prog)s doctor --threads 4\n"
        "  %(prog)s infer-grn --method pcor --input exp.csv --output W.csv --threads 4\n"
        "  %(prog)s activity --help\n"
        "\nUses the current Python environment; installs nothing. Native options and\n"
        "outputs are unchanged except threads are bounded by available CPUs and 63."
    )
    parser = argparse.ArgumentParser(description=__doc__, usage=usage)
    parser.add_argument("command", choices=["doctor", "infer-grn", "activity"])
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(args)
    # Only inspect the native resource option. All scientific options stay native.
    threads_parser = argparse.ArgumentParser(add_help=parsed.command == "doctor")
    threads_parser.add_argument("--threads", type=positive_int, default=None)
    if parsed.command == "doctor":
        options = threads_parser.parse_args(parsed.native_args)
    else:
        options, _ = threads_parser.parse_known_args(parsed.native_args)
    try:
        resources = configure_threads(options.threads)
        if parsed.command == "doctor":
            return doctor(resources)
        native_args = [parsed.command, *parsed.native_args,
                       "--threads", str(resources["effective_threads"])]
        print(json.dumps({"event": "metatf_resources", "python": sys.executable,
                          "argv": native_args, **resources}), file=sys.stderr, flush=True)
        # Replace, rather than add a worker/job manager. A fresh interpreter also
        # observes PYTHON_CPU_COUNT where supported, before numerical imports.
        os.execve(sys.executable, [sys.executable, "-m", "metatf.cli", *native_args],
                  os.environ.copy())
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"metatf harness: {exc}\n")
    return 0  # execve does not return in a real run


if __name__ == "__main__":
    raise SystemExit(main())
