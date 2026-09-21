"""Rust extension backend resolution.

Resolution order for every accelerated method:

1. ``metatf_rust`` — the aggregate extension wheel built from ``rust/`` by
   ``build.sh`` (cargo workspace: rust_pcor_v2 + rust_puic + rust_genie3
   wrapped as submodules of one cdylib).  Preferred backend.
2. the standalone campaign wheels (``rust_pcor_v2`` / ``rust_puic`` /
   ``rust_genie3``) — same crate sources, installed separately during the
   campaign; kept as a compatibility path.
3. ``None`` — no Rust backend.  ``pcor`` then uses the NumPy same-path
   implementation (metatf.grn.pcor), ``puic``/``genie3`` raise with an
   install hint.

The import uses importlib and takes attributes off the returned module, so a
stale ``sys.modules`` entry (shared-kernel state) can never raise NameError —
the round-2 import bug class, fixed in round 3 and preserved here.
"""

from __future__ import annotations

import importlib

_FACADE = "metatf_rust"
_LEGACY = ("rust_pcor_v2", "rust_puic", "rust_genie3")

_INSTALL_HINT = (
    "the Rust extension wheel is not installed. Build it with ./build.sh from the "
    "metatf-py repository root (requires cargo + maturin; see docs/BUILD.md) or "
    "`pip install metatf-rust`. Only `pcor` has a pure-NumPy fallback; "
    "{method} has none."
)

_cache: dict = {}


def _load_facade():
    try:
        return importlib.import_module(_FACADE)
    except ImportError:
        return None


def _ext(sub: str):
    """Return the raw extension module for one accelerator, or None.

    ``sub`` is one of 'rust_pcor_v2' | 'rust_puic' | 'rust_genie3'.
    """
    if sub in _cache:
        return _cache[sub]
    mod = None
    facade = _load_facade()
    if facade is not None:
        try:
            mod = importlib.import_module(f"{_FACADE}.{sub}")
        except ImportError:
            mod = getattr(facade, sub, None)
    if mod is None:
        try:
            pkg = importlib.import_module(sub)
        except ImportError:
            pkg = None
        if pkg is not None:
            # legacy campaign wheels: rust_pcor_v2 / rust_puic are flat modules,
            # rust_genie3 is a package whose __init__ re-exports the extension.
            mod = pkg
            if sub == "rust_genie3":
                try:
                    mod = importlib.import_module("rust_genie3.rust_genie3")
                except ImportError:
                    pass
    _cache[sub] = mod
    return mod


def rust_available() -> dict:
    """Which accelerated backends are importable right now."""
    return {sub: _ext(sub) is not None for sub in _LEGACY}


def rust_backend(method: str):
    """Extension module for a method, or raise RuntimeError with install hint."""
    sub = {"pcor": "rust_pcor_v2", "puic": "rust_puic", "genie3": "rust_genie3"}[method]
    mod = _ext(sub)
    if mod is None:
        raise RuntimeError(_INSTALL_HINT.format(method=method))
    return mod
