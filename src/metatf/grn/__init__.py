"""GRN inference methods.

pcor / puic / genie3 wrap the campaign Rust winners (pcor has a NumPy
fallback); sincerities is the numba-compiled pure-Python winner.  All are
imported lazily from metatf.api to keep `import metatf` cheap.
"""
