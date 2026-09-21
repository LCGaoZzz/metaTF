"""metatf command-line interface.

    metatf infer-grn --method pcor     --input exp.csv --output W.csv [--threads N --seed S]
    metatf infer-grn --method genie3   --input exp.csv --output W.csv --mode parity --n-trees 1000
    metatf infer-grn --method sincerities --input exp.csv --col-data time.csv --output W.csv
    metatf activity  --method viper    --input exp.csv --network net.csv --output nes.csv
    metatf activity  --method ulm      --input exp.csv --network net.csv --semantics full_axis --output t.csv
"""

from __future__ import annotations

import argparse
import sys

from .adapters import read_expression, read_network, write_table
from .api import infer_grn, regulon_activity


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True,
                   help="expression matrix, genes x cells (.csv/.tsv/.csv.gz/.parquet/.h5ad)")
    p.add_argument("--output", required=True, help="output table (.csv/.tsv/.parquet)")
    p.add_argument("--threads", type=int, default=None,
                   help="worker threads (default min(8, cores), hard cap 63)")
    p.add_argument("--seed", type=int, default=1, help="RNG seed (genie3; campaign refs use 1)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metatf",
        description="metaTF GRN inference and regulon activity (no R runtime).")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("infer-grn", help="infer a GRN weight matrix")
    _add_common(g)
    g.add_argument("--method", required=True,
                   choices=["pcor", "puic", "genie3", "sincerities"])
    g.add_argument("--col-data", default=None,
                   help="sincerities only: CSV with columns sample,time (time numeric, >=3 points)")
    g.add_argument("--exp-cutoff", type=float, default=0.0,
                   help="metaTF gene filter (0 = keep rowSums > 0; (0,1) fraction; >=1 count)")
    g.add_argument("--n-trees", type=int, default=1000, help="genie3 only")
    g.add_argument("--tree-method", default="RF", choices=["RF", "ET"], help="genie3 only")
    g.add_argument("--K", default="sqrt", help="genie3 mtry: 'sqrt' | 'all' | int")
    g.add_argument("--mode", default="fast", choices=["fast", "parity"],
                   help="genie3 only: parity = bit-level R single-core RNG stream")

    a = sub.add_parser("activity", help="score regulon activity per cell")
    _add_common(a)
    a.add_argument("--method", required=True, choices=["viper", "aucell", "gsva", "ulm"])
    a.add_argument("--network", default=None,
                   help="edge table tf/target/weight (.csv/.tsv/.csv.gz/.parquet)")
    a.add_argument("--semantics", default="subset", choices=["subset", "full_axis"],
                   help="ulm only (see docs/semantics.md)")
    a.add_argument("--unweighted", action="store_true",
                   help="viper only: ignore the weight column (all-1 likelihoods, "
                        "dfToList unweighted entry)")
    a.add_argument("--minsize", type=int, default=5,
                   help="viper min regulon size / ulm min matched targets")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    exp = read_expression(args.input)

    if args.command == "infer-grn":
        col_data = None
        if args.method == "sincerities":
            if not args.col_data:
                ap.error("--col-data is required for --method sincerities")
            import pandas as pd
            col_data = pd.read_csv(args.col_data)
            if col_data.shape[1] < 2:
                ap.error("--col-data needs sample,time columns")
        K = args.K
        if K not in ("sqrt", "all"):
            try:
                K = int(K)
            except ValueError:
                ap.error("--K must be 'sqrt', 'all' or an integer")
        out = infer_grn(exp, args.method, threads=args.threads, seed=args.seed,
                        exp_cutoff=args.exp_cutoff, col_data=col_data,
                        n_trees=args.n_trees, tree_method=args.tree_method,
                        K=K, mode=args.mode)
        write_table(out, args.output)
        print(f"metatf infer-grn[{args.method}] -> {args.output} "
              f"({out.shape[0]} regulators x {out.shape[1]} targets)")
        return 0

    # activity
    if not args.network:
        ap.error("--network is required for the activity subcommand")
    net = read_network(args.network)
    weighted = False if args.unweighted else None
    res = regulon_activity(exp, network=net, method=args.method,
                           weighted=weighted, minsize=args.minsize,
                           semantics=args.semantics, threads=args.threads)
    if args.method == "ulm":
        activity, pval = res
        base = args.output
        write_table(activity, base)
        pv_out = _sibling(base, "_pval")
        write_table(pval, pv_out)
        print(f"metatf activity[ulm/{args.semantics}] -> {base} (+ p-values {pv_out}) "
              f"({activity.shape[0]} TFs x {activity.shape[1]} cells)")
    else:
        write_table(res, args.output)
        print(f"metatf activity[{args.method}] -> {args.output} "
              f"({res.shape[0]} regulons x {res.shape[1]} cells)")
    return 0


def _sibling(path: str, suffix: str) -> str:
    import os
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
