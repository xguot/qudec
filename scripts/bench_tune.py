"""Paired ADMM rho/alpha and OSD-CS tuning under closed memory noise.

  python scripts/bench_tune.py 72 6 300 .02 --rhos .5,1,2,4,8 --alphas 1,1.5,1.8

Uses the same bounded benchmark and JSONL outcome format as bench_phenom.
Plain and weighted ADMM receive the same grid under greedy and lambda
OSD. Every effective configuration appears exactly once. BP supplies
the matching baseline for each OSD setting.
"""

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .bench_phenom import common_parser, environment_metadata, positive_int, run_variant
else:
    from bench_phenom import common_parser, environment_metadata, positive_int, run_variant
from qudec.admm import AdmmOsdDecoder
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals, medium_code
from qudec.phenom import phenom_priors, sample_phenom


def positive_list(value):
    try:
        values = [float(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("grid values must be distinct")
    return values


def parse_args(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--rhos", type=positive_list, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--alphas", type=positive_list, default=[1.0, 1.5, 1.8])
    parser.add_argument("--lam", type=positive_int, default=60)
    args = parser.parse_intermixed_args(argv)
    if args.seed < 0 or any(v >= 2 for v in args.alphas):
        parser.error("require seed >= 0 and alpha < 2")
    args.ps = list(dict.fromkeys(args.ps or [0.02]))
    return args


def main(argv=None):
    args = parse_args(argv)
    h_x, h_z = medium_code() if args.code == "72" else gross_code()
    matrices = h_x, h_z, *logicals(h_x, h_z)
    output = args.output or Path("results") / f"tune_{args.code}.jsonl"
    environment = environment_metadata()
    for p in args.ps:
        samples = sample_phenom(h_x, h_z, p, args.rounds, args.shots,
                                seed=args.seed, p_meas=args.p_meas,
                                terminal_perfect=not args.legacy_open_boundary)
        p_x, p_z = phenom_priors(h_x, h_z, args.rounds, p, args.p_meas)
        bp = {"max_iter": 30, "alpha": 0.9, "osd_order": 1}
        variants = [("bp", BpOsdDecoder, bp),
                    ("bp-lam", BpOsdDecoder, dict(bp, osd_lam=args.lam))]
        for name, weights in (("admm", {}), ("admm-w", {
                "weights_x": np.log((1-p_x)/p_x),
                "weights_z": np.log((1-p_z)/p_z)})):
            for rho in args.rhos:
                for alpha in args.alphas:
                    for osd_lam in (None, args.lam):
                        label = f"{name}-r{rho:g}-a{alpha:g}"
                        if osd_lam is not None:
                            label += f"-lam{osd_lam}"
                        kw = dict(max_iter=150, osd_order=1, osd_lam=osd_lam,
                                  rho=rho, alpha=alpha, **weights)
                        variants.append((label, AdmmOsdDecoder, kw))
        for name, cls, kw in variants:
            run_variant(args, matrices, p, samples, name, cls, kw, output, environment)


if __name__ == "__main__":
    main()
