"""Preliminary circuit-level benchmark via stim detector sampling.

Usage:
    python -u scripts/bench_circuit.py <72|144> <rounds> <shots> <p> [p ...] \
        [--basis Z|X] [--variants bp,admm] [--seed N] [--batch-size N]

The decoders use the probabilities of the undecomposed DEM mechanisms.
Results include the solver settings and the number of failed code blocks.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from qudec.circuit import build_memory_circuit, dem_check_matrix, sample_memory
from qudec.circuit_decoders import AdmmOsdDemDecoder, BpOsdDemDecoder
from qudec.codes import gross_code, logicals, medium_code

COLUMNS = ["code", "d", "p", "shots", "basis", "seed", "variant",
           "batch_size", "max_iter", "alpha", "rho", "osd_order", "osd_lam",
           "prior", "noise_model", "detectors", "observables", "mechanisms",
           "failures", "ler", "secs"]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def probability(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value < 0.5:
        raise argparse.ArgumentTypeError("must satisfy 0 <= p < 0.5")
    return value


def variants(value):
    names = value.split(",")
    if not names or any(name not in ("bp", "admm") for name in names):
        raise argparse.ArgumentTypeError("choose bp, admm, or bp,admm")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("variants must not repeat")
    return names


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", choices=("72", "144"))
    parser.add_argument("d", type=positive_int)
    parser.add_argument("shots", type=positive_int)
    parser.add_argument("ps", type=probability, nargs="+")
    parser.add_argument("--basis", choices=("Z", "X"), default="Z")
    parser.add_argument("--variants", type=variants, default=["bp", "admm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--bp-max-iter", type=positive_int, default=30)
    parser.add_argument("--bp-alpha", type=float, default=0.9)
    parser.add_argument("--admm-max-iter", type=positive_int, default=500)
    parser.add_argument("--admm-alpha", type=float, default=1.0)
    parser.add_argument("--admm-rho", type=float, default=2.0)
    parser.add_argument("--osd-order", type=int, choices=(0, 1), default=1)
    parser.add_argument("--osd-lam", type=positive_int)
    parser.add_argument("--output", help="CSV path; default results/circuit_<code>.csv")
    args = parser.parse_args(argv[1:])
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not np.isfinite(args.bp_alpha) or args.bp_alpha <= 0:
        parser.error("--bp-alpha must be finite and positive")
    if not np.isfinite(args.admm_alpha) or not 0 < args.admm_alpha < 2:
        parser.error("--admm-alpha must satisfy 0 < alpha < 2")
    if not np.isfinite(args.admm_rho) or args.admm_rho <= 0:
        parser.error("--admm-rho must be finite and positive")
    return args


def decode_failures(decoder, dets, obs, batch_size):
    """Count block failures while bounding dense solver batches."""
    failures = 0
    for start in range(0, len(dets), batch_size):
        stop = start + batch_size
        pred = decoder.decode_batch(dets[start:stop])
        if pred.shape != obs[start:stop].shape:
            raise ValueError("decoder prediction shape does not match observables")
        failures += int(np.any(pred != obs[start:stop], axis=1).sum())
    return failures


def main(argv=None):
    args = parse_args(sys.argv if argv is None else argv)
    h_x, h_z = medium_code() if args.code == "72" else gross_code()
    l_x, l_z = logicals(h_x, h_z)
    out_path = args.output or os.path.join("results", f"circuit_{args.code}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    if not new_file:
        with open(out_path, newline="") as fh:
            if next(csv.reader(fh), None) != COLUMNS:
                raise ValueError(f"CSV schema differs at {out_path}; choose --output")
    for p in args.ps:
        circ = build_memory_circuit(h_x, h_z, l_x, l_z, args.d, p, args.basis)
        # BP and ADMM accept hyperedges; forcing graphlike decomposition can
        # reject perfectly valid BB-code error mechanisms.
        dem = circ.detector_error_model()
        h, l, probabilities = dem_check_matrix(dem, return_probabilities=True)
        dets, obs = sample_memory(circ, args.shots, args.seed)
        for name in args.variants:
            settings = dict(osd_order=args.osd_order, osd_lam=args.osd_lam,
                            probabilities=probabilities)
            if name == "bp":
                max_iter, alpha, rho = args.bp_max_iter, args.bp_alpha, ""
                dec = BpOsdDemDecoder(h, l, max_iter=max_iter, alpha=alpha,
                                      **settings)
            else:
                max_iter = args.admm_max_iter
                alpha, rho = args.admm_alpha, args.admm_rho
                dec = AdmmOsdDemDecoder(h, l, max_iter=max_iter, alpha=alpha,
                                        rho=rho, **settings)
            t0 = time.perf_counter()
            failures = decode_failures(dec, dets, obs, args.batch_size)
            secs = time.perf_counter() - t0
            ler = failures / args.shots
            with open(out_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                if new_file:
                    writer.writeheader()
                    new_file = False
                writer.writerow({
                    "code": args.code, "d": args.d, "p": p,
                    "shots": args.shots, "basis": args.basis, "seed": args.seed,
                    "variant": name, "batch_size": args.batch_size,
                    "max_iter": max_iter, "alpha": alpha, "rho": rho,
                    "osd_order": args.osd_order, "osd_lam": args.osd_lam,
                    "prior": "dem", "noise_model": "depolarizing_measurement_flip",
                    "detectors": h.shape[0], "observables": l.shape[0],
                    "mechanisms": h.shape[1], "failures": failures,
                    "ler": f"{ler:.8g}", "secs": f"{secs:.3f}"})
            print(f"[circuit] {name} p={p:5.3f} basis={args.basis} "
                  f"failures={failures}/{args.shots} ler={ler:.5f} {secs:.1f}s")


if __name__ == "__main__":
    main()
