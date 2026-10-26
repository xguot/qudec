"""Circuit-level benchmark and tuning via stim detector sampling.

Usage:
    python -u scripts/bench_circuit.py <72|144> <rounds> <shots> <p> [p ...] \
        [--basis Z|X] [--variants bp,admm,admm-u,admm-w,admm-ldr] \
        [--rhos 0.5,1,2,4,8] [--alphas 1,1.5,1.8] \
        [--seed N] [--batch-size N] [--output path.csv|path.jsonl]

The decoders use the probabilities of the undecomposed DEM mechanisms.
CSV output preserves the legacy marginal schema (COLUMNS); JSONL output
records per-shot failure bitstrings and sample hashes for paired McNemar
tests via scripts/analyze_phenom.py.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

import numpy as np

try:
    from scripts.bench_phenom import environment_metadata
except ImportError:
    from bench_phenom import environment_metadata
from qudec.circuit import build_memory_circuit, dem_check_matrix, sample_memory
from qudec.circuit_decoders import AdmmOsdDemDecoder, BpOsdDemDecoder
from qudec.codes import gross_code, logicals, medium_code

COLUMNS = ["code", "d", "p", "shots", "basis", "seed", "variant",
           "batch_size", "max_iter", "alpha", "rho", "osd_order", "osd_lam",
           "prior", "noise_model", "detectors", "observables", "mechanisms",
           "failures", "ler", "secs"]
VARIANTS = ("bp", "admm", "admm-u", "admm-w", "admm-ldr")


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
    if not names or any(name not in VARIANTS for name in names):
        raise argparse.ArgumentTypeError(
            "choose from " + ",".join(VARIANTS))
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("variants must not repeat")
    return names


def positive_grid(value):
    try:
        items = [float(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not items or any(not np.isfinite(v) or v <= 0 for v in items):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("grid values must not repeat")
    return items


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
    parser.add_argument("--rhos", type=positive_grid,
                        help="optional comma-separated ADMM rho grid for circuit tuning")
    parser.add_argument("--alphas", type=positive_grid,
                        help="optional comma-separated ADMM alpha grid for circuit tuning")
    parser.add_argument("--osd-order", type=int, choices=(0, 1), default=1)
    parser.add_argument("--osd-lam", type=positive_int)
    parser.add_argument("--output", help="CSV or JSONL path; default results/circuit_<code>.csv")
    args = parser.parse_args(argv[1:])
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not np.isfinite(args.bp_alpha) or args.bp_alpha <= 0:
        parser.error("--bp-alpha must be finite and positive")
    if not np.isfinite(args.admm_alpha) or not 0 < args.admm_alpha < 2:
        parser.error("--admm-alpha must satisfy 0 < alpha < 2")
    if not np.isfinite(args.admm_rho) or args.admm_rho <= 0:
        parser.error("--admm-rho must be finite and positive")
    if args.alphas is not None and any(a >= 2 for a in args.alphas):
        parser.error("--alphas must satisfy 0 < alpha < 2")
    return args


def sample_fingerprint(dets, obs):
    digest = hashlib.sha256()
    for arr in (dets, obs):
        digest.update(str(arr.shape).encode())
        digest.update(np.asarray(arr, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def decode_failures(decoder, dets, obs, batch_size, return_outcomes=False):
    """Count block failures while bounding dense solver batches."""
    failed = np.zeros(len(dets), dtype=bool)
    for start in range(0, len(dets), batch_size):
        stop = min(len(dets), start + batch_size)
        pred = decoder.decode_batch(dets[start:stop])
        if pred.shape != obs[start:stop].shape:
            raise ValueError("decoder prediction shape does not match observables")
        failed[start:stop] = np.any(pred != obs[start:stop], axis=1)
    failures = int(failed.sum())
    if return_outcomes:
        return failures, failed
    return failures


def expand_runs(args):
    """Expand requested variants across any supplied ADMM tuning grid."""
    rhos = args.rhos or [args.admm_rho]
    alphas = args.alphas or [args.admm_alpha]
    grid = args.rhos is not None or args.alphas is not None
    runs = []
    for name in args.variants:
        if name == "bp":
            runs.append((name, name, args.bp_max_iter, args.bp_alpha, ""))
        else:
            for rho in rhos:
                for alpha in alphas:
                    label = f"{name}-r{rho:g}-a{alpha:g}" if grid else name
                    runs.append((name, label, args.admm_max_iter, alpha, rho))
    return runs


def build_decoder(family, h, l, probabilities, max_iter, alpha, rho, args):
    settings = dict(osd_order=args.osd_order, osd_lam=args.osd_lam,
                    probabilities=probabilities)
    if family == "bp":
        return BpOsdDemDecoder(h, l, max_iter=max_iter, alpha=alpha, **settings)
    extra = {}
    if family == "admm-u":
        extra["weighted"] = False
    elif family == "admm-w":
        extra["weighted"] = True
    elif family == "admm-ldr":
        extra["ldr"] = True
        extra["ldr_outer"] = 4
    return AdmmOsdDemDecoder(h, l, max_iter=max_iter, alpha=alpha,
                             rho=float(rho), **settings, **extra)


def main(argv=None):
    args = parse_args(sys.argv if argv is None else argv)
    h_x, h_z = medium_code() if args.code == "72" else gross_code()
    l_x, l_z = logicals(h_x, h_z)
    out_path = args.output or os.path.join("results", f"circuit_{args.code}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    jsonl_mode = str(out_path).endswith(".jsonl")
    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    if not jsonl_mode and not new_file:
        with open(out_path, newline="") as fh:
            if next(csv.reader(fh), None) != COLUMNS:
                raise ValueError(f"CSV schema differs at {out_path}; choose --output")
    env_meta = environment_metadata() if jsonl_mode else None
    runs = expand_runs(args)
    for p in args.ps:
        circ = build_memory_circuit(h_x, h_z, l_x, l_z, args.d, p, args.basis)
        # BP and ADMM accept hyperedges; forcing graphlike decomposition can
        # reject perfectly valid BB-code error mechanisms.
        dem = circ.detector_error_model()
        h, l, probabilities = dem_check_matrix(dem, return_probabilities=True)
        dets, obs = sample_memory(circ, args.shots, args.seed)
        sample_id = sample_fingerprint(dets, obs) if jsonl_mode else None
        for family, label, max_iter, alpha, rho in runs:
            dec = build_decoder(family, h, l, probabilities, max_iter, alpha, rho, args)
            t0 = time.perf_counter()
            if jsonl_mode:
                failures, failed = decode_failures(
                    dec, dets, obs, args.batch_size, return_outcomes=True)
            else:
                failures = decode_failures(dec, dets, obs, args.batch_size)
                failed = None
            secs = time.perf_counter() - t0
            ler = failures / args.shots
            prior = "uniform" if family == "admm-u" else "dem"
            if jsonl_mode:
                device = ("cuda" if family != "bp" and env_meta["cuda_available"]
                          else "cpu")
                cfg = {"max_iter": max_iter, "alpha": float(alpha),
                       "osd_order": args.osd_order, "osd_lam": args.osd_lam,
                       "device": device}
                if family != "bp":
                    cfg.update(rho=float(rho), max_r=None,
                               ldr=family == "admm-ldr",
                               ldr_outer=4 if family == "admm-ldr" else 5,
                               ldr_beta=1.0)
                    if family in ("admm", "admm-w"):
                        cfg["weights"] = "normalized-dem-llr"
                record = {
                    "schema_version": 1, "code": args.code, "d": args.d,
                    "p": p, "p_meas": p, "basis": args.basis,
                    "shots": args.shots, "seed": args.seed,
                    "batch_size": args.batch_size, "variant": label,
                    "decoder": type(dec).__name__, "decoder_config": cfg,
                    "prior": prior, "model": "depolarizing_measurement_flip",
                    "protocol": f"circuit-{args.basis.lower()}-v1",
                    "projection": "exact",
                    "detectors": int(h.shape[0]),
                    "observables": int(l.shape[0]),
                    "mechanisms": int(h.shape[1]),
                    "sample_id": sample_id,
                    "failure": failures, "ler": ler,
                    "outcomes": {"failure": "".join(failed.astype(np.uint8).astype(str))},
                    "solver_device": device, "osd_device": "cpu",
                    "secs": secs, "environment": env_meta,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
            else:
                with open(out_path, "a", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                    if new_file:
                        writer.writeheader()
                        new_file = False
                    writer.writerow({
                        "code": args.code, "d": args.d, "p": p,
                        "shots": args.shots, "basis": args.basis, "seed": args.seed,
                        "variant": label, "batch_size": args.batch_size,
                        "max_iter": max_iter, "alpha": alpha, "rho": rho,
                        "osd_order": args.osd_order, "osd_lam": args.osd_lam,
                        "prior": prior, "noise_model": "depolarizing_measurement_flip",
                        "detectors": h.shape[0], "observables": l.shape[0],
                        "mechanisms": h.shape[1], "failures": failures,
                        "ler": f"{ler:.8g}", "secs": f"{secs:.3f}"})
            print(f"[circuit] {label} p={p:5.3f} basis={args.basis} "
                  f"failures={failures}/{args.shots} ler={ler:.5f} {secs:.1f}s")


if __name__ == "__main__":
    main()
