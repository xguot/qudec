"""Circuit-level benchmark via stim detector sampling.

Builds a d-round memory circuit at circuit-level depolarizing rate p,
extracts the detector error model, decodes the sampled shots with the
dem-matrix decoders, and appends one row per (variant, p, basis) to
results/circuit_<code>.csv. The uniform-prior dem decoding is the
standard first-order decoder for circuit-level dems; report it as
preliminary.

Usage:

    python -u scripts/bench_circuit.py <72|144> <rounds> <shots> <p> [p ...] \\
        [--basis Z|X] [--variants bp,admm] [--seed N] [--osd-lam K]
"""

import csv
import os
import sys
import time

from qudec.circuit import (
    build_memory_circuit,
    dem_check_matrix,
    sample_memory,
)
from qudec.circuit_decoders import AdmmOsdDemDecoder, BpOsdDemDecoder
from qudec.codes import gross_code, logicals, medium_code

COLUMNS = ["code", "d", "p", "shots", "basis", "seed", "variant",
           "ler", "secs"]


def parse_args(argv):
    code = argv[1]
    d = int(argv[2])
    shots = int(argv[3])
    ps = [float(t) for t in argv[4:] if not t.startswith("-")]
    basis = "Z"
    variants = ["bp", "admm"]
    seed = 0
    osd_lam = None
    rest = [t for t in argv[4:] if t.startswith("-")]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--basis":
            basis = rest[i + 1]
            i += 2
        elif tok == "--variants":
            variants = rest[i + 1].split(",")
            i += 2
        elif tok == "--seed":
            seed = int(rest[i + 1])
            i += 2
        elif tok == "--osd-lam":
            osd_lam = int(rest[i + 1])
            i += 2
        else:
            raise SystemExit(f"unknown option {tok}")
    return code, d, shots, ps, basis, variants, seed, osd_lam


def main():
    code, d, shots, ps, basis, variants, seed, osd_lam = parse_args(sys.argv)
    h_x, h_z = medium_code() if code == "72" else gross_code()
    l_x, l_z = logicals(h_x, h_z)
    out_path = os.path.join("results", f"circuit_{code}.csv")
    os.makedirs("results", exist_ok=True)
    new_file = not os.path.exists(out_path)
    for p in ps:
        circ = build_memory_circuit(h_x, h_z, l_x, l_z, d, p, basis)
        dem = circ.detector_error_model(decompose_errors=True)
        h, l = dem_check_matrix(dem)
        dets, obs = sample_memory(circ, shots, seed)
        for name in variants:
            t0 = time.time()
            if name == "bp":
                dec = BpOsdDemDecoder(h, l, p=p, osd_lam=osd_lam)
            else:
                dec = AdmmOsdDemDecoder(h, l, osd_lam=osd_lam)
            pred = dec.decode_batch(dets)
            ler = float((pred != obs).any(axis=1).mean())
            with open(out_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                if new_file:
                    writer.writeheader()
                    new_file = False
                writer.writerow({
                    "code": code, "d": d, "p": p, "shots": shots,
                    "basis": basis, "seed": seed, "variant": name,
                    "ler": f"{ler:.5f}", "secs": f"{time.time() - t0:.1f}"})
            print(f"[circuit] {name} p={p:5.3f} ler={ler:.5f} "
                  f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
