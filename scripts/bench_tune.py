"""Tune the plain-ADMM hyperparameters under phenomenological noise.

Sweeps the ADMM penalty rho and over-relaxation alpha on the points
where plain ADMM trails BP+OSD, and compares the one-pass greedy OSD
sweep against the reference OSD-CS(lambda = 60) on both BP+OSD and
ADMM+OSD. The weighted variant is included as the paper anchor. One
CSV row per (variant, p) is appended to results/tune_<code>.csv.

Usage:

    python -u scripts/bench_tune.py <72|144> <rounds> [shots] [p ...] \\
        [--rhos 1,2,4] [--alphas 1.0,1.5,1.8] [--lam 60] [--seed N]
"""

import csv
import os
import sys
import time

import numpy as np

from qudec.admm import AdmmOsdDecoder
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals, medium_code
from qudec.phenom import PhenomDecoder, benchmark_phenom

COLUMNS = ["code", "d", "p", "shots", "seed", "variant", "rho", "alpha",
           "osd_lam", "ler", "ler_x", "ler_z", "invalid_x", "invalid_z",
           "secs"]


def parse_args(argv):
    """Return (code, rounds, shots, ps, rhos, alphas, lam, seed)."""
    code = argv[1]
    d = int(argv[2])
    shots = int(argv[3]) if len(argv) > 3 else 300
    ps = [0.02]
    rhos = [1.0, 2.0, 4.0]
    alphas = [1.0, 1.5, 1.8]
    lam = 60
    seed = 0
    rest = argv[4:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--rhos":
            rhos = [float(x) for x in rest[i + 1].split(",")]
            i += 2
        elif tok == "--alphas":
            alphas = [float(x) for x in rest[i + 1].split(",")]
            i += 2
        elif tok == "--lam":
            lam = int(rest[i + 1])
            i += 2
        elif tok == "--seed":
            seed = int(rest[i + 1])
            i += 2
        elif tok.startswith("-"):
            raise SystemExit(f"unknown option {tok}")
        else:
            ps = [float(t) for t in rest[i:]]
            break
    return code, d, shots, ps, rhos, alphas, lam, seed


def run(dec, h_x, h_z, l_x, l_z, p, d, shots, seed):
    t0 = time.time()
    res = benchmark_phenom(dec, h_x, h_z, l_x, l_z, p, d, shots, seed=seed)
    res["secs"] = time.time() - t0
    return res


def main():
    code, d, shots, ps, rhos, alphas, lam, seed = parse_args(sys.argv)
    h_x, h_z = medium_code() if code == "72" else gross_code()
    l_x, l_z = logicals(h_x, h_z)
    out_path = os.path.join("results", f"tune_{code}.csv")
    os.makedirs("results", exist_ok=True)
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for p in ps:
            variants = [
                ("bp", BpOsdDecoder,
                 {"max_iter": 30, "osd_order": 1}, None, 0.0),
                ("bp-lam", BpOsdDecoder,
                 {"max_iter": 30, "osd_order": 1}, lam, 0.0),
                ("admm", AdmmOsdDecoder,
                 {"max_iter": 150, "osd_order": 1, "max_r": 4}, None, 2.0),
                ("admm-lam", AdmmOsdDecoder,
                 {"max_iter": 150, "osd_order": 1, "max_r": 4}, lam, 2.0),
            ]
            for rho in rhos:
                for alpha in alphas:
                    variants.append(
                        (f"admm-r{rho:g}-a{alpha:g}", AdmmOsdDecoder,
                         {"max_iter": 150, "osd_order": 1, "max_r": 4,
                          "rho": rho, "alpha": alpha}, None, rho))
            for name, cls, kw, osd_lam, rho in variants:
                if osd_lam is not None:
                    kw = dict(kw, osd_lam=osd_lam)
                dec = PhenomDecoder(cls, h_x, h_z, l_x, l_z, d,
                                    p_x=2 * p / 3, p_z=2 * p / 3, **kw)
                res = run(dec, h_x, h_z, l_x, l_z, p, d, shots, seed)
                row = {"code": code, "d": d, "p": p, "shots": shots,
                       "seed": seed, "variant": name, "rho": rho,
                       "alpha": kw.get("alpha", 1.0),
                       "osd_lam": osd_lam if osd_lam else "",
                       "ler": f"{res['ler']:.5f}",
                       "ler_x": f"{res['ler_x']:.5f}",
                       "ler_z": f"{res['ler_z']:.5f}",
                       "invalid_x": res["invalid_x"],
                       "invalid_z": res["invalid_z"],
                       "secs": f"{res['secs']:.1f}"}
                writer.writerow(row)
                fh.flush()
                print("[tune]", name, f"p={p:5.3f}",
                      f"ler={res['ler']:.5f}", f"{res['secs']:.0f}s")


if __name__ == "__main__":
    main()
