"""Sweep the phenomenological-noise LER of BP+OSD vs ADMM+OSD.

The detector syndromes come from d rounds of data plus measurement
errors; decoding runs on the time-expanded check matrices. This is the
first LP-family decoding result beyond code capacity for the bivariate
bicycle codes.

Usage:

    python -u scripts/bench_phenom.py <72|144> <rounds> [shots] \\
        [--variants a,b] [--seed N] [--osd-lam K] [--admm-rho R] \\
        [--admm-alpha A] [p ...]

Fan out high-shot confirmations with distinct seeds per job so the
sampled shots never repeat across jobs.
"""

import sys
import time

import numpy as np

from qudec.admm import AdmmOsdDecoder
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals, medium_code
from qudec.phenom import PhenomDecoder, benchmark_phenom


def parse_args(argv):
    """Return (code, rounds, shots, variants, ps, seed, osd_lam)."""
    code = argv[1]
    d = int(argv[2])
    shots = int(argv[3]) if len(argv) > 3 else 300
    variants = ["bp", "admm", "admm-w", "admm-ldr"]
    ps = [0.005, 0.01, 0.02, 0.03]
    seed = 0
    osd_lam = None
    admm_rho = None
    admm_alpha = None
    rest = argv[4:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--variants":
            variants = rest[i + 1].split(",")
            i += 2
        elif tok == "--seed":
            seed = int(rest[i + 1])
            i += 2
        elif tok == "--osd-lam":
            osd_lam = int(rest[i + 1])
            i += 2
        elif tok == "--admm-rho":
            admm_rho = float(rest[i + 1])
            i += 2
        elif tok == "--admm-alpha":
            admm_alpha = float(rest[i + 1])
            i += 2
        elif tok.startswith("-"):
            raise SystemExit(f"unknown option {tok}")
        else:
            ps = [float(t) for t in rest[i:]]
            break
    return (code, d, shots, variants, ps, seed, osd_lam,
            admm_rho, admm_alpha)


def main():
    (code, d, shots, variants, ps, seed, osd_lam, admm_rho,
     admm_alpha) = parse_args(sys.argv)
    h_x, h_z = medium_code() if code == "72" else gross_code()
    l_x, l_z = logicals(h_x, h_z)
    n = h_x.shape[1]
    for p in ps:
        w_data = np.log((1 - 2 * p / 3) / (2 * p / 3))
        w_meas = np.log((1 - p) / p)
        weights_x = np.concatenate(
            [np.full(d * n, w_data),
             np.full(d * h_z.shape[0], w_meas)]).astype(np.float32)
        weights_z = np.concatenate(
            [np.full(d * n, w_data),
             np.full(d * h_x.shape[0], w_meas)]).astype(np.float32)
        for name, cls, kw in [
                ("bp", BpOsdDecoder, {"max_iter": 30, "osd_order": 1}),
                ("admm", AdmmOsdDecoder, {"max_iter": 150, "osd_order": 1,
                                          "max_r": 4}),
                ("admm-w", AdmmOsdDecoder,
                 {"max_iter": 150, "osd_order": 1, "max_r": 4,
                  "weights_x": weights_x, "weights_z": weights_z}),
                ("admm-ldr", AdmmOsdDecoder,
                 {"max_iter": 150, "osd_order": 1, "max_r": 4,
                  "ldr": True, "ldr_outer": 4})]:
            if name not in variants:
                continue
            if osd_lam is not None:
                kw = dict(kw, osd_lam=osd_lam)
            if admm_rho is not None and name.startswith("admm"):
                kw["rho"] = admm_rho
            if admm_alpha is not None and name.startswith("admm"):
                kw["alpha"] = admm_alpha
            dec = PhenomDecoder(cls, h_x, h_z, l_x, l_z, d,
                                p_x=2 * p / 3, p_z=2 * p / 3, **kw)
            t0 = time.time()
            res = benchmark_phenom(dec, h_x, h_z, l_x, l_z, p, d, shots,
                                   seed=seed)
            out = ("[{code}] {name} p={p:5.3f} ler={ler:.5f} "
                   "ler_x={lx:.5f} ler_z={lz:.5f} invalid={ix}+{iz} "
                   "seed={seed} {secs:.1f}s").format(
                code=code, name=name, p=p, ler=res["ler"],
                lx=res["ler_x"], lz=res["ler_z"],
                ix=res["invalid_x"], iz=res["invalid_z"],
                seed=seed, secs=time.time() - t0)
            print(out)


if __name__ == "__main__":
    main()
