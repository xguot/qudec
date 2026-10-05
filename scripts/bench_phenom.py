"""Sweep the phenomenological-noise LER of BP+OSD vs ADMM+OSD.

The detector syndromes come from d rounds of data plus measurement
errors; decoding runs on the time-expanded check matrices. This is the
first LP-family decoding result beyond code capacity for the bivariate
bicycle codes.

Usage:

    python -u scripts/bench_phenom.py <72|144> <rounds> [shots]
"""

import sys
import time

import numpy as np

from qudec.admm import AdmmOsdDecoder
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals, medium_code
from qudec.phenom import PhenomDecoder, benchmark_phenom


def main():
    code = sys.argv[1]
    d = int(sys.argv[2])
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    rest = sys.argv[4:]
    variants = ["bp", "admm", "admm-w", "admm-ldr"]
    ps = [0.005, 0.01, 0.02, 0.03]
    if rest and rest[0] == "--variants":
        variants = rest[1].split(",")
        rest = rest[2:]
    if rest:
        ps = [float(x) for x in rest]
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
            dec = PhenomDecoder(cls, h_x, h_z, l_x, l_z, d,
                                p_x=2 * p / 3, p_z=2 * p / 3, **kw)
            t0 = time.time()
            res = benchmark_phenom(dec, h_x, h_z, l_x, l_z, p, d, shots,
                                   seed=0)
            out = ("[{code}] {name} p={p:5.3f} ler={ler:.5f} "
                   "ler_x={lx:.5f} ler_z={lz:.5f} invalid={ix}+{iz} "
                   "{secs:.1f}s").format(
                code=code, name=name, p=p, ler=res["ler"],
                lx=res["ler_x"], lz=res["ler_z"],
                ix=res["invalid_x"], iz=res["invalid_z"],
                secs=time.time() - t0)
            print(out)


if __name__ == "__main__":
    main()
