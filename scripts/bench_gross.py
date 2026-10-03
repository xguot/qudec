"""Sweep the code-capacity LER of the gross code over the noise rate.

Usage:

    python -u scripts/bench_gross.py [shots]
"""

import sys
import time

from qudec.bench import benchmark_iid
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    h_x, h_z = gross_code()
    l_x, l_z = logicals(h_x, h_z)
    for p in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        dec = BpOsdDecoder(h_x, h_z, l_x, l_z, 2 * p / 3, 2 * p / 3)
        t0 = time.time()
        res = benchmark_iid(dec, p, shots, "depolarizing", seed=0)
        out = ("p={p:5.3f} ler={ler:.5f} ler_x={lx:.5f} ler_z={lz:.5f} "
               "invalid={ix}+{iz} {secs:.1f}s").format(
            p=p, ler=res["ler"], lx=res["ler_x"], lz=res["ler_z"],
            ix=res["invalid_x"], iz=res["invalid_z"],
            secs=time.time() - t0)
        print(out)


if __name__ == "__main__":
    main()
