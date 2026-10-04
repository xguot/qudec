"""Sweep the code-capacity LER of the LP + OSD decoder on the gross code.

Parity gate for the Gu-Soleimanifar reproduction: LP+OSD-CS should sit at
or slightly below the BP+OSD curve at code capacity for the [[144,12,12]]
code. The exact LP is solved per shot with HiGHS, so keep shot counts
modest locally; the ADMM solver covers larger sweeps.

Usage:

    python -u scripts/bench_lp_gross.py [shots]
"""

import sys
import time

from qudec.bench import benchmark_iid
from qudec.codes import gross_code, logicals
from qudec.lp import LpOsdDecoder


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    h_x, h_z = gross_code()
    l_x, l_z = logicals(h_x, h_z)
    for p in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        dec = LpOsdDecoder(h_x, h_z, l_x, l_z, 2 * p / 3, 2 * p / 3)
        t0 = time.time()
        res = benchmark_iid(dec, p, shots, "depolarizing", seed=0)
        out = ("p={p:5.3f} ler={ler:.5f} ler_x={lx:.5f} ler_z={lz:.5f} "
               "invalid={ix}+{iz} {secs:.1f}s").format(
            p=p, res=res, ler=res["ler"], lx=res["ler_x"], lz=res["ler_z"],
            ix=res["invalid_x"], iz=res["invalid_z"],
            secs=time.time() - t0)
        print(out)


if __name__ == "__main__":
    main()
