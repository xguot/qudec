"""Sweep the gross-code LER with the reference bposd decoder.

Reproduce the published code-capacity logical error rate of the
[[144, 12, 12]] bivariate bicycle code using Roffe's BP + OSD
implementation (bposd), the decoder behind the paper's numbers.
Error sampling and the logical-success check reuse the in-tree
benchmark, so results are directly comparable to
scripts/bench_gross.py.

Install the optional dependencies first:

    pip install -e ".[reference]"

Usage:

    python -u scripts/bench_gross_ref.py [shots]

Decoder defaults (minimum-sum BP, 1000 iterations, OSD-CS) follow the
bposd package. Adjust max_iter / bp_method / osd_method to mirror the
paper's methods section when matching published data points.
"""

import sys

import numpy as np

from qudec.bench import benchmark_iid
from qudec.codes import gross_code, logicals


class BposdRefDecoder:
    """bposd reference decoder wrapped in the in-tree benchmark interface."""

    def __init__(self, h_x, h_z, l_x, l_z, p, max_iter=1000,
                 bp_method="minimum_sum", osd_method="osd_cs"):
        from bposd import bposd_decoder
        from bposd.css import css_code

        self.h_x = h_x
        self.h_z = h_z
        self.l_x = l_x
        self.l_z = l_z
        self.n = h_x.shape[1]
        code = css_code(hx=h_x.astype(int), hz=h_z.astype(int))
        self.bpd = bposd_decoder(
            code, error_rate=p, max_iter=max_iter, bp_method=bp_method,
            osd_method=osd_method)

    def decode_corrections(self, sx, sz):
        """Decode (batch, m) X and Z syndromes to (batch, n) corrections."""
        c_x = np.empty_like(sx)
        c_z = np.empty_like(sz)
        for i in range(sx.shape[0]):
            out = self.bpd.decode(np.concatenate([sx[i], sz[i]]))
            corr = np.asarray(out[0] if isinstance(out, tuple) else out)
            if corr.shape != (2 * self.n,):
                raise ValueError(
                    f"unexpected bposd correction shape {corr.shape}, "
                    f"expected {(2 * self.n,)} as [x-part | z-part]")
            c_x[i] = corr[: self.n]
            c_z[i] = corr[self.n:]
        return c_x, c_z


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    h_x, h_z = gross_code()
    l_x, l_z = logicals(h_x, h_z)
    for p in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        dec = BposdRefDecoder(h_x, h_z, l_x, l_z, p)
        res = benchmark_iid(dec, p, shots, "depolarizing", seed=0)
        out = ("p={p:5.3f} ler={ler:.5f} ler_x={lx:.5f} ler_z={lz:.5f} "
               "invalid={ix}+{iz} shots={shots}").format(
            p=p, ler=res["ler"], lx=res["ler_x"], lz=res["ler_z"],
            ix=res["invalid_x"], iz=res["invalid_z"], shots=shots)
        print(out)


if __name__ == "__main__":
    main()
