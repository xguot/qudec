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
    """bposd reference decoder wrapped in the in-tree benchmark interface.

    The syndrome packing and correction layout of the installed bposd
    version are detected on the first shot that decodes to a valid
    correction, then reused for the rest of the batch.
    """

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
        self.bpd = self._build(code, bposd_decoder, p, max_iter,
                               bp_method, osd_method)
        self._pack = None
        self._split = None

    @staticmethod
    def _build(code, cls, p, max_iter, bp_method, osd_method):
        """Construct the decoder, trying the matrix attributes of css_code."""
        errors = []
        for attr in ("hx", "h"):
            mat = getattr(code, attr, None)
            if mat is None:
                continue
            try:
                return cls(mat, error_rate=p, max_iter=max_iter,
                           bp_method=bp_method, osd_method=osd_method)
            except Exception as exc:
                errors.append(f"{attr}: {type(exc).__name__}: {exc}")
        raise ValueError(
            "no css_code matrix built a bposd_decoder: " + "; ".join(errors))

    def _valid(self, sx, sz, c_x, c_z):
        """Check that a correction reproduces the given syndrome."""
        try:
            ok_x = np.array_equal((self.h_z @ c_x.astype(int)) % 2, sx)
            ok_z = np.array_equal((self.h_x @ c_z.astype(int)) % 2, sz)
            return bool(ok_x and ok_z)
        except ValueError:
            return False

    def _calibrate(self, sx, sz):
        """Detect the bposd syndrome packing and correction layout."""
        packs = [
            lambda a, b: np.concatenate([a, b]),
            lambda a, b: np.concatenate([b, a]),
        ]
        if sx.shape[1] == sz.shape[1]:
            packs += [
                lambda a, b: np.ravel(np.stack([a, b], axis=1)),
                lambda a, b: np.ravel(np.stack([b, a], axis=1)),
            ]
        splits = [
            lambda c: (c[: self.n], c[self.n:]),
            lambda c: (c[self.n:], c[: self.n]),
            lambda c: (c & 1, c >> 1),
            lambda c: (c >> 1, c & 1),
        ]
        for i in range(min(sx.shape[0], 64)):
            for pack in packs:
                out = self.bpd.decode(pack(sx[i], sz[i]))
                corr = np.asarray(out[0] if isinstance(out, tuple) else out)
                corr = corr.ravel().astype(np.int8)
                for split in splits:
                    try:
                        c_x, c_z = split(corr)
                    except (TypeError, ValueError):
                        continue
                    if self._valid(sx[i], sz[i], c_x, c_z):
                        return pack, split
        raise ValueError(
            "could not infer the bposd syndrome/correction format; "
            "decode one zero syndrome and report the output shape")

    def decode_corrections(self, sx, sz):
        """Decode (batch, m) X and Z syndromes to (batch, n) corrections."""
        c_x = np.empty_like(sx)
        c_z = np.empty_like(sz)
        if self._pack is None:
            self._pack, self._split = self._calibrate(sx, sz)
        for i in range(sx.shape[0]):
            out = self.bpd.decode(self._pack(sx[i], sz[i]))
            corr = np.asarray(out[0] if isinstance(out, tuple) else out)
            c_x[i], c_z[i] = self._split(corr.ravel().astype(np.int8))
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
