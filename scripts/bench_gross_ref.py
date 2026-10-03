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

The combined check matrix (css_code.h) stores the error vector as
[z-part | x-part] with syndrome [sx | sz]. The decoder runs through
the native ldpc v2 API with the bposd v1 defaults (0.625 min-sum
scaling, 10 combination-sweep orders) and per-part channel
probabilities; the wrapper calibrates the syndrome packing and
correction layout on the first shot whose X and Z syndromes are both
nonzero, then reuses that format.
"""

import sys

import numpy as np

from qudec.bench import benchmark_iid
from qudec.codes import gross_code, logicals


class BposdRefDecoder:
    """bposd reference decoder wrapped in the in-tree benchmark interface."""

    def __init__(self, h_x, h_z, l_x, l_z, p, max_iter=1000,
                 bp_method="minimum_sum", osd_method="osd_cs"):
        from bposd.css import css_code

        self.h_x = h_x
        self.h_z = h_z
        self.l_x = l_x
        self.l_z = l_z
        self.n = h_x.shape[1]
        code = css_code(hx=h_x.astype(int), hz=h_z.astype(int))
        self.decoders = self._build(code, p, max_iter,
                                    bp_method, osd_method)
        self._decoder = None
        self._pack = None
        self._split = None

    @staticmethod
    def _build(code, p, max_iter, bp_method, osd_method):
        """Build the decoder on the combined binary parity check.

        Prefer the native ldpc v2 BpOsdDecoder with per-part channel
        probabilities and the bposd v1 defaults; fall back to the
        legacy bposd shim when the native class is unavailable.
        """
        mat = getattr(code, "h", None)
        errors = []
        if mat is not None:
            try:
                from ldpc import BpOsdDecoder

                dec = BpOsdDecoder(
                    mat, error_channel=[2 * p / 3] * mat.shape[1],
                    max_iter=max_iter, bp_method=bp_method,
                    ms_scaling_factor=0.625, schedule="parallel",
                    osd_method=osd_method, osd_order=10)
                return [("h", mat.shape, dec)]
            except Exception as exc:
                errors.append(
                    f"ldpc.BpOsdDecoder: {type(exc).__name__}: {exc}")
        try:
            from bposd import bposd_decoder

            mat = mat if mat is not None else getattr(code, "hx", None)
            if mat is None:
                raise ValueError("css_code exposes no parity check matrix")
            dec = bposd_decoder(mat, error_rate=p, max_iter=max_iter,
                                bp_method=bp_method, osd_method=osd_method)
            return [("h", mat.shape, dec)]
        except Exception as exc:
            errors.append(f"bposd shim: {type(exc).__name__}: {exc}")
        raise ValueError("no decoder built: " + "; ".join(errors))

    def _valid(self, sx, sz, c_x, c_z):
        """Check that a correction reproduces the given syndrome."""
        try:
            ok_x = np.array_equal((self.h_z @ c_x.astype(int)) % 2, sx)
            ok_z = np.array_equal((self.h_x @ c_z.astype(int)) % 2, sz)
            return bool(ok_x and ok_z)
        except ValueError:
            return False

    @staticmethod
    def _pair_packing_fns():
        """Return 2-bit-per-row packings for a 2m-row decoder."""
        def make(x_pos, z_pos, order):
            def fn(a, b):
                xs = np.column_stack(
                    [a, np.zeros_like(a)] if x_pos == 0
                    else [np.zeros_like(a), a])
                zs = np.column_stack(
                    [b, np.zeros_like(b)] if z_pos == 0
                    else [np.zeros_like(b), b])
                if order == "xz":
                    seq = np.concatenate([zs, xs])
                elif order == "zx":
                    seq = np.concatenate([xs, zs])
                else:
                    seq = np.empty((2 * a.shape[0], 2), dtype=a.dtype)
                    seq[0::2], seq[1::2] = zs, xs
                return seq.ravel()
            return fn
        fns = {}
        for x_pos in (0, 1):
            for z_pos in (0, 1):
                for order in ("xz", "zx", "alt"):
                    fns[f"pairs_x{x_pos}z{z_pos}_{order}"] = make(
                        x_pos, z_pos, order)
        return fns

    @staticmethod
    def _symbol_packing_fns():
        """Return GF(4) symbol packings for an m-row decoder."""
        def make(x_mul, z_mul, order):
            def fn(a, b):
                xs = a * x_mul
                zs = b * z_mul
                if order == "xz":
                    seq = np.concatenate([zs, xs])
                elif order == "zx":
                    seq = np.concatenate([xs, zs])
                else:
                    seq = np.empty(2 * a.shape[0], dtype=a.dtype)
                    seq[0::2], seq[1::2] = zs, xs
                return seq
            return fn
        fns = {}
        for x_mul in (1, 2):
            for z_mul in (1, 2):
                for order in ("xz", "zx", "alt"):
                    fns[f"sym_x{x_mul}z{z_mul}_{order}"] = make(
                        x_mul, z_mul, order)
        return fns

    def _packing_fns(self, nq, m):
        """Return {name: fn(sx, sz)} for syndrome packings of length m."""
        fns = {}
        if m == 2 * nq:
            fns["sx_sz"] = lambda a, b: np.concatenate([a, b])
            fns["sz_sx"] = lambda a, b: np.concatenate([b, a])
            fns["alt_sx_sz"] = lambda a, b: np.ravel(np.stack([a, b], axis=1))
            fns["alt_sz_sx"] = lambda a, b: np.ravel(np.stack([b, a], axis=1))
            fns.update(self._symbol_packing_fns())
        if m == 4 * nq:
            fns.update(self._pair_packing_fns())
        return fns

    def _splits(self):
        """Yield candidate corrections layouts as (c_x, c_z) splitters."""
        n = self.n
        yield lambda c: (c[:n], c[n:])
        yield lambda c: (c[n:], c[:n])
        yield lambda c: (c[0::2], c[1::2])
        yield lambda c: (c[1::2], c[0::2])
        yield lambda c: (c & 1, c >> 1)
        yield lambda c: (c >> 1, c & 1)

    def _calibrate(self, sx, sz):
        """Detect the working decoder, syndrome packing, and split."""
        for i in range(min(sx.shape[0], 64)):
            if sx[i].sum() == 0 or sz[i].sum() == 0:
                continue
            for _name, _shape, dec in self.decoders:
                fns = self._packing_fns(sx.shape[1], _shape[0])
                for pack_name, fn in fns.items():
                    packed = fn(sx[i], sz[i])
                    try:
                        out = dec.decode(packed)
                    except Exception:
                        continue
                    corr = np.asarray(
                        out[0] if isinstance(out, tuple) else out)
                    corr = corr.ravel().astype(np.int8)
                    for split in self._splits():
                        try:
                            c_x, c_z = split(corr)
                        except (TypeError, ValueError):
                            continue
                        if self._valid(sx[i], sz[i], c_x, c_z):
                            return dec, fn, split
        shapes = [(n, s) for n, s, _ in self.decoders]
        raise ValueError(
            f"no bposd matrix/syndrome/correction layout reproduced a "
            f"valid correction; matrices tried: {shapes}")

    def decode_corrections(self, sx, sz):
        """Decode (batch, m) X and Z syndromes to (batch, n) corrections."""
        c_x = np.empty((sx.shape[0], self.n), dtype=sx.dtype)
        c_z = np.empty((sz.shape[0], self.n), dtype=sz.dtype)
        if self._decoder is None:
            self._decoder, self._pack, self._split = self._calibrate(sx, sz)
        for i in range(sx.shape[0]):
            out = self._decoder.decode(self._pack(sx[i], sz[i]))
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
