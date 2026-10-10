"""Tests for the numpy-only OSD post-processing module.

Runs without torch so it also serves as the laptop smoke test for the
post-processing layer.
"""

import unittest

import numpy as np

from qudec.codes import (
    gross_code,
    medium_code,
    repetition_code,
    steane_code,
)
from qudec.noise import sample_iid_errors
from qudec.osd import osd0, osd_cs_exhaustive, osd_cs_greedy, osd_decode


def brute_force_ml(h, s):
    """Return (correction, weight) of the minimum-weight consistent error."""
    n = h.shape[1]
    best = None
    best_w = n + 1
    for mask in range(1 << n):
        e = np.array([(mask >> j) & 1 for j in range(n)], dtype=np.int8)
        if not np.array_equal((h @ e) % 2, s):
            continue
        w = int(e.sum())
        if w < best_w:
            best_w = w
            best = e
    return best, best_w


def has_ml_representative_with_small_t_support(h, s, pivots):
    """Whether some minimum-weight consistent error has T-support <= 2."""
    n = h.shape[1]
    best_w = n + 1
    ok = False
    for mask in range(1 << n):
        e = np.array([(mask >> j) & 1 for j in range(n)], dtype=np.int8)
        if not np.array_equal((h @ e) % 2, s):
            continue
        w = int(e.sum())
        if w < best_w:
            best_w = w
            ok = False
        if w == best_w:
            t_w = int(e[np.setdiff1d(np.arange(n), pivots)].sum())
            if t_w <= 2:
                ok = True
    return ok


class TestOsd(unittest.TestCase):
    def test_zero_syndrome_zero_correction(self):
        for h_x, h_z in (repetition_code(5), steane_code(),
                         medium_code(), gross_code()):
            for h in (h_x, h_z):
                if h.shape[0] == 0:
                    continue
                n = h.shape[1]
                s = np.zeros(h.shape[0], dtype=np.int8)
                for corr in (osd0(h, s, np.arange(n)),
                             osd_cs_greedy(h, s, np.arange(n)),
                             osd_cs_exhaustive(h, s, np.arange(n), 60)):
                    self.assertTrue(np.all(corr == 0))

    def test_corrections_consistent(self):
        rng = np.random.default_rng(0)
        for h_x, h_z in (repetition_code(5), steane_code(), medium_code()):
            for h in (h_x, h_z):
                if h.shape[0] == 0:
                    continue
                n = h.shape[1]
                e_x, e_z = sample_iid_errors(n, 0.1, 64, seed=1)
                for shot in range(32):
                    e = e_x[shot] if h is h_z else e_z[shot]
                    s = ((h @ e) % 2).astype(np.int8)
                    order = rng.permutation(n)
                    for corr in (osd0(h, s, order),
                                 osd_cs_greedy(h, s, order),
                                 osd_cs_exhaustive(h, s, order, 60)):
                        self.assertTrue(np.array_equal((h @ corr) % 2, s))

    def test_exhaustive_matches_ml_repetition(self):
        h = repetition_code(5)[1]
        n = h.shape[1]
        e_x, _ = sample_iid_errors(n, 0.5, 64, model="x", seed=2)
        for e in e_x:
            s = ((h @ e) % 2).astype(np.int8)
            _, ml_w = brute_force_ml(h, s)
            corr = osd_cs_exhaustive(h, s, np.arange(n), n)
            if ml_w <= 2:
                self.assertEqual(int(corr.sum()), ml_w)

    def test_exhaustive_matches_ml_steane_weight1(self):
        h = steane_code()[0]
        n = h.shape[1]
        for j in range(n):
            e = np.zeros(n, dtype=np.int8)
            e[j] = 1
            s = ((h @ e) % 2).astype(np.int8)
            corr = osd_cs_exhaustive(h, s, np.arange(n), n)
            self.assertEqual(int(corr.sum()), 1)

    def test_dispatch_matches_components(self):
        h = steane_code()[0]
        n = h.shape[1]
        s = np.zeros(h.shape[0], dtype=np.int8)
        self.assertTrue(np.array_equal(
            osd_decode(h, s, np.arange(n), order_kind=0), osd0(h, s, np.arange(n))))
        self.assertTrue(np.array_equal(
            osd_decode(h, s, np.arange(n), order_kind=1),
            osd_cs_greedy(h, s, np.arange(n))))
        self.assertTrue(np.array_equal(
            osd_decode(h, s, np.arange(n), order_kind=1, lam=60),
            osd_cs_exhaustive(h, s, np.arange(n), 60)))


if __name__ == "__main__":
    unittest.main()
