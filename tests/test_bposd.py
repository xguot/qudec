"""Tests for the BP + OSD decoder."""

import unittest
from collections import defaultdict

import numpy as np

from qudec.bench import benchmark_iid
from qudec.bposd import BpOsdDecoder
from qudec.codes import (
    gross_code,
    logicals,
    medium_code,
    repetition_code,
    steane_code,
)
from qudec.noise import sample_iid_errors


def make_decoder(h_x, h_z, p=0.1, osd_order=1):
    l_x, l_z = logicals(h_x, h_z)
    return BpOsdDecoder(h_x, h_z, l_x, l_z, p, p, osd_order=osd_order)


class TestBpOsd(unittest.TestCase):
    def test_zero_syndrome_zero_correction(self):
        for h_x, h_z in (gross_code(), steane_code(), repetition_code(5)):
            dec = make_decoder(h_x, h_z)
            m_z, m_x = h_z.shape[0], h_x.shape[0]
            c_x, c_z = dec.decode_corrections(
                np.zeros((2, m_z), np.int8), np.zeros((2, m_x), np.int8))
            self.assertTrue(np.all(c_x == 0))
            self.assertTrue(np.all(c_z == 0))

    def test_correction_validity(self):
        h_x, h_z = steane_code()
        dec = make_decoder(h_x, h_z, p=0.1)
        n = h_x.shape[1]
        e_x, e_z = sample_iid_errors(n, 0.1, 50, "depolarizing", seed=1)
        sx = (h_z @ e_x.T) % 2
        sz = (h_x @ e_z.T) % 2
        c_x, c_z = dec.decode_corrections(
            sx.T.astype(np.int8), sz.T.astype(np.int8))
        self.assertTrue(np.all((h_z @ c_x.T) % 2 == sx))
        self.assertTrue(np.all((h_x @ c_z.T) % 2 == sz))

    def test_osd_cs_not_worse_than_osd0(self):
        h_x, h_z = medium_code()
        dec0 = make_decoder(h_x, h_z, p=0.05, osd_order=0)
        dec1 = make_decoder(h_x, h_z, p=0.05, osd_order=1)
        n = h_x.shape[1]
        e_x, _ = sample_iid_errors(n, 0.05, 100, "x", seed=2)
        sx = (h_z @ e_x.T) % 2
        empty = np.zeros((100, h_x.shape[0]), np.int8)
        c0, _ = dec0.decode_corrections(sx.T.astype(np.int8), empty)
        c1, _ = dec1.decode_corrections(sx.T.astype(np.int8), empty)
        for i in range(100):
            self.assertLessEqual(int(c1[i].sum()), int(c0[i].sum()))

    def test_repetition_code_matches_mld(self):
        h_x, h_z = repetition_code(5)
        dec = make_decoder(h_x, h_z, p=0.05)
        res = benchmark_iid(dec, 0.05, 2000, "x", seed=11)
        expected = (10 * 0.05**3 * 0.95**2 + 5 * 0.05**4 * 0.95 + 0.05**5)
        self.assertAlmostEqual(res["ler"], expected, delta=0.003)
        self.assertEqual(res["invalid_x"], 0)

    def test_steane_close_to_mld(self):
        h_x, h_z = steane_code()
        _, l_z = logicals(h_x, h_z)
        dec = make_decoder(h_x, h_z, p=0.4)
        n = 7
        patterns = np.array(
            [[(i >> j) & 1 for j in range(n)] for i in range(1 << n)],
            dtype=np.int8)
        syn = (h_z @ patterns.T) % 2
        best = {}
        for i, e in enumerate(patterns):
            key = syn[:, i].tobytes()
            if key not in best or int(e.sum()) < best[key][0]:
                best[key] = (int(e.sum()), e)
        mld_fail = 0
        dec_fail = 0
        for i, e in enumerate(patterns):
            key = syn[:, i].tobytes()
            e_hat = best[key][1]
            if int(((l_z @ (e ^ e_hat)) % 2).item()):
                mld_fail += 1
            c, _ = dec.decode(syn[:, i], np.zeros(3, np.int8))
            if int(((l_z @ (e ^ c)) % 2).item()):
                dec_fail += 1
        self.assertLessEqual(dec_fail, mld_fail + 8)


if __name__ == "__main__":
    unittest.main()
