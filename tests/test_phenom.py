"""Tests for the phenomenological noise module and decoder wrapper."""

import unittest

import numpy as np

from qudec.codes import logicals, steane_code
from qudec.lp import LpOsdDecoder
from qudec.phenom import (
    PhenomDecoder,
    benchmark_phenom,
    build_phenom_matrix,
    sample_phenom,
)


class TestPhenomMatrix(unittest.TestCase):
    def test_block_structure(self):
        h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        d = 2
        out = build_phenom_matrix(h, d)
        self.assertEqual(out.shape, (4, 6 + 4))
        self.assertTrue(np.array_equal(out[0:2, 0:3], h))
        self.assertTrue(np.array_equal(out[2:4, 3:6], h))
        self.assertTrue(np.array_equal(out[0:2, 6:8], np.eye(2, dtype=np.int8)))
        self.assertTrue(np.array_equal(out[2:4, 6:8], np.eye(2, dtype=np.int8)))
        self.assertTrue(np.array_equal(out[2:4, 8:10], np.eye(2, dtype=np.int8)))
        self.assertTrue(np.array_equal(out[0:2, 8:10], np.zeros((2, 2), np.int8)))


def make_decoder(h_x, h_z, d):
    l_x, l_z = logicals(h_x, h_z)
    return PhenomDecoder(LpOsdDecoder, h_x, h_z, l_x, l_z, d, osd_order=1)


class TestPhenom(unittest.TestCase):
    def test_zero_noise_zero_correction(self):
        h_x, h_z = steane_code()
        dec = make_decoder(h_x, h_z, 2)
        res = benchmark_phenom(dec, h_x, h_z, *logicals(h_x, h_z),
                               0.0, 2, 20, seed=3)
        self.assertEqual(res["ler"], 0.0)
        self.assertEqual(res["invalid_x"], 0)
        self.assertEqual(res["invalid_z"], 0)

    def test_correction_validity(self):
        h_x, h_z = steane_code()
        dec = make_decoder(h_x, h_z, 2)
        l_x, l_z = logicals(h_x, h_z)
        res = benchmark_phenom(dec, h_x, h_z, l_x, l_z, 0.05, 2, 40, seed=4)
        self.assertEqual(res["invalid_x"], 0)
        self.assertEqual(res["invalid_z"], 0)

    def test_weight_one_round_one_corrected(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        dec = make_decoder(h_x, h_z, 1)
        n = 7
        e = np.zeros(n, np.int8)
        e[2] = 1
        sx = (h_z @ e) % 2
        c_x, _ = dec.decode_corrections(
            sx.reshape(1, -1), np.zeros((1, 3), np.int8))
        self.assertEqual(int(((l_z @ (e ^ c_x[0][:n])) % 2).item()), 0)

    def test_measurement_error_matches_system(self):
        # a unit-vector detector syndrome decodes to a full correction
        # consistent with the time-expanded system
        h_x, h_z = steane_code()
        dec = make_decoder(h_x, h_z, 1)
        sx = np.array([[0, 1, 0]], dtype=np.int8)
        c_x, _ = dec.decode_corrections(sx, np.zeros((1, 3), np.int8))
        self.assertTrue(np.all(((dec.h_z @ c_x[0].T) % 2) == sx[0]))


if __name__ == "__main__":
    unittest.main()
