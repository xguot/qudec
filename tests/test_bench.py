"""Tests for the i.i.d. benchmark loop."""

import unittest

from qudec.bench import benchmark_iid
from qudec.bposd import BpOsdDecoder
from qudec.codes import logicals, steane_code


class TestBench(unittest.TestCase):
    def test_zero_noise_perfect(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        dec = BpOsdDecoder(h_x, h_z, l_x, l_z, 0.05, 0.05)
        res = benchmark_iid(dec, 0.0, 50, "depolarizing", seed=0)
        self.assertEqual(res["ler"], 0.0)
        self.assertEqual(res["invalid_x"] + res["invalid_z"], 0)

    def test_steane_beats_identity(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        dec = BpOsdDecoder(h_x, h_z, l_x, l_z, 0.05, 0.05)
        res = benchmark_iid(dec, 0.03, 500, "x", seed=4)
        self.assertLess(res["ler"], 0.03)


if __name__ == "__main__":
    unittest.main()
