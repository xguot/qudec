"""Closed-memory boundaries, logical observables and bounded benchmarking."""

import unittest

import numpy as np

from qudec.bposd import BpOsdDecoder
from qudec.codes import logicals, repetition_code, steane_code
from qudec.lp import LpOsdDecoder
from qudec.phenom import (PhenomDecoder, benchmark_phenom, build_phenom_matrix,
                          phenom_priors, sample_phenom)


class TestPhenomMatrix(unittest.TestCase):
    def test_block_structure_and_terminal_measurement(self):
        h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        out = build_phenom_matrix(h, 2)
        self.assertEqual(out.shape, (6, 10))
        np.testing.assert_array_equal(out[:2, :3], h)
        np.testing.assert_array_equal(out[2:4, 3:6], h)
        np.testing.assert_array_equal(out[:2, 6:8], np.eye(2))
        np.testing.assert_array_equal(out[2:4, 6:8], np.eye(2))
        np.testing.assert_array_equal(out[2:4, 8:10], np.eye(2))
        np.testing.assert_array_equal(out[4:, :8], 0)
        np.testing.assert_array_equal(out[4:, 8:], np.eye(2))
        np.testing.assert_array_equal(build_phenom_matrix(h, 2, False), out[:4])

    def test_closed_detector_parity_is_final_data_syndrome(self):
        hx, hz = steane_code()
        sx, sz, ex, ez = sample_phenom(hx, hz, .15, 3, 40, seed=1)
        np.testing.assert_array_equal(sx.reshape(40, 4, 3).sum(1) % 2,
                                      (hz @ ex.T).T % 2)
        np.testing.assert_array_equal(sz.reshape(40, 4, 3).sum(1) % 2,
                                      (hx @ ez.T).T % 2)
        legacy = sample_phenom(hx, hz, .15, 3, 40, seed=1, terminal_perfect=False)
        np.testing.assert_array_equal(legacy[0], sx[:, :9])
        np.testing.assert_array_equal(legacy[1], sz[:, :9])
        np.testing.assert_array_equal(legacy[2], ex)

    def test_bp_priors_distinguish_measurement_columns(self):
        hx = np.zeros((2, 7), np.int8)
        hz = np.zeros((3, 7), np.int8)
        px, pz = phenom_priors(hx, hz, 2, .03, .07)
        np.testing.assert_allclose(px[:14], .02)
        np.testing.assert_allclose(px[14:], .07)
        self.assertEqual(px.size, 20)
        self.assertEqual(pz.size, 18)


def make_decoder(h_x, h_z, d, **kwargs):
    return PhenomDecoder(LpOsdDecoder, h_x, h_z, *logicals(h_x, h_z), d,
                          osd_order=1, **kwargs)


class TestPhenom(unittest.TestCase):
    def test_zero_noise_zero_correction(self):
        hx, hz = steane_code()
        res = benchmark_phenom(make_decoder(hx, hz, 2), hx, hz, *logicals(hx, hz),
                               0, 2, 20, seed=3)
        self.assertEqual(res["failure"], 0)
        self.assertEqual(res["residual_x"], 0)
        self.assertEqual(res["invalid_x"], 0)
        self.assertEqual(res["invalid_z"], 0)

    def test_correction_validity(self):
        hx, hz = steane_code()
        res = benchmark_phenom(make_decoder(hx, hz, 2), hx, hz, *logicals(hx, hz),
                               .05, 2, 40, seed=4, batch_size=7)
        for key in ("invalid_x", "invalid_z", "residual_x", "residual_z"):
            self.assertEqual(res[key], 0, key)

    def test_weight_one_round_one_corrected_and_decode_batch(self):
        hx, hz = steane_code()
        lx, lz = logicals(hx, hz)
        dec = make_decoder(hx, hz, 1)
        e = np.zeros(7, np.int8)
        e[2] = 1
        sx = np.concatenate([(hz @ e) % 2, np.zeros(3, np.int8)])[None, :]
        sz = np.zeros((1, 6), np.int8)
        cx, cz = dec.decode_corrections(sx, sz)
        self.assertEqual(int(((lz @ (e ^ cx[0, :7])) % 2).item()), 0)
        obs = dec.decode_batch(np.concatenate([sx, sz], axis=1))
        np.testing.assert_array_equal(obs, np.concatenate(
            [(dec.l_z @ cx.T).T % 2, (dec.l_x @ cz.T).T % 2], axis=1))
        self.assertEqual(dec.l_z.shape[1], dec.h_z.shape[1])

    def test_measurement_error_matches_system(self):
        hx, hz = steane_code()
        dec = make_decoder(hx, hz, 1)
        sx = np.array([[0, 1, 0, 0, 1, 0]], np.int8)
        cx, _ = dec.decode_corrections(sx, np.zeros((1, 6), np.int8))
        np.testing.assert_array_equal((dec.h_z @ cx[0]) % 2, sx[0])

    def test_batch_size_preserves_samples_and_outcomes(self):
        hx, hz = steane_code()
        matrices = hx, hz, *logicals(hx, hz)
        dec = make_decoder(hx, hz, 1)
        samples = sample_phenom(hx, hz, .1, 1, 17, seed=5)
        a = benchmark_phenom(dec, *matrices, .1, 1, 17, samples=samples,
                             batch_size=3, return_failures=True)
        b = benchmark_phenom(dec, *matrices, .1, 1, 17, samples=samples,
                             batch_size=17, return_failures=True)
        self.assertEqual(a["sample_id"], b["sample_id"])
        self.assertEqual(a["outcomes"], b["outcomes"])
        self.assertEqual(a["failure"], b["failure"])

    def test_closed_memory_counts_residual_syndrome(self):
        class ZeroDecoder:
            def __init__(self, hx, hz, *unused):
                self.h_x, self.h_z = hx, hz

            def decode_corrections(self, sx, sz):
                return (np.zeros((len(sx), self.h_z.shape[1]), np.int8),
                        np.zeros((len(sz), self.h_x.shape[1]), np.int8))

        h = np.array([[1, 1]], np.int8)
        l = np.array([[0, 1]], np.int8)
        ex, ez = np.array([[1, 0]], np.int8), np.zeros((1, 2), np.int8)
        for closed in (True, False):
            dec = PhenomDecoder(ZeroDecoder, h, h, l, l, 1, terminal_perfect=closed)
            sx = np.array([[1, 0]] if closed else [[1]], np.int8)
            sz = np.zeros_like(sx)
            res = benchmark_phenom(dec, h, h, l, l, .1, 1, 1,
                                   samples=(sx, sz, ex, ez))
            self.assertEqual(res["logical_x"], 0)
            self.assertEqual(res["residual_x"], 1)
            self.assertEqual(res["failure"], int(closed))

    def test_asymmetric_repetition_bp_memory(self):
        hx, hz = repetition_code(3)
        lx, lz = logicals(hx, hz)
        px, pz = phenom_priors(hx, hz, 2, .03, .07)
        dec = PhenomDecoder(BpOsdDecoder, hx, hz, lx, lz, 2, p_x=px, p_z=pz)
        self.assertEqual(dec.h_x.shape, (0, 6))
        self.assertEqual(dec.h_z.shape, (6, 10))
        self.assertEqual(dec.l_x.shape, (1, 6))
        self.assertEqual(dec.l_z.shape, (1, 10))
        np.testing.assert_array_equal(dec.base.p_x, px)
        np.testing.assert_array_equal(dec.base.p_z, pz)
        # A single X data fault, closed by a noiseless terminal syndrome.
        ex = np.array([[1, 0, 0]], np.int8)
        sx = np.concatenate([(hz @ ex.T).T, np.zeros((1, 4), np.int8)], axis=1)
        sz = np.zeros((1, 0), np.int8)
        samples = sx, sz, ex, np.zeros_like(ex)
        result = benchmark_phenom(dec, hx, hz, lx, lz, .03, 2, 1, samples=samples)
        self.assertEqual(result["failure"], 0)
        self.assertEqual(result["residual_x"], 0)
        expected = np.concatenate([(lz @ ex.T).T, np.zeros((1, 1), np.int8)], axis=1)
        np.testing.assert_array_equal(dec.decode_batch(np.concatenate([sx, sz], axis=1)), expected)


if __name__ == "__main__":
    unittest.main()
