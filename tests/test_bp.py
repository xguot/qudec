"""Message-level BP regressions independent of OSD correction validity."""

import unittest

import numpy as np

from qudec.bp import channel_llrs, min_sum_bp, probability_vector
from qudec.circuit_decoders import AdmmOsdDemDecoder, BpOsdDemDecoder


def reference_min_sum(h, syndromes, llr, iterations, alpha):
    """Small direct edge-exclusion implementation for the test oracle."""
    out = []
    for syndrome in syndromes:
        messages = np.zeros(h.shape, dtype=np.float64)
        incoming = np.broadcast_to(llr, h.shape).copy()
        for _ in range(iterations):
            for j, row in enumerate(h):
                neighbors = np.flatnonzero(row)
                for i in neighbors:
                    others = neighbors[neighbors != i]
                    magnitude = min(1e4, np.abs(incoming[j, others]).min()) \
                        if len(others) else 1e4
                    parity = int(syndrome[j]) + int((incoming[j, others] < 0).sum())
                    messages[j, i] = alpha * (1 - 2 * (parity % 2)) * magnitude
            posterior = llr + messages.sum(axis=0)
            incoming = posterior[None, :] - messages
        out.append(llr + messages.sum(axis=0))
    return np.asarray(out)


class TestMinSumMessages(unittest.TestCase):
    def test_equal_minima_are_preserved(self):
        h = np.ones((1, 3), dtype=np.int8)
        for llr, expected in (([2, 2, 2], [4, 4, 4]),
                              ([2, 2, 4], [4, 4, 6])):
            actual = min_sum_bp(h, np.array([[0]]), llr, max_iter=1, alpha=1)
            np.testing.assert_allclose(actual.numpy(), [expected])
        actual = min_sum_bp(h, np.array([[0]]), [2, 2, 2], max_iter=1,
                            alpha=None)
        np.testing.assert_allclose(actual.numpy(), [[3, 3, 3]])

    def test_mixed_signs_and_syndromes_match_edge_exclusion(self):
        h = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]], np.int8)
        syndromes = np.array([[0, 1, 0], [1, 0, 1]], np.int8)
        llr = np.array([-2.0, 2.0, 4.0, -2.0])
        expected = reference_min_sum(h, syndromes, llr, 3, 0.9)
        actual = min_sum_bp(h, syndromes, llr, max_iter=3, alpha=0.9)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-3)

    def test_degree_one_check_and_isolated_variable(self):
        h = np.array([[1, 0]], np.int8)
        s = np.array([[0], [1]], np.int8)
        actual = min_sum_bp(h, s, [2.0, -3.0], max_iter=1, alpha=1).numpy()
        np.testing.assert_array_equal(actual, [[10002, -3], [-9998, -3]])
        single = min_sum_bp(np.ones((1, 1)), s, [2.0], 1, 1).numpy()
        np.testing.assert_array_equal(single, [[10002], [-9998]])

    def test_empty_checks_and_zero_iterations(self):
        h = np.array([[0, 0], [1, 1]], np.int8)
        s = np.array([[0, 0], [0, 1]], np.int8)
        actual = min_sum_bp(h, s, [2, 2], 1, 1).numpy()
        np.testing.assert_array_equal(actual, [[4, 4], [0, 0]])
        prior = min_sum_bp(h, s, [2, -3], max_iter=0).numpy()
        np.testing.assert_array_equal(prior, [[2, -3], [2, -3]])
        with self.assertRaisesRegex(ValueError, "empty check"):
            min_sum_bp(h, np.array([[1, 0]]), [2, 2])

    def test_empty_dimensions(self):
        actual = min_sum_bp(np.zeros((0, 2)), np.zeros((3, 0)), [2, -3]).numpy()
        np.testing.assert_array_equal(actual, [[2, -3]] * 3)
        actual = min_sum_bp(np.zeros((2, 0)), np.zeros((3, 2)), [])
        self.assertEqual(tuple(actual.shape), (3, 0))
        actual = min_sum_bp(np.ones((1, 2)), np.zeros((0, 1)), [2, 3])
        self.assertEqual(tuple(actual.shape), (0, 2))

    def test_input_validation(self):
        for iterations in (-1, 1.5):
            with self.assertRaisesRegex(ValueError, "max_iter"):
                min_sum_bp(np.ones((1, 2)), np.zeros((1, 1)), [2, 3], iterations)
        with self.assertRaisesRegex(ValueError, "llr"):
            min_sum_bp(np.ones((1, 2)), np.zeros((1, 1)), [2])
        with self.assertRaisesRegex(ValueError, "syndrome"):
            min_sum_bp(np.ones((1, 2)), np.zeros((1, 2)), [2, 3])


class TestChannelPriors(unittest.TestCase):
    def test_scalar_and_vector_llrs(self):
        np.testing.assert_allclose(channel_llrs(.1, 2), np.log([9, 9]))
        np.testing.assert_allclose(channel_llrs([.1, .25], 2), np.log([9, 3]))
        self.assertEqual(probability_vector([], 0).shape, (0,))

    def test_invalid_priors(self):
        for p in (0, 1, -0.1, float("nan"), float("inf"), [.1], [[.1, .2]]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                probability_vector(p, 2)

    def test_dem_bp_uses_mechanism_priors(self):
        h = np.array([[1, 1]], np.int8)
        l = np.array([[1, 0]], np.int8)
        shots = np.array([[1]], np.int8)
        dec = BpOsdDemDecoder(h, l, probabilities=[.01, .25], max_iter=0,
                             osd_order=0)
        np.testing.assert_array_equal(dec.decode_batch(shots), [[0]])
        dec = BpOsdDemDecoder(h, l, probabilities=[.25, .01], max_iter=0,
                             osd_order=0)
        np.testing.assert_array_equal(dec.decode_batch(shots), [[1]])

    def test_dem_admm_uses_mechanism_priors(self):
        h = np.array([[1, 1]], np.int8)
        l = np.array([[1, 0]], np.int8)
        dec = AdmmOsdDemDecoder(h, l, probabilities=[.01, .25], max_iter=100,
                               osd_order=0)
        np.testing.assert_allclose(dec.weights, channel_llrs([.01, .25], 2))
        np.testing.assert_array_equal(dec.decode_batch(np.array([[1]])), [[0]])
        plain = AdmmOsdDemDecoder(h, l, probabilities=[.01, .25],
                                  weighted=False, max_iter=50, osd_order=0)
        self.assertFalse(plain.weighted)
        self.assertIsNone(plain.weights)
        ldr = AdmmOsdDemDecoder(h, l, probabilities=[.01, .25],
                                weighted=True, ldr=True, ldr_outer=2,
                                max_iter=50, osd_order=0)
        self.assertTrue(ldr.ldr)
        np.testing.assert_array_equal(ldr.decode_batch(np.array([[1]])), [[0]])
        with self.assertRaisesRegex(ValueError, "probabilities"):
            AdmmOsdDemDecoder(h, l, weighted=True)

    def test_empty_dem_and_batches(self):
        for cls in (BpOsdDemDecoder, AdmmOsdDemDecoder):
            dec = cls(np.zeros((2, 0)), np.zeros((1, 0)), probabilities=[])
            np.testing.assert_array_equal(dec.decode_batch(np.zeros((3, 2))),
                                          np.zeros((3, 1), np.uint8))
            with self.assertRaisesRegex(ValueError, "empty check"):
                dec.decode_batch(np.array([[0, 1]]))
            dec = cls(np.ones((1, 2)), np.ones((1, 2)))
            self.assertEqual(dec.decode_batch(np.zeros((0, 1))).shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
