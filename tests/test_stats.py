"""Counts and paired discordances must support reproducible conclusions."""

import unittest

from qudec.stats import decode_outcomes, mcnemar, two_proportion


class TestStats(unittest.TestCase):
    def test_default_admm_gap_is_not_3p9_sigma_under_pooled_test(self):
        result = two_proportion(236, 2000, 276, 2000)
        self.assertAlmostEqual(result["z"], 1.893070415, places=8)
        self.assertGreater(result["p_value"], .05)
        self.assertAlmostEqual(result["delta"], .02)

    def test_tuned_gap(self):
        result = two_proportion(1144, 10000, 1020, 10000)
        self.assertAlmostEqual(result["z"], -2.82266490, places=7)
        self.assertLess(result["p_value"], .005)

    def test_exact_mcnemar_uses_discordant_shots(self):
        result = mcnemar([1, 1, 1, 1, 0], [0, 0, 0, 1, 0])
        self.assertEqual(result["a_only"], 3)
        self.assertEqual(result["b_only"], 0)
        self.assertEqual(result["p_value"], .25)
        self.assertLess(result["z"], 0)
        self.assertEqual(mcnemar([0], [0])["p_value"], 1)

    def test_validate_record(self):
        record = {"shots": 4, "failure": 2, "outcomes": {"failure": "0101"}}
        self.assertEqual(decode_outcomes(record).tolist(), [False, True, False, True])
        record["failure"] = 3
        with self.assertRaises(ValueError):
            decode_outcomes(record)

    def test_invalid_counts_and_pairing(self):
        with self.assertRaises(ValueError):
            two_proportion(2, 1, 0, 1)
        with self.assertRaises(ValueError):
            mcnemar([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
