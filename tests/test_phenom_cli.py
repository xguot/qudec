"""CLI arguments and aggregation preserve experimental comparisons."""

from copy import deepcopy
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

from scripts import analyze_phenom, bench_phenom, bench_tune


def record(variant, sample_id, failures, **changes):
    result = {"code": "72", "d": 6, "p": .02, "p_meas": .02,
              "model": "depolarizing", "protocol": "closed-v1", "projection": "exact",
              "environment": {"source_sha256": "source"},
              "decoder_config": {"osd_lam": 60}, "variant": variant,
              "sample_id": sample_id, "shots": len(failures),
              "seed": 0 if sample_id == "a" else 1,
              "failure": failures.count("1"), "outcomes": {"failure": failures}}
    result.update(changes)
    return result


class TestPhenomCli(unittest.TestCase):
    def test_options_before_and_after_rates(self):
        args = bench_phenom.parse_args(["72", "6", "100", ".01", "--seed", "4",
                                        ".02", "--variants", "bp,admm-w", "--batch-size", "7"])
        self.assertEqual(args.ps, [.01, .02])
        self.assertEqual(args.seed, 4)
        self.assertEqual(args.batch_size, 7)
        self.assertEqual(args.variants, ["bp", "admm-w"])
        args = bench_tune.parse_args(["144", "6", "300", ".02", "--rhos", "1,4",
                                      "--legacy-open-boundary", ".01"])
        self.assertEqual(args.rhos, [1, 4])
        self.assertEqual(args.ps, [.02, .01])
        self.assertTrue(args.legacy_open_boundary)

    def test_rejects_invalid_code_variant_and_batch(self):
        for argv in (["bad", "6"], ["72", "6", "--variants", "typo"],
                     ["72", "6", "--batch-size", "0"],
                     ["72", "6", "--admm-rho", "nan"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench_phenom.parse_args(argv)

    def test_duplicate_rates_are_normalized_and_duplicate_grid_rejected(self):
        self.assertEqual(bench_phenom.parse_args(["72", "6", "10", ".02", ".02"]).ps, [.02])
        self.assertEqual(bench_tune.parse_args(["72", "6", "10", ".02", ".02"]).ps, [.02])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench_tune.parse_args(["72", "6", "--rhos", "1,1", "--alphas", "1"])

    def test_default_tune_grid_is_matched_and_has_no_duplicate_configs(self):
        args = bench_tune.parse_args(["72", "1", "2"])
        self.assertEqual(args.rhos, [.5, 1., 2., 4., 8.])
        with patch.object(bench_tune, "run_variant") as run, \
                patch.object(bench_tune, "environment_metadata", return_value={}):
            bench_tune.main(["72", "1", "2"])
        self.assertEqual(run.call_count, 62)
        configurations, names = set(), set()
        for call in run.call_args_list:
            name, cls, cfg = call.args[4:7]
            key = (cls.__name__, "weights_x" in cfg, cfg.get("rho"),
                   cfg["alpha"], cfg.get("osd_lam"))
            self.assertNotIn(key, configurations)
            self.assertNotIn(name, names)
            configurations.add(key)
            names.add(name)
        for weighted in (False, True):
            for rho in args.rhos:
                for alpha in args.alphas:
                    for lam in (None, 60):
                        self.assertIn(("AdmmOsdDecoder", weighted, rho, alpha, lam), configurations)

    def test_aggregate_independent_seeds_and_pair_matching_shots(self):
        records = [record("bp", "a", "1110"), record("admm-w", "a", "0010"),
                   record("bp", "b", "0010"), record("admm-w", "b", "0000")]
        result = analyze_phenom.analyze(records)[0]
        self.assertEqual(result["variants"]["bp"]["shots"], 8)
        self.assertEqual(result["variants"]["bp"]["failures"], 4)
        paired = result["comparisons"]["admm-w"]["mcnemar"]
        self.assertEqual(paired["a_only"], 3)
        self.assertEqual(paired["b_only"], 0)

    def test_rejects_duplicate_samples_and_mixed_decoder_settings(self):
        a = record("bp", "a", "001")
        with self.assertRaises(ValueError):
            analyze_phenom.analyze([a, deepcopy(a)])
        b = record("bp", "b", "000", decoder_config={"rho": 4, "osd_lam": 60})
        with self.assertRaises(ValueError):
            analyze_phenom.analyze([a, b])

    def test_different_protocols_cannot_be_pooled(self):
        records = [record("bp", "a", "001"),
                   record("bp", "b", "001", protocol="legacy-open-v1")]
        self.assertEqual(len(analyze_phenom.analyze(records)), 2)

    def test_reused_seed_rejected_even_when_shots_and_hash_differ(self):
        records = [record("bp", "a", "00", seed=7),
                   record("bp", "b", "000", seed=7)]
        with self.assertRaises(ValueError):
            analyze_phenom.analyze(records)

    def test_independent_seeds_can_have_identical_realized_samples(self):
        records = [record("bp", "a", "00", seed=0),
                   record("bp", "a", "00", seed=1)]
        self.assertEqual(analyze_phenom.analyze(records)[0]["variants"]["bp"]["shots"], 4)

    def test_osd_orders_are_separate_experiments(self):
        records = [record("bp", "a", "001", decoder_config={"osd_order": 0}),
                   record("bp", "b", "001", decoder_config={"osd_order": 1})]
        self.assertEqual(len(analyze_phenom.analyze(records)), 2)


if __name__ == "__main__":
    unittest.main()
