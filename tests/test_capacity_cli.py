"""Paired code-capacity CLI, bounded decoding, and optional reference gate."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_phenom, bench_capacity
from qudec.codes import logicals, steane_code
from qudec.lp import LpOsdDecoder
from qudec.stats import decode_outcomes


class TestCapacityArguments(unittest.TestCase):
    def test_defaults_and_historical_rates(self):
        args = bench_capacity.parse_args(["--output", "results/capacity.jsonl"])
        self.assertEqual(args.code, "144")
        self.assertEqual((args.shots, args.seed, args.batch_size), (500, 0, 32))
        self.assertEqual(args.ps, [.03, .05, .08, .1])
        self.assertEqual(args.variants, ["bp", "lp", "admm"])
        args = bench_capacity.parse_args([
            "--code", "72", "--variants", "bp", "--seed", "4", "--shots", "7",
            "--ps", ".005", ".01", ".02", "--osd-lam", "60",
            "--output", "results/capacity.jsonl"])
        self.assertEqual(args.ps, [.005, .01, .02])
        self.assertEqual(args.osd_lam, 60)

    def test_rejects_invalid_or_duplicate_settings(self):
        for argv in ([], ["--code", "7"], ["--shots", "0"], ["--seed", "-1"],
                     ["--batch-size", "0"], ["--ps", "nan"], ["--ps", ".02", ".02"],
                     ["--variants", "bp,bp"], ["--variants", "typo"],
                     ["--admm-rho", "nan"], ["--admm-alpha", "2"]):
            if argv:
                argv = [*argv, "--output", "unused.jsonl"]
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench_capacity.parse_args(argv)


class TestCapacityBenchmark(unittest.TestCase):
    def test_invalid_corrections_are_failures_even_without_logical_flip(self):
        h, l = np.array([[1, 1]], np.int8), np.array([[0, 1]], np.int8)

        class ZeroDecoder:
            def decode_corrections(self, sx, sz):
                return np.zeros((len(sx), 2), np.int8), np.zeros((len(sz), 2), np.int8)

        samples = (np.array([[1]], np.int8), np.array([[0]], np.int8),
                   np.array([[1, 0]], np.int8), np.array([[0, 0]], np.int8))
        result = bench_capacity.benchmark_capacity(ZeroDecoder(), (h, h, l, l), samples, 1)
        self.assertEqual(result["logical_x"], 0)
        self.assertEqual(result["invalid_x"], 1)
        self.assertEqual(result["failure"], 1)
        self.assertEqual(result["outcomes"]["failure"], "1")

    def test_nonbinary_corrections_are_invalid(self):
        h, l = np.array([[1, 1]], np.int8), np.array([[0, 1]], np.int8)

        class NonbinaryDecoder:
            def decode_corrections(self, sx, sz):
                return np.full((len(sx), 2), 2), np.zeros((len(sz), 2), np.int8)

        samples = (np.zeros((1, 1), np.int8), np.zeros((1, 1), np.int8),
                   np.zeros((1, 2), np.int8), np.zeros((1, 2), np.int8))
        result = bench_capacity.benchmark_capacity(NonbinaryDecoder(), (h, h, l, l), samples, 1)
        self.assertEqual(result["invalid_x"], 1)
        self.assertEqual(result["failure"], 1)

    def test_decode_batch_bound_and_sample_identity(self):
        h_x, h_z = steane_code()
        matrices = h_x, h_z, *logicals(h_x, h_z)
        samples = bench_capacity.sample_capacity(h_x, h_z, .1, 9, 4)
        sizes = []

        class ZeroDecoder:
            def decode_corrections(self, sx, sz):
                sizes.append(len(sx))
                return np.zeros((len(sx), 7), np.int8), np.zeros((len(sz), 7), np.int8)

        a = bench_capacity.benchmark_capacity(ZeroDecoder(), matrices, samples, 4)
        self.assertEqual(sizes, [4, 4, 1])
        b = bench_capacity.benchmark_capacity(
            ZeroDecoder(), matrices, bench_capacity.sample_capacity(h_x, h_z, .1, 9, 4), 1)
        self.assertEqual(a["sample_id"], b["sample_id"])
        self.assertEqual(a["outcomes"], b["outcomes"])


class TestCapacityCli(unittest.TestCase):
    def test_real_small_code_variants_and_isolated_worker_pairing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench_capacity, "gross_code", steane_code), \
                patch.object(bench_capacity, "load_reference_decoder") as load_reference, \
                redirect_stdout(io.StringIO()):
            together = Path(tmp) / "together.jsonl"
            isolated = Path(tmp) / "isolated.jsonl"
            common = ["--code", "144", "--shots", "7", "--seed", "5", "--ps", ".08",
                      "--bp-max-iter", "5", "--admm-max-iter", "10"]
            bench_capacity.main([*common, "--batch-size", "3", "--output", str(together)])
            bench_capacity.main([*common, "--variants", "bp", "--batch-size", "1",
                                 "--output", str(isolated)])
            load_reference.assert_not_called()
            records = [json.loads(line) for line in together.read_text().splitlines()]
            alone = json.loads(isolated.read_text())
        self.assertEqual([r["variant"] for r in records], ["bp", "lp", "admm"])
        self.assertEqual(len({r["sample_id"] for r in records}), 1)
        self.assertEqual(alone["sample_id"], records[0]["sample_id"])
        self.assertEqual(alone["outcomes"], records[0]["outcomes"])
        for record in records:
            self.assertEqual(record["protocol"], "code-capacity-v1")
            self.assertEqual(record["model"], "depolarizing")
            self.assertEqual((record["d"], record["p_meas"]), (0, 0))
            self.assertEqual(len(record["environment"]["source_sha256"]), 64)
            self.assertTrue(all(key in record["environment"] for key in
                                ("python", "numpy", "scipy", "torch")))
            self.assertEqual(record["invalid_x"] + record["invalid_z"], 0)
            self.assertEqual(int(decode_outcomes(record).sum()), record["failure"])
            self.assertEqual(record["osd_device"], "cpu")
        analysis = analyze_phenom.analyze(records)
        self.assertEqual(len(analysis), 1)
        self.assertIn("mcnemar", analysis[0]["comparisons"]["admm"])
        self.assertIn("mcnemar", analysis[0]["comparisons"]["lp"])

    def test_optional_reference_is_loaded_only_when_selected(self):
        calls = []

        class ReferenceStub:
            def __init__(self, h_x, h_z, l_x, l_z, p, max_iter):
                self.decoder = LpOsdDecoder(h_x, h_z, l_x, l_z)
                calls.append((p, max_iter))

            def decode_corrections(self, sx, sz):
                return self.decoder.decode_corrections(sx, sz)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench_capacity, "gross_code", steane_code), \
                patch.object(bench_capacity, "load_reference_decoder", return_value=ReferenceStub) as load, \
                redirect_stdout(io.StringIO()):
            output = Path(tmp) / "reference.jsonl"
            bench_capacity.main(["--variants", "reference", "--ps", ".001", "--shots", "2",
                                 "--batch-size", "1", "--output", str(output)])
            result = json.loads(output.read_text())
        load.assert_called_once_with()
        self.assertEqual(calls, [(.001, 1000)])
        self.assertEqual(result["variant"], "reference")
        self.assertEqual(result["shots"], 2)  # Calibration probes are excluded.
        self.assertEqual(result["decoder_config"]["calibration"], "fixed-seed-nonzero-syndromes")
        self.assertEqual(result["solver_device"], "cpu")
        self.assertEqual(result["invalid_x"] + result["invalid_z"], 0)


if __name__ == "__main__":
    unittest.main()
