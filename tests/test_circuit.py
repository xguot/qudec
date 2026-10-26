"""Regression coverage for circuit construction, DEM extraction, and CLI."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qudec.circuit import (
    build_memory_circuit, dem_check_matrix, detector_count, sample_memory,
)
from qudec.codes import logicals, medium_code, steane_code
from scripts import analyze_phenom

try:
    import stim
except ImportError:
    stim = None

_spec = importlib.util.spec_from_file_location(
    "bench_circuit", Path(__file__).resolve().parents[1] / "scripts/bench_circuit.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


@unittest.skipIf(stim is None, "requires optional stim dependency")
class TestMemoryCircuit(unittest.TestCase):
    def test_both_bases_noiseless_detectors_and_separate_logicals(self):
        for code in (steane_code, medium_code):
            h_x, h_z = code()
            l_x, l_z = logicals(h_x, h_z)
            for basis in ("Z", "X"):
                for rounds in (1, 3):
                    with self.subTest(code=code.__name__, basis=basis, d=rounds):
                        circuit = build_memory_circuit(
                            h_x, h_z, l_x, l_z, rounds, 0, basis)
                        self.assertEqual(circuit.num_detectors,
                                         detector_count(h_x, h_z, rounds, basis))
                        self.assertEqual(circuit.num_observables, len(l_x))
                        dets, obs = sample_memory(circuit, 16, seed=5)
                        self.assertEqual(dets.shape, (16, circuit.num_detectors))
                        self.assertEqual(obs.shape, (16, len(l_x)))
                        self.assertFalse(dets.any())
                        self.assertFalse(obs.any())
                        h, l, ps = dem_check_matrix(
                            circuit.detector_error_model(), True)
                        self.assertEqual(h.shape, (circuit.num_detectors, 0))
                        self.assertEqual(l.shape, (len(l_x), 0))
                        self.assertEqual(ps.shape, (0,))

    def test_measurements_flip_records_at_requested_probability(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        for basis in ("Z", "X"):
            circuit = build_memory_circuit(h_x, h_z, l_x, l_z, 2, 0.02, basis)
            measurements = [op for op in circuit if op.name == "M"]
            self.assertTrue(measurements)
            self.assertTrue(all(op.gate_args_copy() == [0.02]
                                for op in measurements))
            dets, obs = sample_memory(circuit, 128, seed=3)
            self.assertTrue(dets.any())
            self.assertTrue(obs.any())

    def test_bb_hyperedge_dem_extracts_without_graph_decomposition(self):
        h_x, h_z = medium_code()
        l_x, l_z = logicals(h_x, h_z)
        for basis in ("Z", "X"):
            circuit = build_memory_circuit(h_x, h_z, l_x, l_z, 2, 0.001, basis)
            h, l, probabilities = dem_check_matrix(
                circuit.detector_error_model(), True)
            self.assertEqual(h.shape[0], circuit.num_detectors)
            self.assertEqual(l.shape[0], 12)
            self.assertEqual(h.shape[1], len(probabilities))
            self.assertTrue(np.all((probabilities > 0) & (probabilities < 0.5)))
            self.assertTrue(np.any(h.sum(axis=0) > 2))

    def test_invalid_basis_rounds_or_probability(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        for rounds, p, basis in [(0, 0.1, "Z"), (1, -0.1, "Z"),
                                  (1, float("nan"), "Z"), (1, 0.1, "Y")]:
            with self.subTest(d=rounds, p=p, basis=basis):
                with self.assertRaises(ValueError):
                    build_memory_circuit(h_x, h_z, l_x, l_z, rounds, p, basis)


@unittest.skipIf(stim is None, "requires optional stim dependency")
class TestDemExtraction(unittest.TestCase):
    def test_component_cancellations_and_probabilities(self):
        dem = stim.DetectorErrorModel(
            "error(0.1) D0 L0 ^ D1 L0\n"
            "error(0.25) D0 D2 ^ D0 L1\n")
        h, l, probabilities = dem_check_matrix(dem, return_probabilities=True)
        np.testing.assert_array_equal(h, [[1, 0], [1, 0], [0, 1]])
        np.testing.assert_array_equal(l, [[0, 0], [0, 1]])
        np.testing.assert_allclose(probabilities, [0.1, 0.25])
        h_legacy, l_legacy = dem_check_matrix(dem)
        np.testing.assert_array_equal(h_legacy, h)
        np.testing.assert_array_equal(l_legacy, l)

    def test_flattened_repeat_offsets(self):
        dem = stim.DetectorErrorModel(
            "repeat 2 {\nerror(0.1) D0 L0\nshift_detectors 1\n}\n")
        h, l, probabilities = dem_check_matrix(dem, True)
        np.testing.assert_array_equal(h, np.eye(2, dtype=np.int8))
        np.testing.assert_array_equal(l, [[1, 1]])
        np.testing.assert_allclose(probabilities, [0.1, 0.1])


class TestCircuitBenchmark(unittest.TestCase):
    def test_options_do_not_become_probabilities(self):
        args = bench.parse_args([
            "bench", "72", "3", "10", "0.01", "0.02", "--basis", "X",
            "--variants", "admm,admm-u,admm-w,admm-ldr", "--seed", "9",
            "--osd-lam", "60", "--batch-size", "2", "--rhos", "0.5,2.0",
            "--alphas", "1.0,1.5"])
        self.assertEqual(args.ps, [0.01, 0.02])
        self.assertEqual(args.basis, "X")
        self.assertEqual(args.variants, ["admm", "admm-u", "admm-w", "admm-ldr"])
        self.assertEqual((args.seed, args.osd_lam, args.batch_size), (9, 60, 2))
        self.assertEqual(args.rhos, [0.5, 2.0])
        self.assertEqual(args.alphas, [1.0, 1.5])

    def test_invalid_arguments_rejected(self):
        for argv in [
                ["bench", "wrong", "2", "10", ".01"],
                ["bench", "72", "0", "10", ".01"],
                ["bench", "72", "2", "0", ".01"],
                ["bench", "72", "2", "10", "nan"],
                ["bench", "72", "2", "10", ".01", "--variants", "typo"],
                ["bench", "72", "2", "10", ".01", "--batch-size", "0"],
                ["bench", "72", "2", "10", ".01", "--rhos", "1,1"],
                ["bench", "72", "2", "10", ".01", "--alphas", "2.0"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(argv)

    def test_decoding_bounds_batches_and_counts_block_failures(self):
        batches = []

        class Decoder:
            def decode_batch(self, dets):
                batches.append(len(dets))
                return np.zeros((len(dets), 2), dtype=np.uint8)

        dets = np.zeros((9, 4), dtype=bool)
        obs = np.zeros((9, 2), dtype=bool)
        obs[[0, 5, 8], 0] = True
        obs[5, 1] = True  # One block failure even when both logicals flip.
        self.assertEqual(bench.decode_failures(Decoder(), dets, obs, 4), 3)
        self.assertEqual(batches, [4, 4, 1])
        failures, outcomes = bench.decode_failures(
            Decoder(), dets, obs, 4, return_outcomes=True)
        self.assertEqual(failures, 3)
        np.testing.assert_array_equal(
            outcomes, [True, False, False, False, False, True, False, False, True])

    @unittest.skipIf(stim is None, "requires optional stim dependency")
    def test_csv_records_actual_solver_settings_and_dem_priors(self):
        settings = []

        class Decoder:
            def __init__(self, h, l, **kwargs):
                self.k = l.shape[0]
                if len(kwargs["probabilities"]) != h.shape[1]:
                    raise AssertionError("one prior is required per DEM mechanism")
                settings.append(kwargs)

            def decode_batch(self, dets):
                return np.zeros((len(dets), self.k), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench, "medium_code", steane_code), \
                patch.object(bench, "BpOsdDemDecoder", Decoder), \
                patch.object(bench, "AdmmOsdDemDecoder", Decoder), \
                contextlib.redirect_stdout(io.StringIO()):
            output = str(Path(tmp) / "circuit.csv")
            bench.main(["bench", "72", "2", "5", ".001", "--basis", "X",
                        "--seed", "4", "--batch-size", "2", "--bp-max-iter", "3",
                        "--bp-alpha", ".8", "--admm-max-iter", "4",
                        "--admm-alpha", "1.2", "--admm-rho", "1.3",
                        "--osd-order", "0", "--osd-lam", "2", "--output", output])
            with open(output, newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, bench.COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual((row["basis"], row["seed"], row["shots"]),
                                 ("X", "4", "5"))
                self.assertEqual((row["batch_size"], row["prior"], row["osd_lam"]),
                                 ("2", "dem", "2"))
            self.assertEqual((rows[0]["max_iter"], rows[0]["alpha"], rows[0]["rho"]),
                             ("3", "0.8", ""))
            self.assertEqual((rows[1]["max_iter"], rows[1]["alpha"], rows[1]["rho"]),
                             ("4", "1.2", "1.3"))
            self.assertTrue(all(np.all(s["probabilities"] > 0) for s in settings))

    @unittest.skipIf(stim is None, "requires optional stim dependency")
    def test_jsonl_records_paired_outcomes_and_tuning_grid(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench, "medium_code", steane_code), \
                contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp) / "circuit.jsonl"
            bench.main(["bench", "72", "2", "4", ".01", "--basis", "Z",
                        "--variants", "bp,admm-u,admm-w", "--rhos", "1.0,2.0",
                        "--alphas", "1.0", "--bp-max-iter", "3",
                        "--admm-max-iter", "5", "--seed", "7",
                        "--batch-size", "2", "--output", str(output)])
            records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r["variant"] for r in records],
                         ["bp", "admm-u-r1-a1", "admm-u-r2-a1",
                          "admm-w-r1-a1", "admm-w-r2-a1"])
        self.assertEqual(len({r["sample_id"] for r in records}), 1)
        self.assertEqual(records[1]["prior"], "uniform")
        self.assertNotIn("weights", records[1]["decoder_config"])
        self.assertEqual(records[3]["prior"], "dem")
        self.assertEqual(records[3]["decoder_config"]["weights"], "normalized-dem-llr")
        summary = analyze_phenom.analyze(records)[0]
        self.assertIn("mcnemar", summary["comparisons"]["admm-w-r2-a1"])


if __name__ == "__main__":
    unittest.main()
