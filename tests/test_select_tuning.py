"""Parameter selection uses complete, independent and matched tuning grids."""

from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from scripts.select_tuning import main, select, write_atomic


EXPECTED = {"codes": ["72"], "seeds": [100000, 100001], "rhos": [.5, 2.0],
            "alphas": [1.0, 1.5], "osd_lam": 60, "p": .02, "shots": 20,
            "rounds": 6, "source_sha256": "source", "protocol": "closed-v1"}


def fixture():
    records = []
    for seed, lam in itertools.product(EXPECTED["seeds"], (None, 60)):
        common = {"code": "72", "d": 6, "p": .02, "p_meas": .02,
                  "model": "depolarizing", "protocol": "closed-v1", "projection": "exact",
                  "environment": {"source_sha256": "source"}, "shots": 20, "seed": seed}
        bp = dict(common, decoder="BpOsdDecoder", variant="bp" if lam is None else "bp-lam",
                  decoder_config={"max_iter": 30, "alpha": .9, "osd_order": 1, "osd_lam": lam})
        bp.update(failure=5, outcomes={"failure": "11111"+"0"*15})
        records.append(bp)
        for weighted, rho, alpha in itertools.product((False, True), EXPECTED["rhos"], EXPECTED["alphas"]):
            config = {"max_iter": 150, "rho": rho, "alpha": alpha, "osd_order": 1,
                      "osd_lam": lam, "ldr": False, "ldr_beta": 1.0}
            if weighted:
                config.update(weights_x="normalized-channel-llr", weights_z="normalized-channel-llr")
            row = dict(common, decoder="AdmmOsdDecoder", decoder_config=config,
                       variant=f"opaque-name-{len(records)}")
            # Plain prefers rho2 across both seeds, despite rho.5 winning one seed.
            if weighted:
                failures = 4  # Tie: deterministic rho.5, alpha1.
            elif alpha != 1:
                failures = 10
            elif rho == .5:
                failures = 0 if seed == 100000 else 8
            else:
                failures = 3
            row.update(failure=failures, outcomes={"failure": "1"*failures+"0"*(20-failures)})
            records.append(row)
    return records


class TestSelectTuning(unittest.TestCase):
    def test_selects_total_failures_and_deterministic_ties(self):
        result = select(fixture(), EXPECTED)
        for mode in ("greedy", "lam60"):
            self.assertEqual(result["72"][mode]["admm"], {"rho": 2., "alpha": 1.})
            self.assertEqual(result["72"][mode]["admm-w"], {"rho": .5, "alpha": 1.})

    def test_missing_or_duplicate_cell_rejected(self):
        rows = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select(rows[:-1], EXPECTED)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select(rows+[deepcopy(rows[-1])], EXPECTED)

    def test_declared_axes_detect_wholly_missing_training_seed(self):
        rows = [r for r in fixture() if r["seed"] != 100001]
        with self.assertRaisesRegex(ValueError, "seeds"):
            select(rows, EXPECTED)
        # Inference alone cannot distinguish a deliberately smaller experiment.
        self.assertIn("72", select(rows))

    def test_rejects_confirmation_seeds(self):
        rows = fixture()
        for row in rows:
            row["seed"] -= 100000
        with self.assertRaisesRegex(ValueError, "seeds"):
            select(rows, EXPECTED)

    def test_mixed_source_protocol_order_shots_or_settings_rejected(self):
        for change in ("source", "protocol", "order", "shots", "iterations", "outcomes"):
            rows = fixture()
            row = rows[1]
            if change == "source":
                row["environment"]["source_sha256"] = "different"
            elif change == "protocol":
                row["protocol"] = "legacy-open-v1"
            elif change == "order":
                row["decoder_config"]["osd_order"] = 0
            elif change == "shots":
                row["shots"] = 21
            elif change == "iterations":
                row["decoder_config"]["max_iter"] = 1
            else:
                row["outcomes"]["failure"] = "1"*20
            with self.subTest(change=change), self.assertRaises(ValueError):
                select(rows, EXPECTED)

    def test_matching_osd_baseline_required(self):
        rows = [r for r in fixture() if not (r["decoder"] == "BpOsdDecoder"
                                            and r["decoder_config"]["osd_lam"] == 60)]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select(rows, EXPECTED)

    def test_atomic_cli_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, expected, output = root/"tune.jsonl", root/"expected.json", root/"selected.json"
            rows.write_text("".join(json.dumps(r)+"\n" for r in fixture()))
            expected.write_text(json.dumps(EXPECTED))
            main([str(rows), "--expected", str(expected), "--output", str(output)])
            self.assertEqual(json.loads(output.read_text()), select(fixture(), EXPECTED))
            write_atomic(output, {"replacement": True})
            self.assertEqual(json.loads(output.read_text()), {"replacement": True})
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
