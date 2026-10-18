"""Campaign coverage, dependency gates, immutable inputs and atomic retries."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import rerun_all as campaign


def selection():
    return {"schema_version": 1, "settings": {
        code: {mode: {"admm": {"rho": .5, "alpha": 1.0},
                      "admm-w": {"rho": .5, "alpha": 1.0}}
               for mode in ("greedy", "lam60")} for code in campaign.CODES}}


def rows_for(task, manifest, tune=False):
    n = task["shots"]
    common = {"schema_version": 1, "code": task["code"], "d": task["rounds"],
              "p": task["p"], "p_meas": 0 if task.get("kind") == "capacity" else task["p"],
              "shots": n, "seed": task["seed"], "model": "depolarizing",
              "protocol": "code-capacity-v1" if task.get("kind") == "capacity" else "closed-v1",
              "projection": "exact", "batch_size": manifest["config"]["batch_size"],
              "environment": {"source_sha256": manifest["source_sha256"], "cuda_available": False},
              "solver_device": "cpu", "osd_device": "cpu",
              "sample_id": hashlib.sha256(str((task["code"], task["p"], task["seed"], n)).encode()).hexdigest(),
              "ler": 0., "ler_x": 0., "ler_z": 0.}
    flags = ("logical_x", "logical_z", "invalid_x", "invalid_z", "residual_x", "residual_z",
             "failure_x", "failure_z", "failure")
    common.update({key: 0 for key in flags})
    common["outcomes"] = {key: "0" * n for key in flags}
    rows = []

    def add(family, mode, rho=.5, alpha=1.0, label=None):
        row = deepcopy(common)
        if family == "admm-ldr":
            rho, alpha = 2., 1.
        lam = None if mode == "greedy" else 60
        config = dict(osd_order=1, osd_lam=lam, device="cpu")
        if family == "bp":
            cls = "BpOsdDecoder"
            config.update(max_iter=30, alpha=.9)
        elif family == "lp":
            cls = "LpOsdDecoder"
        else:
            cls = "AdmmOsdDecoder"
            config.update(max_iter=500 if task.get("kind") == "capacity" else 150,
                          rho=rho, alpha=alpha, max_r=None, ldr=family == "admm-ldr",
                          ldr_outer=4 if family == "admm-ldr" else 5, ldr_beta=1.0)
            if family == "admm-w":
                config.update(weights_x="normalized-channel-llr", weights_z="normalized-channel-llr")
        row.update(variant=label or family, decoder=cls, decoder_config=config)
        rows.append(row)
    if tune:
        for mode in ("greedy", "lam60"):
            add("bp", mode, label="bp" if mode == "greedy" else "bp-lam")
            for family in ("admm", "admm-w"):
                for rho in campaign.RHOS:
                    for alpha in campaign.ALPHAS:
                        add(family, mode, rho, alpha, f"{family}-{mode}-r{rho}-a{alpha}")
    else:
        for family in campaign.task_variants(task, manifest["config"]):
            add(family, task.get("mode", "greedy"), rho=2. if task["kind"] == "capacity" else .5)
    return rows


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def mark_complete(run_dir, task, manifest, tune=False):
    path, marker = campaign.paths_for(run_dir, task, tune)
    save_rows(path, rows_for(task, manifest, tune))
    campaign.atomic_json(marker, {"task": task, "source_sha256": manifest["source_sha256"],
                                 "selection_sha256": None if tune else campaign.digest(run_dir / "selected.json"),
                                 "output_sha256": campaign.digest(path)})


class TestCampaign(unittest.TestCase):
    def test_default_complete_scope_and_independent_cohorts(self):
        plan = campaign.make_plan(campaign.config_from_env({}))
        self.assertEqual(len(plan["tuning"]), 4)
        self.assertEqual(len(plan["tasks"]), 71)
        self.assertEqual(plan["coverage"]["confirmation_shots_per_cell"], 10000)
        confirmation = [t for t in plan["tasks"] if t["kind"] == "confirm"]
        for code in campaign.CODES:
            for rate in campaign.RATES:
                for mode in ("greedy", "lam60"):
                    cells = [t for t in confirmation if (t["code"], t["p"], t["mode"]) == (code, rate, mode)]
                    self.assertEqual({t["seed"] for t in cells}, {0, 1, 2, 3})
                    self.assertEqual(sum(t["shots"] for t in cells), 10000)
        self.assertTrue(set(plan["coverage"]["tuning_seeds"]).isdisjoint(
            plan["coverage"]["confirmation_seeds"] + plan["coverage"]["replication_seeds"]))
        self.assertEqual(plan["coverage"]["replication_seeds"], [4, 5, 6, 7])

    def test_overrides_and_invalid_namespaces(self):
        plan = campaign.make_plan(campaign.config_from_env({"QUDEC_CONFIRM_SEEDS": "40",
                                                           "QUDEC_INCLUDE_CIRCUIT": "0"}))
        self.assertEqual(plan["coverage"]["confirmation_shots_per_cell"], 100000)
        self.assertEqual(plan["coverage"]["replication_seeds"], [40, 41, 42, 43])
        self.assertNotIn("circuit", plan["coverage"]["tasks_by_kind"])
        for env in ({"QUDEC_CONFIRM_SEEDS": "99997"}, {"QUDEC_BATCH_SIZE": "0"},
                    {"QUDEC_INCLUDE_REFERENCE": "yes"}, {"QUDEC_TASK_TIME": "bad"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                campaign.config_from_env(env)

    def test_commands_use_separate_frozen_family_settings(self):
        manifest = campaign.make_plan(campaign.config_from_env({}))
        task = next(t for t in manifest["tasks"] if t["kind"] == "confirm" and t["mode"] == "lam60")
        picked = selection()
        picked["settings"]["72"]["lam60"]["admm-w"] = {"rho": 8., "alpha": 1.8}
        commands = campaign.commands_for(task, manifest, picked, Path("/tmp/output.jsonl"))
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertIn("--osd-lam", command)
        self.assertEqual(commands[1][commands[1].index("--admm-rho") + 1], "0.5")
        self.assertEqual(commands[2][commands[2].index("--admm-rho") + 1], "8.0")
        self.assertTrue(all(command[command.index("--seed") + 1] == str(task["seed"]) for command in commands))
        self.assertNotIn("--legacy-open-boundary", commands[0])
        exploratory = next(t for t in manifest["tasks"] if t["kind"] == "explore")
        ldr = campaign.commands_for(exploratory, manifest, picked, Path("/tmp/ldr.jsonl"))[-1]
        self.assertEqual(ldr[ldr.index("--variants") + 1], "admm-ldr")
        self.assertEqual(ldr[ldr.index("--admm-rho") + 1], "2.0")

    def test_retry_can_change_scheduling_without_changing_samples(self):
        plan = campaign.make_plan(campaign.config_from_env({}))
        changed = campaign.submission_manifest(plan, {"QUDEC_TASK_TIME": "24:00:00",
                                                      "QUDEC_MAX_PARALLEL": "2",
                                                      "QUDEC_CONFIRM_SHOTS": "9999"})
        self.assertEqual(changed["config"]["task_time"], "24:00:00")
        self.assertEqual(changed["config"]["max_parallel"], 2)
        self.assertEqual(changed["config"]["confirm_shots"], 2500)
        self.assertEqual(changed["tasks"], plan["tasks"])
        self.assertEqual(plan["config"]["task_time"], "12:00:00")

    def test_one_slurm_and_source_snapshot_is_frozen(self):
        self.assertEqual([p.name for p in (campaign.ROOT / "hpc").glob("*.slurm")], ["rerun_all.slurm"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign, "run_checked") as execute, \
                patch.object(campaign, "schedule"), redirect_stdout(io.StringIO()):
            root = Path(tmp) / "campaign"
            campaign.launch(root)
            loaded, manifest, source = campaign.load_campaign(root)
            self.assertEqual(loaded, root.resolve())
            self.assertIn("validate", execute.call_args.args[0])
            self.assertEqual(campaign.source_hash(source), manifest["source_sha256"])
            self.assertFalse((source / ".git").exists())
            self.assertTrue((source / "hpc/sync.sh").is_file())
            (source / "src/qudec/bp.py").write_text("modified")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                campaign.load_campaign(root)

    def test_dependency_graph_uses_same_slurm_and_bounded_arrays(self):
        plan = campaign.make_plan(campaign.config_from_env({}))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(campaign, "completion_info", return_value=False), \
                patch.object(campaign.subprocess, "check_output", side_effect=["101\n", "102\n", "103\n", "104\n"]) as sbatch:
            root = Path(tmp)
            campaign.schedule(root, plan)
            commands = [call.args[0] for call in sbatch.call_args_list]
            self.assertEqual(len(commands), 4)
            self.assertIn("--array=0,1,2,3%4", commands[0])
            for index, dependency in ((1, "101"), (2, "102"), (3, "103")):
                self.assertIn(f"--dependency=afterok:{dependency}", commands[index])
            self.assertTrue(any(arg.endswith("%4") and arg.startswith("--array=") for arg in commands[2]))
            self.assertTrue(all(str(root / "source/hpc/rerun_all.slurm") in command for command in commands))

    def test_resume_schedules_only_incomplete_work(self):
        plan = campaign.make_plan(campaign.config_from_env({}))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / "selected.json").write_text("{}")
            def complete(_root, task, _manifest, _selection, tune=False):
                return tune or task["id"] != 12
            with patch.object(campaign, "completion_info", side_effect=complete), \
                    patch.object(campaign, "select_settings", return_value=selection()), \
                    patch.object(campaign.subprocess, "check_output", side_effect=["201\n", "202\n"]) as sbatch:
                campaign.schedule(root, plan)
            commands = [call.args[0] for call in sbatch.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn("--array=12%4", commands[0])
            self.assertIn("--dependency=afterok:201", commands[1])

    def test_validation_rejects_incomplete_wrong_or_inconsistent_records(self):
        manifest = campaign.make_plan(campaign.config_from_env({"QUDEC_CONFIRM_SHOTS": "2"}))
        manifest["source_sha256"] = "a" * 64
        task = next(t for t in manifest["tasks"] if t["kind"] == "confirm")
        rows = rows_for(task, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.jsonl"
            save_rows(path, rows)
            campaign.validate_output(path, task, manifest, selection())
            mutations = []
            bad = deepcopy(rows); bad.pop(); mutations.append(bad)
            bad = deepcopy(rows); bad[0]["p_meas"] = .3; mutations.append(bad)
            bad = deepcopy(rows); bad[1]["decoder_config"]["max_iter"] = 1; mutations.append(bad)
            bad = deepcopy(rows); bad[2]["decoder_config"]["weights_z"] = "wrong"; mutations.append(bad)
            bad = deepcopy(rows); del bad[0]["outcomes"]["failure"]; mutations.append(bad)
            bad = deepcopy(rows); bad[0]["ler"] = .9; mutations.append(bad)
            bad = deepcopy(rows); bad[2]["sample_id"] = "b" * 64; mutations.append(bad)
            for bad in mutations:
                with self.subTest(bad=bad[0].get("ler")), self.assertRaises((ValueError, KeyError)):
                    save_rows(path, bad)
                    campaign.validate_output(path, task, manifest, selection())

    def test_checkpoint_checks_content_hash_and_output_semantics(self):
        manifest = campaign.make_plan(campaign.config_from_env({"QUDEC_CONFIRM_SHOTS": "1"}))
        manifest["source_sha256"] = "a" * 64
        task = next(t for t in manifest["tasks"] if t["kind"] == "confirm")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign.atomic_json(root / "selected.json", selection())
            mark_complete(root, task, manifest)
            self.assertTrue(campaign.completion_info(root, task, manifest, selection()))
            path, _ = campaign.paths_for(root, task)
            path.write_text(path.read_text() + "\n")
            self.assertFalse(campaign.completion_info(root, task, manifest, selection()))

    def test_worker_retry_replaces_partial_output_and_then_skips(self):
        manifest = campaign.make_plan(campaign.config_from_env({"QUDEC_CONFIRM_SHOTS": "2"}))
        manifest["source_sha256"] = "a" * 64
        index = next(t["id"] for t in manifest["tasks"] if t["kind"] == "confirm")
        task = manifest["tasks"][index]
        expected = rows_for(task, manifest)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            campaign.atomic_json(root / "selected.json", selection())
            executions = []
            def execute(command, source):
                output = Path(command[command.index("--output") + 1])
                variant = command[command.index("--variants") + 1]
                if executions and len(executions) == 1:
                    raise RuntimeError("injected worker failure")
                with output.open("a") as fh:
                    fh.write(json.dumps(next(r for r in expected if r["variant"] == variant)) + "\n")
                executions.append(variant)
            with patch.object(campaign, "load_campaign", return_value=(root, manifest, campaign.ROOT)), \
                    patch.object(campaign, "read_selection", return_value=selection()), \
                    patch.object(campaign, "run_checked", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    campaign.run_task(root, index)
            output, marker = campaign.paths_for(root, task)
            self.assertFalse(output.exists())
            self.assertFalse(marker.exists())
            def complete(command, source):
                output = Path(command[command.index("--output") + 1])
                variant = command[command.index("--variants") + 1]
                with output.open("a") as fh:
                    fh.write(json.dumps(next(r for r in expected if r["variant"] == variant)) + "\n")
            with patch.object(campaign, "load_campaign", return_value=(root, manifest, campaign.ROOT)), \
                    patch.object(campaign, "read_selection", return_value=selection()), \
                    patch.object(campaign, "run_checked", side_effect=complete) as executor:
                campaign.run_task(root, index)
                self.assertEqual(executor.call_count, 3)
                campaign.run_task(root, index)
                self.assertEqual(executor.call_count, 3)
            self.assertEqual(len(output.read_text().splitlines()), 3)
            self.assertTrue(campaign.completion_info(root, task, manifest, selection()))

    def test_selector_and_complete_report_keep_replication_separate(self):
        config = campaign.config_from_env({"QUDEC_CONFIRM_SEEDS": "1", "QUDEC_CONFIRM_SHOTS": "1",
                                           "QUDEC_TUNE_SHOTS": "1", "QUDEC_CAPACITY_SHOTS": "1",
                                           "QUDEC_EXPLORE_SHOTS": "1", "QUDEC_INCLUDE_CIRCUIT": "0"})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(campaign, "run_checked"), patch.object(campaign, "schedule"):
            root = Path(tmp) / "campaign"
            with patch.object(campaign, "config_from_env", return_value=config):
                campaign.launch(root)
            manifest = campaign.read_json(root / "manifest.json")
            manifest["require_cuda"] = False  # CPU fixture, no scheduler execution.
            campaign.atomic_json(root / "manifest.json", manifest)
            (root / "manifest.sha256").write_text(campaign.digest(root / "manifest.json"))
            for task in manifest["tuning"]:
                mark_complete(root, task, manifest, tune=True)
            picked = campaign.select_settings(root)
            self.assertEqual(picked["settings"], selection()["settings"])
            for task in manifest["tasks"]:
                mark_complete(root, task, manifest)
            campaign.report(root)
            report = campaign.read_json(root / "report.json")
            self.assertTrue(report["complete"])
            self.assertEqual(report["primary_multiple_testing"], {
                "method": "Holm", "comparisons": 24,
                "raw_p_value_field": "mcnemar.p_value",
                "adjusted_p_value_field": "holm_p_value"})
            self.assertEqual(report["analyses"]["confirm"][0]["variants"]["bp"]["shots"], 1)
            self.assertEqual(report["analyses"]["replicate"][0]["variants"]["bp"]["shots"], 4)
            low, high = report["analyses"]["replicate"][0]["variants"]["bp"]["confidence_interval_95"]
            self.assertEqual(low, 0.)
            self.assertGreater(high, .5)
            for group in report["analyses"]["confirm"]:
                self.assertTrue(group["p_values_multiple_testing_corrected"])
                for result in group["comparisons"].values():
                    self.assertEqual(result["mcnemar"]["p_value"], 1.)
                    self.assertEqual(result["holm_p_value"], 1.)
            for phase in ("capacity", "explore", "replicate"):
                for group in report["analyses"][phase]:
                    self.assertFalse(group["p_values_multiple_testing_corrected"])
                    for result in group["comparisons"].values():
                        self.assertNotIn("holm_p_value", result)
            # Standalone analysis remains raw, even for confirmation records.
            from scripts.analyze_phenom import analyze
            task = next(t for t in manifest["tasks"] if t["kind"] == "confirm")
            self.assertFalse(analyze(rows_for(task, manifest))[0]["p_values_multiple_testing_corrected"])
            first, _ = campaign.paths_for(root, manifest["tasks"][0])
            first.unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                campaign.report(root)
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
