"""Plan, checkpoint, and dispatch the single-SLURM corrected rerun campaign.

Local preview: python scripts/rerun_all.py plan
Cluster entry: sbatch hpc/rerun_all.slurm
Resume: sbatch hpc/rerun_all.slurm resume /absolute/path/to/campaign
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import re
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CODES = ["72", "144"]
RATES = [0.005, 0.01, 0.02]
RHOS = [0.5, 1.0, 2.0, 4.0, 8.0]
ALPHAS = [1.0, 1.5, 1.8]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as fh:
        tmp = Path(fh.name)
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root, entire=False):
    paths = [*root.glob("src/qudec/*.py"), *root.glob("scripts/*.py"), root / "pyproject.toml"]
    if entire:
        paths += [*root.glob("tests/*.py"), *root.glob("hpc/*.sh"),
                  root / "hpc/rerun_all.slurm", root / "README.md"]
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(str(path.relative_to(root)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def config_from_env(env=None):
    env = os.environ if env is None else env
    fields = {"confirm_seeds": 4, "confirm_shots": 2500, "tune_shots": 300,
              "batch_size": 32, "max_parallel": 4, "capacity_shots": 500,
              "explore_shots": 300, "circuit_shots": 300}
    config = {}
    for key, default in fields.items():
        config[key] = int(env.get("QUDEC_" + key.upper(), default))
        if config[key] < 1:
            raise ValueError(f"QUDEC_{key.upper()} must be positive")
    if config["confirm_seeds"] + 3 >= 100000:
        raise ValueError("confirmation seeds must stay below the tuning seed namespace")
    for key, default in (("include_circuit", "1"), ("include_reference", "0")):
        value = env.get("QUDEC_" + key.upper(), default)
        if value not in ("0", "1"):
            raise ValueError(f"QUDEC_{key.upper()} must be 0 or 1")
        config[key] = value == "1"
    config["task_time"] = env.get("QUDEC_TASK_TIME", "12:00:00")
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]", config["task_time"]):
        raise ValueError("QUDEC_TASK_TIME must be [days-]hours:minutes:seconds")
    return config


def make_plan(config):
    tuning = [dict(id=i, code=code, seed=seed, p=.02,
                   shots=config["tune_shots"], rounds=6)
              for i, (code, seed) in enumerate((c, s) for c in CODES for s in (100000, 100001))]
    tasks = []

    def add(kind, **kwargs):
        tasks.append(dict(id=len(tasks), kind=kind, **kwargs))

    for p in [.005, .01, .02, .03, .05, .08, .1]:
        add("capacity", code="144", p=p, seed=0, shots=config["capacity_shots"], rounds=0)
    for code in CODES:
        for p in [*RATES, .03]:
            add("explore", code=code, p=p, seed=200000,
                shots=config["explore_shots"], rounds=6, mode="greedy")
        for p in RATES:
            for mode in ("greedy", "lam60"):
                for seed in range(config["confirm_seeds"]):
                    add("confirm", code=code, p=p, seed=seed,
                        shots=config["confirm_shots"], rounds=6, mode=mode)
    for seed in range(config["confirm_seeds"], config["confirm_seeds"] + 4):
        add("replicate", code="72", p=.02, seed=seed,
            shots=config["confirm_shots"], rounds=6, mode="lam60")
    if config["include_circuit"]:
        for basis in ("Z", "X"):
            for p in (.001, .002):
                add("circuit", code="72", p=p, seed=300000,
                    shots=config["circuit_shots"], rounds=3, basis=basis)
    return {"schema_version": 1, "config": config, "tuning": tuning, "tasks": tasks,
            "coverage": {"tasks_by_kind": dict(Counter(t["kind"] for t in tasks)),
                         "tuning_tasks": len(tuning),
                         "confirmation_shots_per_cell": config["confirm_seeds"] * config["confirm_shots"],
                         "replication_shots": 4 * config["confirm_shots"],
                         "tuning_seeds": [100000, 100001],
                         "confirmation_seeds": list(range(config["confirm_seeds"])),
                         "replication_seeds": list(range(config["confirm_seeds"], config["confirm_seeds"] + 4)),
                         "reference": "fresh paired capacity reference" if config["include_reference"]
                         else "historical external reference excluded"}}


def submission_manifest(manifest, env=None):
    """Allow scheduling changes on retry without altering scientific inputs."""
    env = os.environ if env is None else env
    overrides = {key: env[key] for key in ("QUDEC_TASK_TIME", "QUDEC_MAX_PARALLEL") if key in env}
    validated = config_from_env(overrides)
    result = dict(manifest, config=dict(manifest["config"]))
    for key in overrides:
        field = key.removeprefix("QUDEC_").lower()
        result["config"][field] = validated[field]
    return result


def load_campaign(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    if digest(run_dir / "manifest.json") != (run_dir / "manifest.sha256").read_text().strip():
        raise ValueError("campaign manifest changed; create a new campaign")
    source = run_dir / "source"
    if (source_hash(source) != manifest["source_sha256"]
            or source_hash(source, entire=True) != manifest["snapshot_sha256"]):
        raise ValueError("campaign source snapshot changed; create a new campaign")
    return run_dir, manifest, source


def expected_tuning(manifest, cells=None):
    cells = manifest["tuning"] if cells is None else cells
    return {"codes": sorted({t["code"] for t in cells}),
            "seeds": sorted({t["seed"] for t in cells}), "rhos": RHOS, "alphas": ALPHAS,
            "osd_lam": 60, "p": .02, "shots": manifest["config"]["tune_shots"],
            "rounds": 6, "source_sha256": manifest["source_sha256"], "protocol": "closed-v1"}


def paths_for(run_dir, task, tune=False):
    stem = f"tune_{task['id']:03d}" if tune else f"task_{task['id']:03d}_{task['kind']}"
    suffix = ".csv" if task.get("kind") == "circuit" else ".jsonl"
    path = run_dir / "artifacts" / (stem + suffix)
    return path, path.with_suffix(".done.json")


def task_variants(task, config):
    if task["kind"] == "circuit":
        return ["bp", "admm"]
    if task["kind"] == "capacity":
        return ["bp", "lp", "admm"] + (["reference"] if config["include_reference"] else [])
    return ["bp", "admm", "admm-w"] + (["admm-ldr"] if task["kind"] == "explore" else [])


def validate_outcomes(row):
    """Require complete, internally consistent per-shot block outcomes."""
    import numpy as np
    from qudec.stats import decode_outcomes
    names = ("logical_x", "logical_z", "invalid_x", "invalid_z", "residual_x", "residual_z",
             "failure_x", "failure_z", "failure")
    if type(row["shots"]) is not int or row["shots"] < 1 or set(row["outcomes"]) != set(names):
        raise ValueError("incomplete per-shot outcome fields")
    flags = {}
    for name in names:
        if type(row[name]) is not int or not 0 <= row[name] <= row["shots"]:
            raise ValueError("invalid integer failure count")
        flags[name] = decode_outcomes(row, name)
    for sector in ("x", "z"):
        expected = flags[f"logical_{sector}"] | flags[f"residual_{sector}"] | flags[f"invalid_{sector}"]
        if not np.array_equal(flags[f"failure_{sector}"], expected):
            raise ValueError("sector failure flags are inconsistent")
    if not np.array_equal(flags["failure"], flags["failure_x"] | flags["failure_z"]):
        raise ValueError("block failure flags are inconsistent")
    for rate, count in (("ler", "failure"), ("ler_x", "failure_x"), ("ler_z", "failure_z")):
        if not math.isclose(row[rate], row[count] / row["shots"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("failure rate does not match integer count")


def validate_decoder(row, task, manifest, selection, tune):
    cfg = row["decoder_config"]
    capacity = task.get("kind") == "capacity"
    if tune:
        family = "bp" if row["decoder"] == "BpOsdDecoder" else (
            "admm-w" if "weights_x" in cfg or "weights_z" in cfg else "admm")
        lam, params = cfg.get("osd_lam"), cfg
    else:
        family = row["variant"]
        lam = None if capacity or task["mode"] == "greedy" else 60
        params = ({"rho": 2.0, "alpha": 1.0} if capacity or family in ("bp", "admm-ldr") else
                  selection["settings"][task["code"]][task["mode"]][family])
    expected_class = {"bp": "BpOsdDecoder", "lp": "LpOsdDecoder", "reference": "BposdRefDecoder"}
    if row["decoder"] != expected_class.get(family, "AdmmOsdDecoder"):
        raise ValueError("wrong decoder class")
    device = "cuda" if family.startswith("admm") and row["environment"]["cuda_available"] else "cpu"
    expected = {"osd_order": 1, "osd_lam": lam, "device": device}
    if family == "bp":
        expected.update(max_iter=30, alpha=.9)
    elif family.startswith("admm"):
        expected.update(max_iter=500 if capacity else 150, rho=params["rho"], alpha=params["alpha"],
                        max_r=None, ldr=family == "admm-ldr", ldr_outer=4 if family == "admm-ldr" else 5,
                        ldr_beta=1.0)
        if family == "admm-w":
            expected.update(weights_x="normalized-channel-llr", weights_z="normalized-channel-llr")
    elif family == "reference":
        expected = {"max_iter": 1000, "osd_lam": None, "device": "cpu",
                    "calibration": "fixed-seed-nonzero-syndromes", "wrapper": "scripts/bench_gross_ref.py"}
    if (any(cfg.get(key) != value for key, value in expected.items())
            or (family != "reference" and set(cfg) != set(expected))
            or row["solver_device"] != device or row["osd_device"] != "cpu"):
        raise ValueError("decoder configuration differs from campaign specification")


def validate_output(path, task, manifest, selection=None, tune=False):
    if task.get("kind") == "circuit":
        if __package__:
            from .bench_circuit import COLUMNS
        else:
            from bench_circuit import COLUMNS
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != COLUMNS:
                raise ValueError("unexpected circuit CSV schema")
            rows = list(reader)
        if len(rows) != 2 or {r["variant"] for r in rows} != {"bp", "admm"}:
            raise ValueError("incomplete circuit variants")
        for row in rows:
            bp = row["variant"] == "bp"
            if (row["code"] != task["code"] or row["basis"] != task["basis"]
                    or int(row["d"]) != task["rounds"] or float(row["p"]) != task["p"]
                    or int(row["seed"]) != task["seed"] or int(row["shots"]) != task["shots"]
                    or not 0 <= int(row["failures"]) <= task["shots"]
                    or int(row["observables"]) != 12 or row["prior"] != "dem"
                    or int(row["max_iter"]) != (30 if bp else 500)
                    or float(row["alpha"]) != (.9 if bp else 1.0)
                    or row["rho"] != ("" if bp else "2.0")
                    or row["osd_order"] != "1" or row["osd_lam"] != ""
                    or int(row["batch_size"]) != min(4, manifest["config"]["batch_size"])
                    or row["noise_model"] != "depolarizing_measurement_flip"
                    or not math.isclose(float(row["ler"]), int(row["failures"]) / task["shots"], abs_tol=1e-8)):
                raise ValueError("circuit output does not match campaign cell/settings")
        return rows
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty task output")
    capacity = task.get("kind") == "capacity"
    for row in rows:
        if (row["code"] != task["code"] or row["p"] != task["p"]
                or row["d"] != task["rounds"] or row["seed"] != task["seed"]
                or row["shots"] != task["shots"] or row["p_meas"] != (0 if capacity else task["p"])
                or row["model"] != "depolarizing" or row["schema_version"] != 1
                or row["batch_size"] != manifest["config"]["batch_size"]
                or row["environment"]["source_sha256"] != manifest["source_sha256"]
                or (manifest.get("require_cuda") and not row["environment"]["cuda_available"])
                or row["protocol"] != ("code-capacity-v1" if capacity else "closed-v1")
                or row["projection"] != "exact"):
            raise ValueError("task output does not match manifest identity/source/protocol")
        validate_outcomes(row)
        if any(row[k] != 0 for k in ("invalid_x", "invalid_z", "residual_x", "residual_z")):
            raise ValueError("task has invalid corrections or residual data syndrome")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sample_id"]):
            raise ValueError("invalid sample fingerprint")
        validate_decoder(row, task, manifest, selection, tune)
    if len({r["sample_id"] for r in rows}) != 1:
        raise ValueError("variants did not use identical sampled errors")
    if tune:
        if __package__:
            from .select_tuning import select
        else:
            from select_tuning import select
        select(rows, expected_tuning(manifest, [task]))
    else:
        wanted = task_variants(task, manifest["config"])
        if len(rows) != len(wanted) or {r["variant"] for r in rows} != set(wanted):
            raise ValueError("missing or repeated variants")
    return rows


def completion_info(run_dir, task, manifest, selection, tune=False):
    path, marker = paths_for(run_dir, task, tune)
    info = {"task": task, "source_sha256": manifest["source_sha256"],
            "selection_sha256": None if tune else digest(run_dir / "selected.json")}
    if not path.exists() or not marker.exists():
        return False
    try:
        saved = read_json(marker)
        if any(saved.get(k) != v for k, v in info.items()) or saved.get("output_sha256") != digest(path):
            return False
        validate_output(path, task, manifest, selection, tune)
    except (KeyError, ValueError, OSError, TypeError):
        return False
    return True


def commands_for(task, manifest, selection, output):
    cfg, py = manifest["config"], sys.executable
    common = ["--seed", str(task["seed"]), "--batch-size", str(cfg["batch_size"]), "--output", str(output)]
    if task.get("kind") == "capacity":
        return [[py, "scripts/bench_capacity.py", "--code", task["code"],
                 "--shots", str(task["shots"]), "--ps", str(task["p"]),
                 "--variants", ",".join(task_variants(task, cfg)), *common]]
    if task.get("kind") == "circuit":
        return [[py, "scripts/bench_circuit.py", task["code"], str(task["rounds"]),
                 str(task["shots"]), str(task["p"]), "--basis", task["basis"],
                 "--batch-size", str(min(4, cfg["batch_size"])), "--seed", str(task["seed"]),
                 "--output", str(output)]]
    if "kind" not in task:
        return [[py, "scripts/bench_tune.py", task["code"], str(task["rounds"]),
                 str(task["shots"]), str(task["p"]), *common,
                 "--rhos", ",".join(map(str, RHOS)), "--alphas", ",".join(map(str, ALPHAS))]]
    commands = []
    for variant in task_variants(task, cfg):
        command = [py, "scripts/bench_phenom.py", task["code"], str(task["rounds"]),
                   str(task["shots"]), str(task["p"]), "--variants", variant, *common]
        if task["mode"] == "lam60":
            command += ["--osd-lam", "60"]
        if variant != "bp":
            # LDR's negative quadratic requires a sufficiently large rho.
            # Plain-ADMM tuning includes rho=.5 and cannot select its rho.
            params = ({"rho": 2.0, "alpha": 1.0} if variant == "admm-ldr" else
                      selection["settings"][task["code"]][task["mode"]][variant])
            command += ["--admm-rho", str(params["rho"]), "--admm-alpha", str(params["alpha"])]
        commands.append(command)
    return commands


def run_checked(command, source):
    print("RUN", json.dumps(command), flush=True)
    env = dict(os.environ, PYTHONPATH=str(source / "src") + os.pathsep + str(source),
               PYTHONUNBUFFERED="1")
    subprocess.run(command, cwd=source, env=env, check=True)


@contextmanager
def task_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"task already running: {path.name}") from exc
        yield


def run_task(run_dir, index, tune=False):
    run_dir, manifest, source = load_campaign(run_dir)
    tasks = manifest["tuning" if tune else "tasks"]
    if index < 0 or index >= len(tasks):
        raise ValueError("task index outside campaign manifest")
    task = tasks[index]
    selection = None if tune else read_selection(run_dir, manifest)
    path, marker = paths_for(run_dir, task, tune)
    with task_lock(marker.with_suffix(".lock")):
        if completion_info(run_dir, task, manifest, selection, tune):
            print("SKIP completed", path, flush=True)
            return
        temporary = path.with_name(path.stem + f".partial.{os.getpid()}" + path.suffix)
        temporary.unlink(missing_ok=True)
        for command in commands_for(task, manifest, selection, temporary):
            run_checked(command, source)
        validate_output(temporary, task, manifest, selection, tune)
        os.replace(temporary, path)
        atomic_json(marker, {"task": task, "source_sha256": manifest["source_sha256"],
                             "selection_sha256": None if tune else digest(run_dir / "selected.json"),
                             "output_sha256": digest(path)})
    print("COMPLETE", path, flush=True)


def select_settings(run_dir, quiet=False):
    run_dir, manifest, _ = load_campaign(run_dir)
    if __package__:
        from .select_tuning import select
    else:
        from select_tuning import select
    rows, inputs = [], {}
    for task in manifest["tuning"]:
        if not completion_info(run_dir, task, manifest, None, tune=True):
            raise ValueError(f"missing or invalid tuning task {task['id']}")
        path, _ = paths_for(run_dir, task, tune=True)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        inputs[path.name] = digest(path)
    value = {"schema_version": 1, "source_sha256": manifest["source_sha256"],
             "tuning_outputs": inputs, "settings": select(rows, expected_tuning(manifest))}
    path = run_dir / "selected.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("frozen tuning changed; start a new campaign instead of changing confirmation settings")
    if not path.exists():
        atomic_json(path, value)
    if not quiet:
        print(json.dumps(value["settings"], indent=2), flush=True)
    return value


def read_selection(run_dir, manifest):
    # Recompute the declared-grid selection rather than trusting editable
    # settings. A changed selection can never silently resume old cells.
    if not (run_dir / "selected.json").exists():
        raise ValueError("tuning settings have not been frozen")
    return select_settings(run_dir, quiet=True)


def submit_stage(run_dir, manifest, stage, dependency=None, indexes=None):
    source = run_dir / "source"
    command = ["sbatch", "--parsable", f"--chdir={source}",
               f"--job-name=qudec_{stage}", "--kill-on-invalid-dep=yes",
               f"--output={run_dir}/logs/{stage}_%A_%a.out",
               f"--error={run_dir}/logs/{stage}_%A_%a.err",
               f"--time={manifest['config']['task_time'] if stage in ('tune', 'work') else '00:30:00'}"]
    if dependency:
        command += [f"--dependency=afterok:{dependency}"]
    if indexes:
        command += ["--array=" + ",".join(map(str, indexes)) + f"%{manifest['config']['max_parallel']}"]
    command += [str(source / "hpc/rerun_all.slurm"), stage, str(run_dir)]
    result = subprocess.check_output(command, text=True, cwd=source).strip()
    job_id = result.split(";")[0]
    if not job_id.isdecimal():
        raise ValueError(f"unexpected sbatch job id: {result}")
    with (run_dir / "submissions.jsonl").open("a") as fh:
        fh.write(json.dumps({"stage": stage, "job_id": job_id, "command": command,
                             "created_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(f"SUBMITTED {stage}: {job_id}", flush=True)
    return job_id


def schedule(run_dir, manifest):
    # afterok gates prevent confirmations from running with failed tuning.
    pending_tune = [t["id"] for t in manifest["tuning"]
                    if not completion_info(run_dir, t, manifest, None, tune=True)]
    dependency = submit_stage(run_dir, manifest, "tune", indexes=pending_tune) if pending_tune else None
    if dependency or not (run_dir / "selected.json").exists():
        dependency = submit_stage(run_dir, manifest, "select", dependency)
        pending_work = [t["id"] for t in manifest["tasks"]]
    else:
        selection = select_settings(run_dir)
        pending_work = [t["id"] for t in manifest["tasks"]
                        if not completion_info(run_dir, t, manifest, selection)]
    if pending_work:
        dependency = submit_stage(run_dir, manifest, "work", dependency, indexes=pending_work)
    submit_stage(run_dir, manifest, "report", dependency)
    print(f"Campaign: {run_dir}\nThe launcher exits; dependency jobs run the campaign.", flush=True)


def preflight(run_dir):
    run_dir, manifest, source = load_campaign(run_dir)
    import stim  # optional locally, required for the corrected cluster gate
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable on allocated node; fix the environment before rerunning")
    if manifest["config"]["include_reference"]:
        import bposd
        import ldpc
    print(f"CUDA {torch.cuda.get_device_name(0)}; torch {torch.__version__}; Stim {stim.__version__}", flush=True)
    run_checked([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], source)
    run_checked([sys.executable, "scripts/smoke_circuit.py"], source)
    # Exercise the actual larger-code CUDA path before any long array is submitted.
    pilot = run_dir / "pilot.jsonl"
    pilot.unlink(missing_ok=True)
    run_checked([sys.executable, "scripts/bench_phenom.py", "144", "6", "8", ".02",
                 "--variants", "bp,admm,admm-w", "--seed", "400000", "--batch-size", "4",
                 "--output", str(pilot)], source)
    rows = [json.loads(line) for line in pilot.read_text().splitlines()]
    if len(rows) != 3 or any(r[k] for r in rows for k in ("invalid_x", "invalid_z", "residual_x", "residual_z")):
        raise ValueError("larger-code pilot failed correction checks")


def launch(run_dir=None, resume=False):
    if resume:
        run_dir, manifest, _ = load_campaign(run_dir)
    else:
        label = os.environ.get("SLURM_JOB_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        run_dir = Path(run_dir or ROOT / "results/reruns" / label).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        source = run_dir / "source"
        source.mkdir()
        for directory in ("src", "scripts", "tests"):
            shutil.copytree(ROOT / directory, source / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (source / "hpc").mkdir()
        shutil.copy2(ROOT / "hpc/rerun_all.slurm", source / "hpc/rerun_all.slurm")
        for helper in (ROOT / "hpc").glob("*.sh"):
            shutil.copy2(helper, source / "hpc" / helper.name)
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(ROOT / name, source / name)
        for directory in ("artifacts", "logs"):
            (run_dir / directory).mkdir()
        manifest = make_plan(config_from_env())
        manifest.update(source_sha256=source_hash(source), snapshot_sha256=source_hash(source, entire=True),
                        created_utc=datetime.now(timezone.utc).isoformat())
        manifest["require_cuda"] = True
        atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "manifest.sha256").write_text(digest(run_dir / "manifest.json") + "\n")
    run_checked([sys.executable, "scripts/rerun_all.py", "validate", str(run_dir)], run_dir / "source")
    schedule(run_dir, submission_manifest(manifest))


def report(run_dir):
    run_dir, manifest, _ = load_campaign(run_dir)
    (run_dir / "report.json").unlink(missing_ok=True)
    selection = select_settings(run_dir)
    if __package__:
        from .analyze_phenom import analyze
    else:
        from analyze_phenom import analyze
    groups = {}
    circuit = []
    for task in manifest["tasks"]:
        if not completion_info(run_dir, task, manifest, selection):
            raise ValueError(f"campaign incomplete: task {task['id']} ({task['kind']})")
        path, _ = paths_for(run_dir, task)
        rows = validate_output(path, task, manifest, selection)
        if task["kind"] == "circuit":
            circuit.extend(rows)
        else:
            groups.setdefault(task["kind"], []).extend(rows)
    analyses = {kind: analyze(rows) for kind, rows in groups.items()}
    from scipy.stats import binomtest
    for phase in analyses.values():
        for group in phase:
            for variant in group["variants"].values():
                interval = binomtest(variant["failures"], variant["shots"]).proportion_ci(
                    confidence_level=.95, method="exact")
                variant["confidence_interval_95"] = [float(interval.low), float(interval.high)]
    for row in circuit:
        interval = binomtest(int(row["failures"]), int(row["shots"])).proportion_ci(
            confidence_level=.95, method="exact")
        row["confidence_interval_95"] = [float(interval.low), float(interval.high)]
    # Holm adjustment across all predeclared primary confirmation tests.
    primary = []
    for group in analyses.get("confirm", []):
        for name, comparison in group["comparisons"].items():
            if "mcnemar" not in comparison:
                raise ValueError("primary comparison lacks identical paired samples")
            primary.append((comparison["mcnemar"]["p_value"], comparison))
    running = 0.0
    for rank, (p, comparison) in enumerate(sorted(primary, key=lambda item: item[0])):
        running = max(running, min(1.0, (len(primary) - rank) * p))
        comparison["holm_p_value"] = running
    result = {"complete": True, "source_sha256": manifest["source_sha256"],
              "selection_sha256": digest(run_dir / "selected.json"), "coverage": manifest["coverage"],
              "analyses": analyses, "circuit_preliminary": circuit,
              "rate_intervals": "95% exact binomial; not adjusted for multiple comparisons",
              "primary_multiple_testing": {"method": "Holm", "comparisons": len(primary)},
              "replication": "separate independent cohort; not pooled into primary confirmation",
              "created_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(run_dir / "report.json", result)
    print(f"COMPLETE: {run_dir / 'report.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "launch", "resume", "validate", "tune", "select", "work", "report"))
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    try:
        if args.stage == "plan":
            print(json.dumps(make_plan(config_from_env()), indent=2))
        elif args.stage in ("launch", "resume"):
            if args.stage == "resume" and args.run_dir is None:
                raise ValueError("resume requires the absolute campaign path")
            launch(args.run_dir, resume=args.stage == "resume")
        else:
            if args.run_dir is None:
                raise ValueError("stage requires a campaign path")
            if args.stage in ("tune", "work"):
                index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
                run_task(args.run_dir, index, tune=args.stage == "tune")
            elif args.stage == "validate":
                preflight(args.run_dir)
            elif args.stage == "select":
                select_settings(args.run_dir)
            else:
                report(args.run_dir)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"campaign failed: {exc}\n")


if __name__ == "__main__":
    main()
