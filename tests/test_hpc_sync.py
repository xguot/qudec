"""Retiring exact remote wrappers preserves artifacts and honors dry runs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RETIRED = ["run_test_bench.slurm", "run_tune.slurm", "run_phenom.slurm", "run_phenom_144.slurm",
           "run_phenom_confirm.slurm", "run_phenom_confirm_rho4.slurm", "run_circuit.slurm", "probe_cuda.slurm"]


class TestHpcSync(unittest.TestCase):
    def run_sync(self, dry_run, fail_transfer=False):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            remote = tmp / "remote space'quote"
            (remote / "hpc/logs").mkdir(parents=True)
            (remote / "results").mkdir()
            for name in RETIRED + ["custom.slurm", "rerun_all.slurm"]:
                (remote / "hpc" / name).write_text("keep unless retired")
            for name in ("hpc/logs/old.out", "results/old.jsonl"):
                (remote / name).write_text("historical")
            binary = tmp / "bin"
            binary.mkdir()
            log = tmp / "commands.jsonl"
            ssh = binary / "ssh"
            ssh.write_text(f"#!{sys.executable}\n" + '''import json, os, subprocess, sys
with open(os.environ["SYNC_TEST_LOG"], "a") as fh:
    fh.write(json.dumps(["ssh", *sys.argv[1:]]) + "\\n")
result = subprocess.run(sys.argv[2], input=sys.stdin.read(), shell=True, text=True, executable="/bin/sh")
sys.exit(result.returncode)
''')
            rsync = binary / "rsync"
            rsync.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["SYNC_TEST_LOG"], "a") as fh:
    fh.write(json.dumps(["rsync", *sys.argv[1:]]) + "\\n")
sys.exit(3 if os.environ.get("SYNC_TEST_FAIL") else 0)
''')
            ssh.chmod(0o755)
            rsync.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       HPC_REMOTE="mock-host", HPC_PATH=str(remote), SYNC_TEST_LOG=str(log))
            if fail_transfer:
                env["SYNC_TEST_FAIL"] = "1"
            command = ["bash", str(ROOT / "hpc/sync.sh"), "push"]
            if dry_run:
                command.append("--dry-run")
            result = subprocess.run(command, env=env, cwd=ROOT, text=True, capture_output=True)
            if fail_transfer:
                self.assertNotEqual(result.returncode, 0)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            if dry_run:
                self.assertFalse(any(call[0] == "ssh" for call in calls))
            for name in RETIRED:
                self.assertEqual((remote / "hpc" / name).exists(), dry_run or fail_transfer)
            for name in ("hpc/custom.slurm", "hpc/rerun_all.slurm", "hpc/logs/old.out", "results/old.jsonl"):
                self.assertTrue((remote / name).exists())
            self.assertNotIn("--delete", next(c for c in calls if c[0] == "rsync"))

    def test_push_retires_only_known_files_even_with_quoted_path(self):
        self.run_sync(False)

    def test_dry_run_cannot_create_or_retire_remote_files(self):
        self.run_sync(True)

    def test_failed_transfer_cannot_retire_remote_wrappers(self):
        self.run_sync(False, fail_transfer=True)


if __name__ == "__main__":
    unittest.main()
