#!/bin/bash
# Control: phase one validation, the gross-code LER sweep, and the
# reference bposd sweep for paper parity on one GPU node.
#
# Usage:
#   sbatch hpc/run_test_bench.slurm

#SBATCH --job-name=qudec_test
#SBATCH --output=hpc/logs/qudec_test_%j.out
#SBATCH --error=hpc/logs/qudec_test_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p hpc/logs

PY="python3"
for CAND in \
    "${HOME}/miniforge3/envs/qudec/bin/python" \
    "${HOME}/.conda/envs/qudec/bin/python"; do
    if [[ -x "${CAND}" ]]; then
        PY="${CAND}"
        break
    fi
done

echo "=== python: ${PY} ==="
"${PY}" -m unittest discover -s tests -v
"${PY}" -u scripts/bench_gross.py
"${PY}" -u scripts/bench_gross_ref.py
