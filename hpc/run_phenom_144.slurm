#!/bin/bash
# Control: code-size dimension on the [[144,12,12]] gross code at the
# low-rate regime where the weighted-ADMM edge showed up. Runs BP, plain
# ADMM, and weighted ADMM at the given p values with 300 shots each.
#
# Usage:
#   sbatch hpc/run_phenom_144.slurm

#SBATCH --job-name=qudec_144
#SBATCH --output=hpc/logs/qudec_144_%j.out
#SBATCH --error=hpc/logs/qudec_144_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00

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

EXTRA="${@}"
if [[ -z "${EXTRA}" ]]; then
    EXTRA="0.005 0.010"
fi

echo "=== python: ${PY} ==="
"${PY}" -u scripts/bench_phenom.py 144 6 300 --variants bp,admm,admm-w ${EXTRA}
