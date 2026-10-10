#!/bin/bash
# Control: ADMM hyperparameter tuning under phenomenological noise and
# the OSD-CS(lambda) reference-parity comparison on the [[72,12,6]]
# code. Writes one CSV row per (variant, p) to results/tune_72.csv.
#
# Usage:
#   sbatch hpc/run_tune.slurm                 # p = 0.020, default grid
#   sbatch hpc/run_tune.slurm 0.020 0.010     # selected p values
#   sbatch hpc/run_tune.slurm 144 6 300 0.010

#SBATCH --job-name=qudec_tune
#SBATCH --output=hpc/logs/qudec_tune_%j.out
#SBATCH --error=hpc/logs/qudec_tune_%j.err
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
mkdir -p hpc/logs results

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
CODE="${1:-72}"
ROUNDS="${2:-6}"
SHOTS="${3:-300}"
EXTRA="${@:4}"
if [[ -z "${EXTRA}" ]]; then
    EXTRA="0.020"
fi
"${PY}" -u scripts/bench_tune.py "${CODE}" "${ROUNDS}" "${SHOTS}" ${EXTRA}
