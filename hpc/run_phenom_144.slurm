#!/bin/bash
# Control: code-size dimension on the [[144,12,12]] gross code. Runs BP,
# plain ADMM, and weighted ADMM at the given p values. Shots and seed
# come from QUDEC_SHOTS/QUDEC_SEED.
#
# Usage:
#   sbatch hpc/run_phenom_144.slurm                          # 300 shots
#   QUDEC_SHOTS=2500 QUDEC_SEED=0 sbatch hpc/run_phenom_144.slurm 0.005
#   QUDEC_SHOTS=2500 QUDEC_SEED=1 sbatch hpc/run_phenom_144.slurm 0.005
#
# Sizing from hpc/logs: 300 shots at p = 0.010 take about 13 minutes
# per ADMM variant, so 2500 shots fit the 12-hour limit per p.

#SBATCH --job-name=qudec_144
#SBATCH --output=hpc/logs/qudec_144_%j.out
#SBATCH --error=hpc/logs/qudec_144_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00

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

SHOTS="${QUDEC_SHOTS:-300}"
SEED="${QUDEC_SEED:-0}"

EXTRA="${@}"
if [[ -z "${EXTRA}" ]]; then
    EXTRA="0.005 0.010"
fi

echo "=== python: ${PY} ==="
echo "=== shots: ${SHOTS} seed: ${SEED} ==="
"${PY}" -u scripts/bench_phenom.py 144 6 "${SHOTS}" \
    --variants bp,admm,admm-w --seed "${SEED}" ${EXTRA}
