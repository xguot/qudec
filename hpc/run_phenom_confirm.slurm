#!/bin/bash
# Control: high-shot confirmation of the low-rate weighted-ADMM edge on
# the [[72,12,6]] code. Runs BP, plain ADMM, and weighted ADMM at the
# given p values with 2000 shots each.
#
# Usage:
#   sbatch hpc/run_phenom_confirm.slurm 0.005 0.010
#   sbatch hpc/run_phenom_confirm.slurm 0.020

#SBATCH --job-name=qudec_conf
#SBATCH --output=hpc/logs/qudec_conf_%j.out
#SBATCH --error=hpc/logs/qudec_conf_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=6:00:00

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
"${PY}" -u scripts/bench_phenom.py 72 6 2000 --variants bp,admm,admm-w ${EXTRA}
