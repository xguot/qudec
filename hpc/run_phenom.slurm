#!/bin/bash
# Control: phenomenological-noise sweep comparing BP+OSD and ADMM+OSD
# on one bivariate bicycle code. The ADMM solver runs on the GPU.
# Submit one job per code so each stays short in the queue.
#
# Usage:
#   sbatch hpc/run_phenom.slurm 72
#   sbatch hpc/run_phenom.slurm 144

#SBATCH --job-name=qudec_phenom
#SBATCH --output=hpc/logs/qudec_phenom_%j.out
#SBATCH --error=hpc/logs/qudec_phenom_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00

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

CODE="${1:-72}"

echo "=== python: ${PY} ==="
"${PY}" -m unittest tests.test_phenom -v
"${PY}" -u scripts/bench_phenom.py "${CODE}" 6 400
