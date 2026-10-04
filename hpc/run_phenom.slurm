#!/bin/bash
# Control: phenomenological-noise sweep comparing BP+OSD and ADMM+OSD
# on the bivariate bicycle codes. The ADMM solver runs on the GPU.
#
# Usage:
#   sbatch hpc/run_phenom.slurm

#SBATCH --job-name=qudec_phenom
#SBATCH --output=hpc/logs/qudec_phenom_%j.out
#SBATCH --error=hpc/logs/qudec_phenom_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
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

echo "=== python: ${PY} ==="
"${PY}" -m unittest tests.test_phenom -v
"${PY}" -u scripts/bench_phenom.py 72 6 400
"${PY}" -u scripts/bench_phenom.py 144 6 400
