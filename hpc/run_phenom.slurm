#!/bin/bash
# Control: phenomenological-noise sweep comparing BP+OSD and ADMM+OSD
# on one bivariate bicycle code. The ADMM solver runs on the GPU.
# Submit one job per code so each stays short in the queue.
#
# Usage:
#   sbatch hpc/run_phenom.slurm 72              # full p sweep
#   sbatch hpc/run_phenom.slurm 72 0.02 0.03    # selected p values

#SBATCH --job-name=qudec_phenom
#SBATCH --output=hpc/logs/qudec_phenom_%j.out
#SBATCH --error=hpc/logs/qudec_phenom_%j.err
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

CODE="${1:-72}"
EXTRA=("${@:2}")
DEC_ARGS=(--batch-size "${QUDEC_BATCH_SIZE:-64}"
          --seed "${QUDEC_SEED:-0}"
          --admm-rho "${QUDEC_RHO:-2}"
          --admm-alpha "${QUDEC_ALPHA:-1.0}"
          --output "${QUDEC_OUTPUT:-results/phenom_${CODE}_${SLURM_JOB_ID:-local}.jsonl}")
if [[ -n "${QUDEC_OSD_LAM:-}" ]]; then
    DEC_ARGS+=(--osd-lam "${QUDEC_OSD_LAM}")
fi
if [[ -n "${QUDEC_P_MEAS:-}" ]]; then
    DEC_ARGS+=(--p-meas "${QUDEC_P_MEAS}")
fi
if [[ "${QUDEC_LEGACY_OPEN_BOUNDARY:-0}" == "1" ]]; then
    DEC_ARGS+=(--legacy-open-boundary)
fi

echo "=== python: ${PY} ==="
"${PY}" -m unittest discover -s tests -p test_phenom.py -v
"${PY}" -u scripts/bench_phenom.py "${CODE}" 6 "${QUDEC_SHOTS:-300}" \
    --variants "${QUDEC_VARIANTS:-bp,admm,admm-w,admm-ldr}" \
    "${DEC_ARGS[@]}" "${EXTRA[@]}"
