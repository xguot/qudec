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
# Historical runtimes predate corrected projection and closed boundaries.
# QUDEC_BATCH_SIZE bounds decoder memory independently of total shots.

#SBATCH --job-name=qudec_144
#SBATCH --output=hpc/logs/qudec_144_%j.out
#SBATCH --error=hpc/logs/qudec_144_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
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

# Corrected closed-memory protocol; historical log rates are not comparable.
SHOTS="${QUDEC_SHOTS:-300}"
SEED="${QUDEC_SEED:-0}"
DEC_ARGS=(--batch-size "${QUDEC_BATCH_SIZE:-64}"
          --admm-rho "${QUDEC_RHO:-2}"
          --admm-alpha "${QUDEC_ALPHA:-1.0}"
          --output "${QUDEC_OUTPUT:-results/phenom_144_${SLURM_JOB_ID:-local}.jsonl}")
if [[ -n "${QUDEC_OSD_LAM:-}" ]]; then
    DEC_ARGS+=(--osd-lam "${QUDEC_OSD_LAM}")
fi
if [[ -n "${QUDEC_P_MEAS:-}" ]]; then
    DEC_ARGS+=(--p-meas "${QUDEC_P_MEAS}")
fi
if [[ "${QUDEC_LEGACY_OPEN_BOUNDARY:-0}" == "1" ]]; then
    DEC_ARGS+=(--legacy-open-boundary)
fi
EXTRA=("$@")
if [[ ${#EXTRA[@]} -eq 0 ]]; then
    EXTRA=(0.005 0.010)
fi

echo "=== python: ${PY} ==="
echo "=== shots: ${SHOTS} seed: ${SEED} dec args: ${DEC_ARGS[*]} ==="
"${PY}" -u scripts/bench_phenom.py 144 6 "${SHOTS}" \
    --variants "${QUDEC_VARIANTS:-bp,admm,admm-w}" --seed "${SEED}" \
    "${DEC_ARGS[@]}" "${EXTRA[@]}"
