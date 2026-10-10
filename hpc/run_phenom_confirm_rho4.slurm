#!/bin/bash
# Control: tuned confirmation of the plain-ADMM deficit at rho = 4 on
# the [[72,12,6]] code. Runs BP, plain ADMM, and weighted ADMM, all
# ADMM variants with rho = 4 per the tuning sweep, at the given p
# values. QUDEC_SHOTS/QUDEC_SEED fan the job out without repeating
# samples; QUDEC_OSD_LAM switches to the reference OSD-CS(lambda) sweep.
#
# Usage:
#   sbatch hpc/run_phenom_confirm_rho4.slurm                # p = 0.020
#   QUDEC_SEED=0 sbatch hpc/run_phenom_confirm_rho4.slurm 0.020
#   QUDEC_SEED=1 sbatch hpc/run_phenom_confirm_rho4.slurm 0.020
#   QUDEC_SHOTS=2000 sbatch hpc/run_phenom_confirm_rho4.slurm 0.010 0.020
#
# Sizing from hpc/logs: 2000 shots at p = 0.020 take about 80 minutes
# (41 minutes per ADMM variant).

#SBATCH --job-name=qudec_conf4
#SBATCH --output=hpc/logs/qudec_conf4_%j.out
#SBATCH --error=hpc/logs/qudec_conf4_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=8:00:00

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

SHOTS="${QUDEC_SHOTS:-2000}"
SEED="${QUDEC_SEED:-0}"
DEC_ARGS=(--admm-rho 4)
if [[ -n "${QUDEC_OSD_LAM:-}" ]]; then
    DEC_ARGS+=(--osd-lam "${QUDEC_OSD_LAM}")
fi

EXTRA="${@}"
if [[ -z "${EXTRA}" ]]; then
    EXTRA="0.020"
fi

echo "=== python: ${PY} ==="
echo "=== shots: ${SHOTS} seed: ${SEED} dec args: ${DEC_ARGS[*]} ==="
"${PY}" -u scripts/bench_phenom.py 72 6 "${SHOTS}" \
    --variants bp,admm,admm-w --seed "${SEED}" "${DEC_ARGS[@]}" ${EXTRA}
