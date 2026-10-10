#!/bin/bash
# Control: high-shot confirmation of the weighted-ADMM parity on the
# [[72,12,6]] code. Runs BP, plain ADMM, and weighted ADMM at the given
# p values. Shots and seed come from QUDEC_SHOTS/QUDEC_SEED so jobs fan
# out without repeating samples. QUDEC_OSD_LAM switches every decoder to
# the reference exhaustive OSD-CS(lambda) sweep.
#
# Usage:
#   sbatch hpc/run_phenom_confirm.slurm 0.005 0.010
#   sbatch hpc/run_phenom_confirm.slurm 0.020
#   QUDEC_SHOTS=2500 QUDEC_SEED=0 sbatch hpc/run_phenom_confirm.slurm 0.020
#   QUDEC_SHOTS=2500 QUDEC_SEED=1 sbatch hpc/run_phenom_confirm.slurm 0.020
#   QUDEC_SHOTS=2000 QUDEC_OSD_LAM=60 sbatch hpc/run_phenom_confirm.slurm 0.020
#
# Sizing from hpc/logs: 2000 shots at p = 0.020 take about 80 minutes
# (41 minutes per ADMM variant). One p at 10^4 shots needs the full
# 12-hour limit; split 10^4 shots as four 2500-shot seeds per p.

#SBATCH --job-name=qudec_conf
#SBATCH --output=hpc/logs/qudec_conf_%j.out
#SBATCH --error=hpc/logs/qudec_conf_%j.err
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

SHOTS="${QUDEC_SHOTS:-2000}"
SEED="${QUDEC_SEED:-0}"
OSD_LAM_ARGS=()
if [[ -n "${QUDEC_OSD_LAM:-}" ]]; then
    OSD_LAM_ARGS=(--osd-lam "${QUDEC_OSD_LAM}")
fi

EXTRA="${@}"
if [[ -z "${EXTRA}" ]]; then
    EXTRA="0.005 0.010"
fi

echo "=== python: ${PY} ==="
echo "=== shots: ${SHOTS} seed: ${SEED} osd-lam: ${QUDEC_OSD_LAM:-default} ==="
"${PY}" -u scripts/bench_phenom.py 72 6 "${SHOTS}" \
    --variants bp,admm,admm-w --seed "${SEED}" "${OSD_LAM_ARGS[@]}" ${EXTRA}
