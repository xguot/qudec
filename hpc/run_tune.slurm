#!/bin/bash
# Control: ADMM hyperparameter tuning under phenomenological noise and
# the OSD-CS(lambda) reference-parity comparison on the [[72,12,6]]
# code. Writes paired JSONL outcomes to a separate results file per job.
#
# Usage:
#   sbatch hpc/run_tune.slurm                 # p = 0.020, default grid
#   sbatch hpc/run_tune.slurm 72 6 300 0.020 0.010     # selected p values
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
EXTRA=("${@:4}")
if [[ ${#EXTRA[@]} -eq 0 ]]; then
    EXTRA=(0.020)
fi
DEC_ARGS=(--batch-size "${QUDEC_BATCH_SIZE:-64}"
          --seed "${QUDEC_SEED:-0}"
          --rhos "${QUDEC_RHOS:-1,2,4}"
          --alphas "${QUDEC_ALPHAS:-1.0,1.5,1.8}"
          --lam "${QUDEC_OSD_LAM:-60}"
          --output "${QUDEC_OUTPUT:-results/tune_${CODE}_${SLURM_JOB_ID:-local}.jsonl}")
if [[ -n "${QUDEC_P_MEAS:-}" ]]; then
    DEC_ARGS+=(--p-meas "${QUDEC_P_MEAS}")
fi
if [[ "${QUDEC_LEGACY_OPEN_BOUNDARY:-0}" == "1" ]]; then
    DEC_ARGS+=(--legacy-open-boundary)
fi
"${PY}" -u scripts/bench_tune.py "${CODE}" "${ROUNDS}" "${SHOTS}" \
    "${DEC_ARGS[@]}" "${EXTRA[@]}"
