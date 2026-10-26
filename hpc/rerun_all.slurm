#!/bin/bash
# Single Slurm entry point for qudec campaigns. Run from the synced repo root:
#   sbatch hpc/rerun_all.slurm                          # default 10k-shot campaign (with greedy + CS(60) replication)
#   sbatch hpc/rerun_all.slurm high_stat                # 100k-shot high-statistics confirmation + replication + reference
#   sbatch hpc/rerun_all.slurm followup                 # 25k-shot confirmation + 10k-shot greedy/CS(60) replication
#   sbatch hpc/rerun_all.slurm resume /abs/path/to/run  # resume an existing campaign
#
# Environment overrides (e.g. QUDEC_CONFIRM_SEEDS=40, QUDEC_MAX_PARALLEL=8)
# take precedence over preset defaults. The launcher validates CUDA/tests,
# freezes a source snapshot, and submits dependent tune/select/work/report jobs.
#SBATCH --job-name=qudec_rerun
#SBATCH --output=hpc/logs/qudec_rerun_%j.out
#SBATCH --error=hpc/logs/qudec_rerun_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00

set -euo pipefail
STAGE="${1:-launch}"
CAMPAIGN="${2:-}"
if [[ "${STAGE}" == "high_stat" || "${STAGE}" == "high-stat" || "${STAGE}" == "100k" || "${STAGE}" == "followup" || "${STAGE}" == "default" ]]; then
    export QUDEC_PRESET="${STAGE}"
    STAGE="launch"
fi
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ -n "${CAMPAIGN}" && -d "${CAMPAIGN}/source" ]]; then
    CAMPAIGN="$(cd "${CAMPAIGN}" && pwd)"
    ROOT_DIR="${CAMPAIGN}/source"
fi
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
PY="${QUDEC_PYTHON:-}"
if [[ -z "${PY}" ]]; then
    for CAND in "${HOME}/miniforge3/envs/qudec/bin/python" "${HOME}/.conda/envs/qudec/bin/python"; do
        if [[ -x "${CAND}" ]]; then
            PY="${CAND}"
            break
        fi
    done
fi
PY="${PY:-python3}"
# Children change into the source snapshot, so retain an absolute runtime.
PY="$("${PY}" -c 'import sys; print(sys.executable)')"
export QUDEC_PYTHON="${PY}"
ARGS=("${STAGE}")
if [[ -n "${CAMPAIGN}" ]]; then ARGS+=("${CAMPAIGN}"); fi
exec "${PY}" -u scripts/rerun_all.py "${ARGS[@]}"
