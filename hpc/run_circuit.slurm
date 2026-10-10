#!/bin/bash
# Control: circuit-level noise benchmark through stim. Runs the smoke
# validation first (noiseless detectors, dem shape, decoder sanity) and
# aborts on failure, then benches the [[72,12,6]] memory experiment.
#
# Usage:
#   sbatch hpc/run_circuit.slurm                     # defaults below
#   sbatch hpc/run_circuit.slurm 144 6 2000 0.010 --basis Z

#SBATCH --job-name=qudec_circuit
#SBATCH --output=hpc/logs/qudec_circuit_%j.out
#SBATCH --error=hpc/logs/qudec_circuit_%j.err
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
"${PY}" -u scripts/smoke_circuit.py
CODE="${1:-72}"
ROUNDS="${2:-6}"
SHOTS="${3:-1000}"
EXTRA="${@:4}"
if [[ -z "${EXTRA}" ]]; then
    EXTRA="0.020 --basis Z"
fi
"${PY}" -u scripts/bench_circuit.py "${CODE}" "${ROUNDS}" "${SHOTS}" ${EXTRA}
