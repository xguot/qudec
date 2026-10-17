#!/bin/bash
# setup.sh — One-time environment setup for qudec on Rivanna HPC
#
# Usage (on Rivanna login node, after `bash hpc/sync.sh push`):
#   bash hpc/setup.sh
#
# Creates a conda environment with PyTorch plus the decoder dependencies,
# the optional ldpc/bposd reference stack, and stim for the circuit-level
# pipeline.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

module purge
module load miniforge

ENV_NAME="qudec"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Conda environment '${ENV_NAME}' already exists."
else
    echo "=== Creating conda environment '${ENV_NAME}' ==="
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

echo "=== Installing Python dependencies ==="
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install -e .
conda run -n "${ENV_NAME}" pip install -e ".[reference,circuit]"

echo ""
echo "=== Setup complete ==="
echo "Complete corrected campaign:    sbatch hpc/rerun_all.slurm"
echo "Plan without submitting:        conda run -n ${ENV_NAME} python scripts/rerun_all.py plan"
echo "Resume a campaign:              sbatch hpc/rerun_all.slurm resume /absolute/campaign/path"
echo "Results and worker logs:        results/reruns/<initial_job_id>/"
echo "Monitor:                        squeue -u \$USER"
