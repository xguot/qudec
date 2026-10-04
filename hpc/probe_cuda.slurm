#!/bin/bash
# Control: probe the GPU node for torch/cuda status. Tiny job, fast queue.
#
# Usage:
#   sbatch hpc/probe_cuda.slurm

#SBATCH --job-name=qudec_probe
#SBATCH --output=hpc/logs/qudec_probe_%j.out
#SBATCH --error=hpc/logs/qudec_probe_%j.err
#SBATCH --account=tongx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0:10:00

set -euo pipefail

nvidia-smi
echo "---"

PY="python3"
for CAND in \
    "${HOME}/miniforge3/envs/qudec/bin/python" \
    "${HOME}/.conda/envs/qudec/bin/python"; do
    if [[ -x "${CAND}" ]]; then
        PY="${CAND}"
        break
    fi
done

"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
