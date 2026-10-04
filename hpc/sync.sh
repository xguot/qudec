#!/bin/bash
# Sync qudec to/from Rivanna — selective, only what the jobs need.
#
#   bash hpc/sync.sh [push|pull] [--dry-run]
#
# Set HPC_REMOTE to override the SSH host (default: rivanna) and
# HPC_PATH to override the remote directory (default: ~/scratch/qudec).

set -euo pipefail

REMOTE_HOST="${HPC_REMOTE:-rivanna}"
REMOTE_PATH="${HPC_PATH:-~/scratch/qudec}"
DRY_RUN=false
DIRECTION="push"

for arg in "$@"; do
  case "$arg" in
    push)      DIRECTION="push" ;;
    pull)      DIRECTION="pull" ;;
    --dry-run) DRY_RUN=true ;;
    *)         echo "Unknown: $arg"
               echo "Usage: bash hpc/sync.sh [push|pull] [--dry-run]"
               exit 1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_NAME="$(basename "$ROOT_DIR")"

RSYNC_FLAGS=(-avz --progress)
$DRY_RUN && RSYNC_FLAGS+=(--dry-run)

echo "=== Sync [${PROJECT_NAME}] -> ${REMOTE_HOST}:${REMOTE_PATH} ==="
echo "  Direction: ${DIRECTION}"
$DRY_RUN && echo "  Mode:      DRY RUN"

if [ "$DIRECTION" = "push" ]; then
  ssh "$REMOTE_HOST" "mkdir -p ${REMOTE_PATH}/hpc/logs"

  # Every directory along a path needs its own include rule, otherwise
  # the trailing exclude-all prunes it and rsync never descends into it.
  rsync "${RSYNC_FLAGS[@]}" \
    --exclude='__pycache__/'    \
    --exclude='*.pyc'           \
    --exclude='.git/'           \
    --exclude='.codewhale/'     \
    --exclude='*.pdf'           \
    --exclude='/hpc/logs/'      \
    --exclude='/results/'       \
    --exclude='/data/'          \
    --include='/src/***'        \
    --include='/scripts/***'    \
    --include='/tests/***'      \
    --include='/hpc/***'        \
    --include='/pyproject.toml' \
    --include='/README.md'      \
    --exclude='*'               \
    "${ROOT_DIR}/" "${REMOTE_HOST}:${REMOTE_PATH}/"

  echo "  Push complete."
  echo "  Tests and code-capacity sweep:"
  echo "    ssh ${REMOTE_HOST} && cd ${REMOTE_PATH} && sbatch hpc/run_test_bench.slurm"
  echo "  Phenomenological sweep:"
  echo "    ssh ${REMOTE_HOST} && cd ${REMOTE_PATH} && sbatch hpc/run_phenom.slurm"
fi

if [ "$DIRECTION" = "pull" ]; then
  for dir in results hpc/logs; do
    if ssh "$REMOTE_HOST" "test -d ${REMOTE_PATH}/${dir}"; then
      rsync "${RSYNC_FLAGS[@]}" \
        "${REMOTE_HOST}:${REMOTE_PATH}/${dir}/" "${ROOT_DIR}/${dir}/"
      echo "  ${dir}/ synced"
    else
      echo "  ${dir}/ not found on remote — skipped"
    fi
  done
  echo "  Pull complete."
fi
