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

# Quote paths for the remote POSIX shell, including spaces and apostrophes.
# Keep a leading ~/ expandable for rsync's remote-shell path handling.
shell_quote() {
  local value=$1 prefix
  printf "'"
  while [[ "$value" == *"'"* ]]; do
    prefix=${value%%"'"*}
    printf '%s%s' "$prefix" "'\"'\"'"
    value=${value#*"'"}
  done
  printf "%s'" "$value"
}

rsync_remote_path() {
  case "$1" in
    '~') printf '~' ;;
    '~/'*) printf '~/%s' "$(shell_quote "${1#'~/'}")" ;;
    *) shell_quote "$1" ;;
  esac
}

REMOTE_ROOT_QUOTED="$(shell_quote "$REMOTE_PATH")"
REMOTE_RSYNC_ROOT="$(rsync_remote_path "$REMOTE_PATH")"

echo "=== Sync [${PROJECT_NAME}] -> ${REMOTE_HOST}:${REMOTE_PATH} ==="
echo "  Direction: ${DIRECTION}"
$DRY_RUN && echo "  Mode:      DRY RUN"

if [ "$DIRECTION" = "push" ]; then
  if $DRY_RUN; then
    echo "  Would create the remote hpc/logs directory."
  else
    ssh "$REMOTE_HOST" "sh -s -- ${REMOTE_ROOT_QUOTED}" <<'REMOTE'
set -eu
remote_root=$1
case "$remote_root" in
  '~') remote_root=$HOME ;;
  '~/'*) remote_root="$HOME/${remote_root#'~/'}" ;;
esac
mkdir -p -- "$remote_root/hpc/logs"
REMOTE
  fi

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
    "${ROOT_DIR}/" "${REMOTE_HOST}:${REMOTE_RSYNC_ROOT}/"

  # Retire only the replaced entry points after a successful code sync.
  # Never use --delete: prior campaign results and logs must remain intact.
  if $DRY_RUN; then
    echo "  Would retire these remote hpc/ files after a successful sync:"
    printf '    %s\n' run_test_bench.slurm run_tune.slurm run_phenom.slurm \
      run_phenom_144.slurm run_phenom_confirm.slurm \
      run_phenom_confirm_rho4.slurm run_circuit.slurm probe_cuda.slurm
  else
    ssh "$REMOTE_HOST" "sh -s -- ${REMOTE_ROOT_QUOTED}" <<'REMOTE'
set -eu
remote_root=$1
case "$remote_root" in
  '~') remote_root=$HOME ;;
  '~/'*) remote_root="$HOME/${remote_root#'~/'}" ;;
esac
for retired in run_test_bench.slurm run_tune.slurm run_phenom.slurm \
  run_phenom_144.slurm run_phenom_confirm.slurm \
  run_phenom_confirm_rho4.slurm run_circuit.slurm probe_cuda.slurm; do
  rm -f -- "$remote_root/hpc/$retired"
done
REMOTE
  fi

  if $DRY_RUN; then
    echo "  Push dry run complete; no remote files were created or retired."
  else
    echo "  Push complete; historical results and logs were preserved."
  fi
  echo "  On Rivanna, enter ${REMOTE_PATH} and submit:"
  echo "    sbatch hpc/rerun_all.slurm"
fi

if [ "$DIRECTION" = "pull" ]; then
  for dir in results hpc/logs; do
    if ssh "$REMOTE_HOST" "sh -s -- ${REMOTE_ROOT_QUOTED} $(shell_quote "$dir")" <<'REMOTE'
set -eu
remote_root=$1
case "$remote_root" in
  '~') remote_root=$HOME ;;
  '~/'*) remote_root="$HOME/${remote_root#'~/'}" ;;
esac
test -d "$remote_root/$2"
REMOTE
    then
      rsync "${RSYNC_FLAGS[@]}" \
        "${REMOTE_HOST}:$(rsync_remote_path "${REMOTE_PATH}/${dir}")/" "${ROOT_DIR}/${dir}/"
      echo "  ${dir}/ synced"
    else
      echo "  ${dir}/ not found on remote — skipped"
    fi
  done
  echo "  Pull complete."
fi
