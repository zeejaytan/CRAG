#!/usr/bin/env bash
# Rsync a relative path from the Spartan CRAG tree to a local directory.
# Usage: ./scripts/remote/fetch_artifacts.sh <remote-relpath> [local-dir]
set -euo pipefail

SPARTAN_HOST="${SPARTAN_HOST:-spartan}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/gpfs/projects/punim2657/CRAG}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <remote-relpath> [local-dir]" >&2
  echo "  remote-relpath: path relative to REMOTE_ROOT (e.g. logs/STG_1_BB/run)" >&2
  echo "  local-dir:      default ./artifacts/" >&2
  exit 1
fi

REMOTE_REL="${1%/}"
LOCAL_DIR="${2:-./artifacts}"

mkdir -p "$LOCAL_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -avz --progress \
    "${SPARTAN_HOST}:${REMOTE_ROOT}/${REMOTE_REL}" \
    "${LOCAL_DIR}/"
else
  # Git Bash on Windows has no rsync; scp -r (Windows OpenSSH) is a fine
  # fallback for the small artifacts this script is meant for.
  echo "rsync not found; falling back to scp -r" >&2
  scp -r \
    "${SPARTAN_HOST}:${REMOTE_ROOT}/${REMOTE_REL}" \
    "${LOCAL_DIR}/"
fi

echo "Fetched → ${LOCAL_DIR}/$(basename "$REMOTE_REL")"
