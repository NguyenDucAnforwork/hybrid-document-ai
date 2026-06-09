#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORKSPACE="${DOCAI_DEMO_WORKSPACE:-${DOCAI_WORKSPACE:-${REPO_ROOT}/.demo-workspace}}"
PORT="${DOCAI_DEMO_PORT:-7861}"

if [[ -z "${HF_TOKEN:-}" && -f "${REPO_ROOT}/.env" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "${REPO_ROOT}/.env" | head -n 1 | cut -d= -f2- || true)"
  export HF_TOKEN
fi

if [[ ! -d "${WORKSPACE}" ]]; then
  echo "Workspace not found at ${WORKSPACE}. Run scripts/download_demo_assets.sh first." >&2
  exit 1
fi

export DOCAI_DEMO_WORKSPACE="${WORKSPACE}"
export DOCAI_DEMO_PORT="${PORT}"
export DOCAI_VLM_MODE="${DOCAI_VLM_MODE:-disabled}"

cd "${REPO_ROOT}"
echo "Launching Docker demo on http://localhost:${PORT}"
echo "Workspace: ${WORKSPACE}"
exec docker compose -f deploy/docker-compose.demo.yml up --build
