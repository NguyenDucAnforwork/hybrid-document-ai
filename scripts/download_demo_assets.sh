#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORKSPACE="${DOCAI_WORKSPACE:-${REPO_ROOT}/.demo-workspace}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DOWNLOAD_FULL_DATASET="${DOWNLOAD_FULL_DATASET:-0}"
SKIP_LAYOUTLMV3="${SKIP_LAYOUTLMV3:-0}"

if [[ -z "${HF_TOKEN:-}" && -f "${REPO_ROOT}/.env" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "${REPO_ROOT}/.env" | head -n 1 | cut -d= -f2- || true)"
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it or add HF_TOKEN=... to ${REPO_ROOT}/.env" >&2
  exit 1
fi

ARGS=(--workspace "${WORKSPACE}" --hf-token "${HF_TOKEN}")

if [[ "${SKIP_LAYOUTLMV3}" == "1" ]]; then
  ARGS+=(--skip-layoutlmv3)
fi

if [[ "${DOWNLOAD_FULL_DATASET}" == "1" ]]; then
  ARGS+=(--download-full-dataset)
fi

echo "Preparing demo workspace at ${WORKSPACE}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/setup_demo_from_hf.py" "${ARGS[@]}"

echo
echo "Done."
echo "Workspace: ${WORKSPACE}"
echo "Examples:  ${WORKSPACE}/examples"
echo "Run local demo: DOCAI_WORKSPACE=${WORKSPACE} ${REPO_ROOT}/scripts/run_demo_local.sh"
echo "Run Docker demo: DOCAI_DEMO_WORKSPACE=${WORKSPACE} ${REPO_ROOT}/scripts/run_demo_docker.sh"
