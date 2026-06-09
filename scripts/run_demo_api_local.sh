#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-pip}"
WORKSPACE="${DOCAI_WORKSPACE:-${REPO_ROOT}/.demo-workspace}"
PORT="${DOCAI_API_PORT:-8001}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

export DOCAI_WORKSPACE="${WORKSPACE}"
export DOCAI_VLM_MODE="${DOCAI_VLM_MODE:-disabled}"
export DOCAI_DB="${DOCAI_DB:-${WORKSPACE}/meta.db}"
export DOCAI_LAYOUTLMV3_MODE="${DOCAI_LAYOUTLMV3_MODE:-merchant}"
export DOCAI_LAYOUTLMV3_ONNX_PATH="${DOCAI_LAYOUTLMV3_ONNX_PATH:-${WORKSPACE}/models/layoutlmv3/model_fp32.onnx}"
export DOCAI_LAYOUTLMV3_DIR="${DOCAI_LAYOUTLMV3_DIR:-${WORKSPACE}/models/layoutlmv3/model}"

if [[ ! -d "${WORKSPACE}" ]]; then
  echo "Workspace not found at ${WORKSPACE}. Run scripts/download_demo_assets.sh first." >&2
  exit 1
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "Installing API dependencies..."
  "${PIP_BIN}" install -r "${REPO_ROOT}/requirements.txt"
fi

cd "${REPO_ROOT}"
echo "Launching API demo on http://localhost:${PORT}"
echo "Swagger docs: http://localhost:${PORT}/docs"
exec "${PYTHON_BIN}" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
