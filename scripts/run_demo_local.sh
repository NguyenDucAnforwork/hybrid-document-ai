#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-pip}"
WORKSPACE="${DOCAI_WORKSPACE:-${REPO_ROOT}/.demo-workspace}"
PORT="${GRADIO_SERVER_PORT:-7861}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
LAYOUTLMV3_MODE="${DOCAI_LAYOUTLMV3_MODE:-merchant}"
LAYOUTLMV3_ONNX_PATH="${DOCAI_LAYOUTLMV3_ONNX_PATH:-${WORKSPACE}/models/layoutlmv3/model_fp32.onnx}"
LAYOUTLMV3_DIR="${DOCAI_LAYOUTLMV3_DIR:-${WORKSPACE}/models/layoutlmv3/model}"

export DOCAI_WORKSPACE="${WORKSPACE}"
export DOCAI_EXAMPLES_DIR="${DOCAI_EXAMPLES_DIR:-${WORKSPACE}/examples}"
export DOCAI_VLM_MODE="${DOCAI_VLM_MODE:-disabled}"
export DOCAI_LAYOUTLMV3_MODE="${LAYOUTLMV3_MODE}"
export DOCAI_LAYOUTLMV3_ONNX_PATH="${LAYOUTLMV3_ONNX_PATH}"
export DOCAI_LAYOUTLMV3_DIR="${LAYOUTLMV3_DIR}"
export GRADIO_SERVER_PORT="${PORT}"

if [[ ! -d "${WORKSPACE}" ]]; then
  echo "Workspace not found at ${WORKSPACE}. Run scripts/download_demo_assets.sh first." >&2
  exit 1
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "Installing demo dependencies..."
  "${PIP_BIN}" install -r "${REPO_ROOT}/space/requirements.txt"
fi

cd "${REPO_ROOT}"
echo "Launching Gradio demo on http://localhost:${PORT}"
echo "Workspace: ${WORKSPACE}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/space/app.py"
