"""Bootstrap a local demo workspace from Hugging Face artifacts.

Downloads the assets needed for local or Docker-based demos:
- production KIE checkpoint
- production document-type router checkpoint
- optional LayoutLMv3 ONNX/tokenizer assets
- example images for Gradio/API smoke tests
- optional full demo dataset snapshot

It also writes a local registry.yaml compatible with the Docker API stack.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


MODEL_REPO = "banhchungtuongot/hybrid-docai-kie"
DATASET_REPO = "banhchungtuongot/hybrid-docai-receipts"

MODEL_FILES = {
    "models/kie/v4/model.joblib": "models/kie/v4/model.joblib",
    "models/doctype/v3/model.joblib": "models/doctype/v3/model.joblib",
}

LAYOUTLMV3_FILES = {
    "models/layoutlmv3/model/config.json": "models/layoutlmv3/model/config.json",
    "models/layoutlmv3/model/merges.txt": "models/layoutlmv3/model/merges.txt",
    "models/layoutlmv3/model/preprocessor_config.json": "models/layoutlmv3/model/preprocessor_config.json",
    "models/layoutlmv3/model/special_tokens_map.json": "models/layoutlmv3/model/special_tokens_map.json",
    "models/layoutlmv3/model/tokenizer.json": "models/layoutlmv3/model/tokenizer.json",
    "models/layoutlmv3/model/tokenizer_config.json": "models/layoutlmv3/model/tokenizer_config.json",
    "models/layoutlmv3/model/training_args.bin": "models/layoutlmv3/model/training_args.bin",
    "models/layoutlmv3/model/vocab.json": "models/layoutlmv3/model/vocab.json",
    "models/layoutlmv3/model_fp32.onnx": "models/layoutlmv3/model_fp32.onnx",
    "models/layoutlmv3/model_int8.onnx": "models/layoutlmv3/model_int8.onnx",
    "models/layoutlmv3/metrics.json": "models/layoutlmv3/metrics.json",
}

EXAMPLE_FILES = {
    "examples/receipt_sample.png": "images/rcpt_0000.png",
    "examples/payment_order_sample.png": "payment_orders/images/po_0000.png",
    "examples/statement_sample.png": "statements_test_hard/images/stmt_0000.png",
}


def _copy_from_hf(repo_id: str, repo_type: str, remote_path: str, local_path: Path, token: str | None) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=remote_path,
        token=token,
    )
    shutil.copy2(cached, local_path)


def _write_registry(workspace: Path) -> None:
    registry = """kie:
  active: v4
  production: v4
  versions:
    v4:
      path: /workspace/models/kie/v4/model.joblib
      stage: production
doctype:
  active: v3
  production: v3
  versions:
    v3:
      path: /workspace/models/doctype/v3/model.joblib
      stage: production
"""
    target = workspace / "models" / "registry.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(registry, encoding="utf-8")


def _download_dataset_snapshot(workspace: Path, token: str | None) -> Path:
    target_dir = workspace / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=token,
        local_dir_use_symlinks=False,
    )
    return target_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.environ.get("DOCAI_WORKSPACE", ".demo-workspace"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--skip-examples", action="store_true")
    ap.add_argument("--skip-layoutlmv3", action="store_true")
    ap.add_argument("--download-full-dataset", action="store_true")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    for local_rel, remote_rel in MODEL_FILES.items():
        _copy_from_hf(MODEL_REPO, "model", remote_rel, workspace / local_rel, args.hf_token)

    if not args.skip_layoutlmv3:
        for local_rel, remote_rel in LAYOUTLMV3_FILES.items():
            _copy_from_hf(MODEL_REPO, "model", remote_rel, workspace / local_rel, args.hf_token)

    _write_registry(workspace)

    if not args.skip_examples:
        for local_rel, remote_rel in EXAMPLE_FILES.items():
            _copy_from_hf(DATASET_REPO, "dataset", remote_rel, workspace / local_rel, args.hf_token)

    if args.download_full_dataset:
        ds_dir = _download_dataset_snapshot(workspace, args.hf_token)
        print(f"Full dataset snapshot ready at: {ds_dir}")

    print(f"Demo workspace ready at: {workspace}")
    print(f"Examples: {workspace / 'examples'}")
    print(f"LayoutLMv3 ONNX: {workspace / 'models' / 'layoutlmv3' / 'model_fp32.onnx'}")
    print("Run Gradio demo with:")
    print(f"  DOCAI_WORKSPACE={workspace}")
    print("  python space/app.py")


if __name__ == "__main__":
    main()
