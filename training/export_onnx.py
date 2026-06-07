"""Export fine-tuned LayoutLMv3 to ONNX and apply INT8 dynamic quantization.

Pipeline:
  PyTorch (GPU)  →  ONNX FP32 (CPU)  →  ONNX INT8 (CPU, dynamic quant)

INT8 dynamic quantization: quantizes Linear layer weights to INT8 at export time;
activations are quantized dynamically at runtime. No calibration dataset needed.
Typical gain: 2-4x speedup on CPU, ~75% model size reduction, <2% F1 drop.

Output:
  $DOCAI_WORKSPACE/models/layoutlmv3/model_fp32.onnx  (~500MB)
  $DOCAI_WORKSPACE/models/layoutlmv3/model_int8.onnx  (~125MB)
  docs/logs/onnx_benchmark_TIMESTAMP.md

Usage:
  python training/export_onnx.py
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WS = os.environ.get("DOCAI_WORKSPACE", "/workspace/docai-ws")

LABEL2ID = {"O": 0, "B-MERCHANT": 1, "I-MERCHANT": 2,
            "B-DATE": 3, "I-DATE": 4, "B-TOTAL": 5, "I-TOTAL": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
ENTITY_MAP = {"MERCHANT": "merchant_name", "DATE": "date", "TOTAL": "total_amount"}


# ── Wrapper: expose only the 4 required inputs to ONNX ───────────────────────

class LayoutLMv3Wrapper(torch.nn.Module):
    """Thin wrapper so torch.onnx.export sees a clean (input_ids, attention_mask,
    bbox, pixel_values) → logits interface, hiding optional kwargs that confuse
    the ONNX tracer."""

    def __init__(self, model: LayoutLMv3ForTokenClassification):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, S)  Long
        attention_mask: torch.Tensor,     # (B, S)  Long
        bbox: torch.Tensor,               # (B, S, 4)  Long  0-1000
        pixel_values: torch.Tensor,       # (B, 3, 224, 224)  Float
    ) -> torch.Tensor:                    # (B, S, num_labels)  Float
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        ).logits


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_to_onnx(model_dir: str, onnx_path: str, seq_len: int = 512) -> None:
    print(f"Loading model from {model_dir} ...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()
    wrapper = LayoutLMv3Wrapper(model)

    # Dummy inputs (batch=1, seq=seq_len, image=224×224)
    B, S = 1, seq_len
    dummy = (
        torch.zeros(B, S, dtype=torch.long),                  # input_ids
        torch.ones(B, S, dtype=torch.long),                   # attention_mask
        torch.zeros(B, S, 4, dtype=torch.long),               # bbox
        torch.zeros(B, 3, 224, 224, dtype=torch.float32),     # pixel_values
    )

    print(f"Exporting to ONNX (opset 14) → {onnx_path} ...")
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input_ids", "attention_mask", "bbox", "pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "bbox":           {0: "batch", 1: "seq_len"},
            "pixel_values":   {0: "batch"},
            "logits":         {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,   # fuse constant sub-graphs (free speedup)
        verbose=False,
    )
    size_mb = Path(onnx_path).stat().st_size / 1024 / 1024
    print(f"  → {onnx_path}  ({size_mb:.0f} MB)")


# ── INT8 dynamic quantization ─────────────────────────────────────────────────

def quantize_int8(fp32_path: str, int8_path: str) -> None:
    """Apply dynamic INT8 quantization to all MatMul/Gemm ops.

    Dynamic quant: weights are pre-quantized to INT8 at export time;
    activations are quantized on-the-fly at runtime.  No calibration set needed.
    Per-channel weight quantization (per_channel=True) gives better accuracy
    than per-tensor at small cost.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing (INT8 dynamic, per-channel) → {int8_path} ...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=True,           # per-channel weight quant (better accuracy)
        reduce_range=False,         # full INT8 range (-128..127)
    )
    size_mb = Path(int8_path).stat().st_size / 1024 / 1024
    print(f"  → {int8_path}  ({size_mb:.0f} MB)")


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_pytorch(model, inputs_pt: dict, device: str) -> np.ndarray:
    model_d = model.to(device)
    inputs_d = {k: v.to(device) for k, v in inputs_pt.items()}
    with torch.no_grad():
        logits = LayoutLMv3Wrapper(model_d)(
            inputs_d["input_ids"], inputs_d["attention_mask"],
            inputs_d["bbox"], inputs_d["pixel_values"],
        )
    return logits.cpu().numpy()


def build_ort_session(onnx_path: str) -> "ort.InferenceSession":
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # CPU only (GPU ORT session needs onnxruntime-gpu package)
    return ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def run_ort(session, inputs_pt: dict) -> np.ndarray:
    feed = {
        "input_ids":      inputs_pt["input_ids"].numpy().astype(np.int64),
        "attention_mask": inputs_pt["attention_mask"].numpy().astype(np.int64),
        "bbox":           inputs_pt["bbox"].numpy().astype(np.int64),
        "pixel_values":   inputs_pt["pixel_values"].numpy().astype(np.float32),
    }
    return session.run(["logits"], feed)[0]


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_latency(fn, inputs, warmup: int = 3, runs: int = 30) -> dict:
    for _ in range(warmup):
        fn(inputs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(inputs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms":  round(sum(times) / len(times), 2),
        "p50_ms":   round(times[len(times) // 2], 2),
        "p95_ms":   round(times[int(len(times) * 0.95)], 2),
        "min_ms":   round(times[0], 2),
    }


# ── F1 sanity check ───────────────────────────────────────────────────────────

def decode_logits(logits_np: np.ndarray, word_ids: list[int | None]) -> dict[str, list[str]]:
    preds = np.argmax(logits_np, axis=2).squeeze(0).tolist()
    entity: dict[str, list[str]] = {f: [] for f in ENTITY_MAP.values()}
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            continue
        prev_wid = wid
        label = ID2LABEL.get(preds[i], "O")
        for ent, field in ENTITY_MAP.items():
            if ent in label:
                entity[field].append(str(wid))   # placeholder; text not needed for F1 check
    return entity


def token_f1(pred_logits: np.ndarray, gold_labels: np.ndarray) -> float:
    """Token-level F1 ignoring padding (-100)."""
    preds = np.argmax(pred_logits, axis=-1).flatten()
    gold = gold_labels.flatten()
    mask = gold != -100
    preds, gold = preds[mask], gold[mask]
    tp = ((preds != 0) & (preds == gold)).sum()
    fp = ((preds != 0) & (preds != gold)).sum()
    fn = ((gold != 0) & (preds != gold)).sum()
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=f"{WS}/models/layoutlmv3/model")
    ap.add_argument("--train-data", default=f"{WS}/data/sroie/train/labels.json")
    ap.add_argument("--train-img-dir", default=f"{WS}/sroie_src/data/img")
    ap.add_argument("--out-dir", default=f"{WS}/models/layoutlmv3")
    ap.add_argument("--out-log", default="docs/logs")
    ap.add_argument("--bench-runs", type=int, default=30)
    ap.add_argument("--val-samples", type=int, default=55)
    args = ap.parse_args()

    fp32_path = str(Path(args.out_dir) / "model_fp32.onnx")
    int8_path = str(Path(args.out_dir) / "model_int8.onnx")

    # ── 1. Export FP32 ──
    export_to_onnx(args.model_dir, fp32_path)

    # ── 2. Quantize INT8 ──
    quantize_int8(fp32_path, int8_path)

    # ── 3. Build a single representative input for latency benchmark ──
    print("\nBuilding benchmark input (seq_len=512) ...")
    processor = AutoProcessor.from_pretrained(args.model_dir, apply_ocr=False)
    dummy_words = ["Total", "Amount", "10.40", "Date", "25", "/", "09", "/", "2017"] * 50
    dummy_boxes = [[100 * (i % 10), 50, 100 * (i % 10) + 80, 70]
                   for i in range(len(dummy_words))]
    dummy_image = Image.new("RGB", (224, 224), (255, 255, 255))
    enc = processor(dummy_image, dummy_words, boxes=dummy_boxes,
                    truncation=True, padding="max_length", max_length=512,
                    return_tensors="pt")
    inputs_pt = {k: v for k, v in enc.items() if k != "word_labels"}

    # ── 4. Latency benchmark ──
    print("Benchmarking latency ...")

    # PyTorch GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_dir).eval()
    if device == "cuda":
        pt_model = pt_model.cuda()
        # Warm CUDA
        for _ in range(3):
            run_pytorch(pt_model, inputs_pt, device)
    pt_lat = benchmark_latency(
        lambda inp: run_pytorch(pt_model, inp, device),
        inputs_pt, warmup=3, runs=args.bench_runs,
    )
    print(f"  PyTorch {device.upper()}: p50={pt_lat['p50_ms']}ms  mean={pt_lat['mean_ms']}ms")

    # ONNX FP32 CPU
    sess_fp32 = build_ort_session(fp32_path)
    fp32_lat = benchmark_latency(
        lambda inp: run_ort(sess_fp32, inp),
        inputs_pt, warmup=3, runs=args.bench_runs,
    )
    print(f"  ONNX FP32 CPU: p50={fp32_lat['p50_ms']}ms  mean={fp32_lat['mean_ms']}ms")

    # ONNX INT8 CPU
    sess_int8 = build_ort_session(int8_path)
    int8_lat = benchmark_latency(
        lambda inp: run_ort(sess_int8, inp),
        inputs_pt, warmup=3, runs=args.bench_runs,
    )
    print(f"  ONNX INT8 CPU: p50={int8_lat['p50_ms']}ms  mean={int8_lat['mean_ms']}ms")

    speedup_vs_pt  = round(pt_lat["mean_ms"] / int8_lat["mean_ms"], 2)
    speedup_vs_fp32 = round(fp32_lat["mean_ms"] / int8_lat["mean_ms"], 2)
    print(f"\n  INT8 speedup vs PyTorch {device.upper()}: {speedup_vs_pt}x")
    print(f"  INT8 speedup vs ONNX FP32 CPU:          {speedup_vs_fp32}x")

    # ── 5. F1 sanity check on val split ──
    print(f"\nRunning F1 sanity check (val_samples={args.val_samples}) ...")
    train_recs = json.loads(Path(args.train_data).read_text())
    val_recs = train_recs[-args.val_samples:]
    img_dir = Path(args.train_img_dir)

    from training.train_layoutlmv3 import SROIEDataset, build_bio
    val_ds = SROIEDataset(val_recs, img_dir if img_dir.exists() else None, processor)

    pt_f1_scores, fp32_f1_scores, int8_f1_scores = [], [], []

    for idx in range(min(20, len(val_ds))):   # sample 20 for speed
        sample = val_ds[idx]
        inp = {k: v.unsqueeze(0) for k, v in sample.items() if k != "labels"}
        gold_labels = sample["labels"].unsqueeze(0).numpy()

        # PyTorch
        with torch.no_grad():
            pt_logits = pt_model(**{k: v.to(device) for k, v in inp.items()}).logits
        pt_f1_scores.append(token_f1(pt_logits.cpu().numpy(), gold_labels))

        # ONNX FP32
        fp32_logits = run_ort(sess_fp32, inp)
        fp32_f1_scores.append(token_f1(fp32_logits, gold_labels))

        # ONNX INT8
        int8_logits = run_ort(sess_int8, inp)
        int8_f1_scores.append(token_f1(int8_logits, gold_labels))

    pt_f1   = round(sum(pt_f1_scores) / len(pt_f1_scores), 4)
    fp32_f1 = round(sum(fp32_f1_scores) / len(fp32_f1_scores), 4)
    int8_f1 = round(sum(int8_f1_scores) / len(int8_f1_scores), 4)
    f1_drop = round(pt_f1 - int8_f1, 4)
    print(f"  F1  PyTorch={pt_f1}  ONNX-FP32={fp32_f1}  ONNX-INT8={int8_f1}  drop={f1_drop}")

    # ── 6. Model sizes ──
    pt_size_mb   = sum(p.numel() * p.element_size() for p in pt_model.parameters()) / 1024 / 1024
    fp32_size_mb = Path(fp32_path).stat().st_size / 1024 / 1024
    int8_size_mb = Path(int8_path).stat().st_size / 1024 / 1024

    # ── 7. Write report ──
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    Path(args.out_log).mkdir(parents=True, exist_ok=True)

    md = [f"# ONNX Export + INT8 Quantization Benchmark {stamp}", "",
          "## Setup",
          f"- Base model: LayoutLMv3-base fine-tuned on SROIE (BIO token classification)",
          f"- PyTorch device: {device.upper()}",
          f"- ONNX runtime: CPU only (ORT v{__import__('onnxruntime').__version__})",
          f"- INT8: dynamic quantization, per-channel weights, MatMul+Gemm ops",
          f"- Benchmark: {args.bench_runs} runs, seq_len=512, batch=1", "",
          "## Latency (seq_len=512, batch=1)", "",
          f"| variant | p50 (ms) | p95 (ms) | mean (ms) | model size |",
          f"|---|---|---|---|---|",
          f"| PyTorch ({device.upper()}) | {pt_lat['p50_ms']} | {pt_lat['p95_ms']} | {pt_lat['mean_ms']} | {pt_size_mb:.0f} MB |",
          f"| ONNX FP32 (CPU) | {fp32_lat['p50_ms']} | {fp32_lat['p95_ms']} | {fp32_lat['mean_ms']} | {fp32_size_mb:.0f} MB |",
          f"| **ONNX INT8 (CPU)** | **{int8_lat['p50_ms']}** | **{int8_lat['p95_ms']}** | **{int8_lat['mean_ms']}** | **{int8_size_mb:.0f} MB** |", "",
          f"INT8 speedup vs PyTorch {device.upper()}: **{speedup_vs_pt}x** | vs ONNX FP32: **{speedup_vs_fp32}x**",
          f"Model size reduction: {fp32_size_mb:.0f} MB → {int8_size_mb:.0f} MB (**{fp32_size_mb/int8_size_mb:.1f}x smaller**)", "",
          "## Accuracy (token-level F1, val split n=20)", "",
          f"| variant | token F1 | F1 drop vs PyTorch |",
          f"|---|---|---|",
          f"| PyTorch ({device.upper()}) | {pt_f1} | — |",
          f"| ONNX FP32 (CPU) | {fp32_f1} | {round(pt_f1 - fp32_f1, 4)} |",
          f"| ONNX INT8 (CPU) | {int8_f1} | **{f1_drop}** |", "",
          f"> F1 drop ≤ 0.02 is acceptable for production INT8 deployment.", "",
          "## Production deployment path",
          "```",
          "Fine-tuned PyTorch → export_onnx.py → model_int8.onnx",
          "                                     → Triton ONNX backend (existing stack)",
          "                                     → dynamic batching (existing batcher)",
          "```",
          "The INT8 model drops into the existing Triton ONNX serving infrastructure",
          "(same as RapidOCR PP-OCRv4) — no additional serving code needed.",
    ]

    out_md = Path(args.out_log) / f"onnx_benchmark_{stamp}.md"
    out_md.write_text("\n".join(md))

    raw = {
        "latency": {"pytorch_gpu": pt_lat, "onnx_fp32_cpu": fp32_lat, "onnx_int8_cpu": int8_lat},
        "speedup": {"int8_vs_pytorch": speedup_vs_pt, "int8_vs_fp32": speedup_vs_fp32},
        "size_mb": {"pytorch": round(pt_size_mb, 1), "onnx_fp32": round(fp32_size_mb, 1),
                    "onnx_int8": round(int8_size_mb, 1)},
        "f1": {"pytorch": pt_f1, "onnx_fp32": fp32_f1, "onnx_int8": int8_f1, "drop": f1_drop},
    }
    (Path(args.out_log) / "onnx_benchmark_raw.json").write_text(json.dumps(raw, indent=2))

    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
