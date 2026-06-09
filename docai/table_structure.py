"""Zero-shot table structure utilities for bank-statement experiments."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

_DET_BUNDLE: dict | None = None
_STR_BUNDLE: dict | None = None


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    requested = os.environ.get("DOCAI_TABLE_DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_bundle(model_name: str):
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    device = _device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name).to(device).eval()
    return {"processor": processor, "model": model, "device": device}


def _get_detector():
    global _DET_BUNDLE
    if _DET_BUNDLE is None:
        _DET_BUNDLE = _load_bundle(
            os.environ.get("DOCAI_TABLE_DET_MODEL", "microsoft/table-transformer-detection")
        )
    return _DET_BUNDLE


def _get_structure_model():
    global _STR_BUNDLE
    if _STR_BUNDLE is None:
        _STR_BUNDLE = _load_bundle(
            os.environ.get(
                "DOCAI_TABLE_STR_MODEL",
                "microsoft/table-transformer-structure-recognition",
            )
        )
    return _STR_BUNDLE


def _postprocess(bundle: dict, image_pil, outputs, threshold: float) -> list[dict]:
    torch = _torch()
    target = torch.tensor([image_pil.size[::-1]], device=bundle["device"])
    result = bundle["processor"].post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target,
    )[0]
    out = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        out.append(
            {
                "label": bundle["model"].config.id2label[int(label)],
                "score": round(float(score), 4),
                "bbox": [float(x) for x in box.tolist()],
            }
        )
    return out


def _infer(bundle: dict, image_pil, threshold: float) -> list[dict]:
    inputs = bundle["processor"](images=image_pil, return_tensors="pt")
    inputs = {k: v.to(bundle["device"]) for k, v in inputs.items()}
    torch = _torch()
    with torch.no_grad():
        outputs = bundle["model"](**inputs)
    return _postprocess(bundle, image_pil, outputs, threshold)


def _round_box(box: list[float]) -> list[int]:
    return [int(round(x)) for x in box]


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _contains(box: list[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _choose_span(center: float, boxes: list[dict], axis: str) -> int | None:
    if not boxes:
        return None
    if axis == "x":
        lo_i, hi_i = 0, 2
    else:
        lo_i, hi_i = 1, 3
    for idx, box in enumerate(boxes):
        lo, hi = box["bbox"][lo_i], box["bbox"][hi_i]
        if lo <= center <= hi:
            return idx
    distances = []
    for idx, box in enumerate(boxes):
        mid = (box["bbox"][lo_i] + box["bbox"][hi_i]) / 2.0
        distances.append((abs(center - mid), idx))
    distances.sort()
    return distances[0][1]


def _sort_boxes(items: list[dict], axis: str) -> list[dict]:
    key_idx = 0 if axis == "x" else 1
    return sorted(items, key=lambda it: it["bbox"][key_idx])


def detect_structure(image_bgr: np.ndarray, tokens: list[dict]) -> dict:
    from PIL import Image

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    det_threshold = float(os.environ.get("DOCAI_TABLE_DET_THRESH", "0.7"))
    str_threshold = float(os.environ.get("DOCAI_TABLE_STR_THRESH", "0.7"))

    detector = _get_detector()
    detections = _infer(detector, image, det_threshold)
    tables = [d for d in detections if d["label"] in {"table", "table rotated"}]
    if not tables:
        return {
            "ok": False,
            "reason": "no_table_detected",
            "detections": detections,
            "rows": [],
            "columns": [],
            "header_boxes": [],
            "table_bbox": None,
            "table_tokens": [],
            "assigned_tokens": [],
            "header_row_index": None,
        }

    table = max(tables, key=lambda d: d["score"])
    tx0, ty0, tx1, ty1 = _round_box(table["bbox"])
    crop = image.crop((tx0, ty0, tx1, ty1))

    structure = _get_structure_model()
    structures = _infer(structure, crop, str_threshold)
    rows = _sort_boxes(
        [
            {"label": s["label"], "score": s["score"],
             "bbox": [s["bbox"][0] + tx0, s["bbox"][1] + ty0, s["bbox"][2] + tx0, s["bbox"][3] + ty0]}
            for s in structures if s["label"] == "table row"
        ],
        axis="y",
    )
    cols = _sort_boxes(
        [
            {"label": s["label"], "score": s["score"],
             "bbox": [s["bbox"][0] + tx0, s["bbox"][1] + ty0, s["bbox"][2] + tx0, s["bbox"][3] + ty0]}
            for s in structures if s["label"] == "table column"
        ],
        axis="x",
    )
    header_boxes = [
        {"label": s["label"], "score": s["score"],
         "bbox": [s["bbox"][0] + tx0, s["bbox"][1] + ty0, s["bbox"][2] + tx0, s["bbox"][3] + ty0]}
        for s in structures if s["label"] == "table column header"
    ]

    table_tokens = []
    assigned = []
    grid: dict[int, dict[int, list[dict]]] = {}
    for tok in tokens:
        bbox = tok.get("bbox", [0, 0, 0, 0])
        cx, cy = _box_center(bbox)
        if not _contains(table["bbox"], cx, cy):
            continue
        table_tokens.append(tok)
        ridx = _choose_span(cy, rows, "y")
        cidx = _choose_span(cx, cols, "x")
        if ridx is None or cidx is None:
            continue
        grid.setdefault(ridx, {}).setdefault(cidx, []).append(tok)
        assigned.append(
            {
                "text": tok.get("text", ""),
                "bbox": bbox,
                "row_index": ridx,
                "col_index": cidx,
            }
        )

    header_row_index = 0 if rows else None
    if rows and header_boxes:
        def overlap_y(row_box, hdr_box):
            return max(0.0, min(row_box[3], hdr_box[3]) - max(row_box[1], hdr_box[1]))

        scores = []
        for idx, row in enumerate(rows):
            y_overlap = sum(overlap_y(row["bbox"], hb["bbox"]) for hb in header_boxes)
            scores.append((y_overlap, idx))
        scores.sort(reverse=True)
        header_row_index = scores[0][1]

    grid_text = []
    for ridx in range(len(rows)):
        row_entry = []
        for cidx in range(len(cols)):
            toks = sorted(
                grid.get(ridx, {}).get(cidx, []),
                key=lambda t: (t["bbox"][0], t["bbox"][1]),
            )
            row_entry.append(
                {
                    "row_index": ridx,
                    "col_index": cidx,
                    "text": " ".join(t.get("text", "").strip() for t in toks).strip(),
                    "token_count": len(toks),
                }
            )
        grid_text.append(row_entry)

    assignment_rate = round(len(assigned) / max(len(table_tokens), 1), 3)
    return {
        "ok": bool(rows and cols),
        "reason": None if rows and cols else "missing_rows_or_columns",
        "detections": detections,
        "structure_detections": structures,
        "table_bbox": table["bbox"],
        "rows": rows,
        "columns": cols,
        "header_boxes": header_boxes,
        "header_row_index": header_row_index,
        "table_tokens": table_tokens,
        "assigned_tokens": assigned,
        "grid_text": grid_text,
        "assignment_rate": assignment_rate,
    }


def save_debug_overlay(image_bgr: np.ndarray, structure: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image_bgr.copy()
    if structure.get("table_bbox"):
        x0, y0, x1, y1 = _round_box(structure["table_bbox"])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 180, 0), 2)
    for idx, row in enumerate(structure.get("rows", [])):
        x0, y0, x1, y1 = _round_box(row["bbox"])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (50, 220, 50), 1)
        cv2.putText(canvas, f"r{idx}", (x0 + 3, max(12, y0 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 50), 1, cv2.LINE_AA)
    for idx, col in enumerate(structure.get("columns", [])):
        x0, y0, x1, y1 = _round_box(col["bbox"])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 120, 255), 1)
        cv2.putText(canvas, f"c{idx}", (x0 + 3, max(12, y0 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 120, 255), 1, cv2.LINE_AA)
    if structure.get("header_row_index") is not None and structure.get("rows"):
        hdr = structure["rows"][structure["header_row_index"]]["bbox"]
        x0, y0, x1, y1 = _round_box(hdr)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
    for a in structure.get("assigned_tokens", []):
        bbox = a["bbox"]
        cx, cy = _box_center(bbox)
        cv2.circle(canvas, (int(cx), int(cy)), 2, (255, 255, 255), -1)
        cv2.putText(canvas, f"{a['row_index']},{a['col_index']}", (int(cx) + 2, int(cy) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
