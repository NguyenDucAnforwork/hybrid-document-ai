"""WP-3 — fine-tune the CRNN+CTC Vietnamese recognizer under hard budget.

Enforces the plan's envelope: ≤1h wall clock, ≤5GB peak VRAM (tracked via
torch.cuda.max_memory_allocated for THIS process). Early-stops on val CER, keeps
best checkpoint only, and exports ONNX. A reproducible budget-compliant run (even
a negative result) satisfies "minimum done".

Usage:
  python training/train_ocr_rec.py --dry-run          # ~20 steps, report VRAM/throughput
  python training/train_ocr_rec.py --epochs 3 --batch 128
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.ocr.crnn import CRNN, CharsetCodec        # noqa
from eval.metrics import cer as cer_metric              # noqa

IMG_H, IMG_W = 32, 256
WALL_LIMIT_S = 3300        # 55 min hard stop (leaves time for export/eval in the 1h budget)
VRAM_LIMIT_MB = 5000


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def preprocess(img_gray: np.ndarray) -> np.ndarray:
    """Resize keep-ratio to height 32, then pad/crop to width 256. Returns (1,32,256) float."""
    h, w = img_gray.shape[:2]
    new_w = max(1, min(IMG_W, int(round(w * IMG_H / max(h, 1)))))
    r = cv2.resize(img_gray, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_H, IMG_W), np.uint8)
    canvas[:, :new_w] = r
    x = canvas.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x[None, :, :]


class OCRDataset(Dataset):
    def __init__(self, txt: Path, codec: CharsetCodec):
        self.items = []
        for ln in txt.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            p, t = ln.split("\t", 1)
            self.items.append((p, t))
        self.codec = codec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t = self.items[i]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), np.uint8)
        x = preprocess(img)
        y = self.codec.encode(t)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), t


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = [b[1] for b in batch]
    texts = [b[2] for b in batch]
    target = torch.cat(ys) if ys else torch.tensor([], dtype=torch.long)
    target_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return xs, target, target_lens, texts


@torch.no_grad()
def evaluate(model, loader, codec, device, max_batches=None):
    model.eval()
    cers, exact, n = [], 0, 0
    for bi, (xs, _, _, texts) in enumerate(loader):
        logits = model(xs.to(device))               # (T,B,C)
        preds = logits.argmax(2).permute(1, 0).cpu().tolist()
        for ids, gold in zip(preds, texts):
            pred = codec.decode_greedy(ids)
            cers.append(cer_metric(pred, gold))
            exact += int(pred == gold)
            n += 1
        if max_batches and bi + 1 >= max_batches:
            break
    return (sum(cers) / max(n, 1)), (exact / max(n, 1)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    ap.add_argument("--out", default=str(_ws() / "models/ocr/vi_mcocr_crnn_ft"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    codec = CharsetCodec.from_dict_file(data / "vi_dict.txt")
    tr = OCRDataset(data / "train.txt", codec)
    va = OCRDataset(data / "val.txt", codec)
    print(f"train={len(tr)} val={len(va)} num_classes={codec.num_classes} device={device}")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=8,
                    collate_fn=collate, drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=8, collate_fn=collate)

    model = CRNN(codec.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    def peak_mb():
        return torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    # ---- dry run: throughput + VRAM ceiling check ----
    if args.dry_run:
        model.train()
        for bi, (xs, target, tlens, _) in enumerate(tl):
            xs = xs.to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xs)
                T, B, _ = logits.shape
                ilens = torch.full((B,), T, dtype=torch.long)
                loss = ctc(logits.log_softmax(2), target.to(device), ilens, tlens)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            if bi >= 20:
                break
        dt = time.perf_counter() - t0
        rep = {"dry_run": True, "peak_vram_mb": round(peak_mb(), 1),
               "steps": 21, "sec": round(dt, 1),
               "img_per_s": round(21 * args.batch / dt, 1),
               "vram_ok": peak_mb() <= VRAM_LIMIT_MB,
               "batch": args.batch}
        print(json.dumps(rep, indent=2))
        if peak_mb() > VRAM_LIMIT_MB:
            print(f"WARNING: peak VRAM {peak_mb():.0f}MB > {VRAM_LIMIT_MB}MB -> lower --batch to 64/32")
            sys.exit(2)
        return

    # ---- full training with budget guards ----
    best_cer, best_state, stop_reason = 1e9, None, "completed"
    step = 0
    for ep in range(args.epochs):
        model.train()
        for xs, target, tlens, _ in tl:
            xs = xs.to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xs)
                T, B, _ = logits.shape
                ilens = torch.full((B,), T, dtype=torch.long)
                loss = ctc(logits.log_softmax(2), target.to(device), ilens, tlens)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            step += 1
            if peak_mb() > VRAM_LIMIT_MB:
                stop_reason = f"vram_exceeded:{peak_mb():.0f}MB"; break
            if time.perf_counter() - t0 > WALL_LIMIT_S:
                stop_reason = "wall_clock_exceeded"; break
        vcer, vexact, n = evaluate(model, vl, codec, device)
        print(f"epoch {ep+1}/{args.epochs} loss={loss.item():.3f} val_CER={vcer:.4f} "
              f"exact={vexact:.3f} peak_vram={peak_mb():.0f}MB t={time.perf_counter()-t0:.0f}s")
        if vcer < best_cer:
            best_cer = vcer
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stop_reason != "completed":
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "num_classes": codec.num_classes,
                "img_h": IMG_H, "img_w": IMG_W}, out / "best.pt")

    # export ONNX (opset 12, fixed shape for portable CPU/RTX1650 inference)
    model.eval().to("cpu")
    dummy = torch.zeros(1, 1, IMG_H, IMG_W)
    onnx_path = out / "model.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["image"],
                      output_names=["logits"], opset_version=12,
                      dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}})
    (out / "vi_dict.txt").write_text("\n".join(codec.chars), encoding="utf-8")

    log = {
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_vram_mb": round(peak_mb(), 1),
        "epochs_done": ep + 1, "best_val_cer": round(best_cer, 4),
        "stop_reason": stop_reason, "batch": args.batch, "lr": args.lr,
        "num_classes": codec.num_classes, "train_n": len(tr), "val_n": len(va),
        "wall_limit_s": WALL_LIMIT_S, "vram_limit_mb": VRAM_LIMIT_MB,
        "onnx_bytes": onnx_path.stat().st_size,
    }
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
