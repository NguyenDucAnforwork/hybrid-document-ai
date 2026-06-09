"""Compact CRNN+CTC text recognizer (WP-3).

Canonical CRNN (Shi et al. 2015): CNN feature extractor that collapses height to 1,
then BiLSTM over the width axis, then a linear CTC head. Tiny (~8M params) so it
trains under the WP-3 ≤5GB VRAM / ≤1h budget on an H100 and exports cleanly to ONNX.

Input: grayscale (B,1,32,W). Charset idx 0 is the CTC blank.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CharsetCodec:
    """Maps characters <-> indices. Index 0 reserved for CTC blank."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = blank
        self.itos = {i + 1: c for i, c in enumerate(chars)}

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1  # + blank

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode_greedy(self, ids: list[int]) -> str:
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0:
                out.append(self.itos.get(i, ""))
            prev = i
        return "".join(out)

    @classmethod
    def from_dict_file(cls, path) -> "CharsetCodec":
        from pathlib import Path
        chars = [ln for ln in Path(path).read_text(encoding="utf-8").split("\n") if ln != ""]
        return cls(chars)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 1, lstm_hidden: int = 256):
        super().__init__()

        def conv(i, o, k=3, s=1, p=1, bn=False):
            layers = [nn.Conv2d(i, o, k, s, p)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.cnn = nn.Sequential(
            *conv(in_ch, 64), nn.MaxPool2d(2, 2),                       # 32x256 -> 16x128
            *conv(64, 128), nn.MaxPool2d(2, 2),                         # -> 8x64
            *conv(128, 256), *conv(256, 256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                       # -> 4x65
            *conv(256, 512, bn=True), *conv(512, 512, bn=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                       # -> 2x66
            *conv(512, 512, k=2, s=1, p=0, bn=True),                    # -> 1x65
        )
        self.rnn = nn.LSTM(512, lstm_hidden, num_layers=2,
                           bidirectional=True, batch_first=False)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):                       # x: (B,1,32,W)
        f = self.cnn(x)                         # (B,512,1,W')
        b, c, h, w = f.shape
        assert h == 1, f"CNN height must collapse to 1, got {h}"
        f = f.squeeze(2).permute(2, 0, 1)       # (W', B, 512)
        f, _ = self.rnn(f)
        logits = self.fc(f)                     # (W', B, num_classes)
        return logits
