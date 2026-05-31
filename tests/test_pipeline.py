"""Smoke tests: quality, KIE, batcher, end-to-end pipeline."""
import asyncio
import io
import sys
from pathlib import Path

import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docai.quality import check_quality
from docai.kie import norm_money, norm_date, norm_field
from docai.batcher import DynamicBatcher


def test_norm():
    assert norm_money("235,000") == 235000
    assert norm_date("31/05/2026") == "2026-05-31"
    assert norm_field("payment_method", "QR Pay") == "QR"


def test_quality_blur_flag():
    blank = np.full((500, 500, 3), 255, np.uint8)
    q = check_quality(blank)
    assert q.is_blurry  # flat image -> low laplacian variance


def test_quality_lowres():
    small = np.random.randint(0, 255, (100, 100, 3), np.uint8)
    q = check_quality(small)
    assert q.low_resolution and not q.quality_pass


def test_batcher_groups():
    async def run():
        b = DynamicBatcher(batch_fn=lambda xs: [x * 2 for x in xs],
                           max_batch=8, max_delay_ms=50)
        b.start()
        res = await asyncio.gather(*(b.submit(i) for i in range(5)))
        assert res == [0, 2, 4, 6, 8]
        assert b.stats["max_seen"] >= 2  # proves real batching
    asyncio.run(run())
