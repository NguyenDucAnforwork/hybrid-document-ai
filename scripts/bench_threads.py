"""OCR throughput sweep over (workers x intra_threads x concurrency).

Sweeping concurrency ALONE is misleading: with W worker processes each running
ONNX Runtime at T intra-op threads, effective CPU load is W*T. Oversubscription
(e.g. 4 workers x 4 threads on 8 cores) can make latency WORSE. So we grid the
two thread variables AND the request concurrency, and report throughput +
serialize/worker/pool-wait breakdown to locate the bottleneck.

  python scripts/bench_threads.py --img-dir $WS/data/sroie/test/images \
      --workers 1,2,4 --intra 1,2,4 --concurrency 1,2,5,10
  python scripts/bench_threads.py --synthetic 16          # no data needed
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ints(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def _load_images(args):
    import cv2
    if args.synthetic:
        import tempfile
        from docai.synth import generate
        d = tempfile.mkdtemp(prefix="bench_synth_")
        generate(d, args.synthetic, 7)
        paths = sorted((Path(d) / "images").glob("*"))
    else:
        paths = sorted(Path(args.img_dir).glob("*.jpg"))[: args.limit]
    return [cv2.imread(str(p)) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--intra", default="1,2,4")
    ap.add_argument("--concurrency", default="1,2,5,10")
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    from docai.serving.ocr_pool import ProcessPoolOCR

    images = _load_images(args)
    if not images:
        print("no images"); sys.exit(1)

    rows = []
    for w in _ints(args.workers):
        for t in _ints(args.intra):
            pool = ProcessPoolOCR(workers=w, intra_threads=t)
            pool.warmup()  # exclude cold start from the measurement
            for c in _ints(args.concurrency):
                batch = (images * ((c // len(images)) + 1))[:c]
                t0 = time.perf_counter()
                _, meta = pool.run_many_timed(batch)
                wall = time.perf_counter() - t0
                rows.append({
                    "workers": w, "intra_threads": t, "concurrency": c,
                    "effective_load": w * t,
                    "wall_s": round(wall, 3),
                    "throughput_docs_per_min": round(c / wall * 60, 1),
                    "serialize_ms": meta["serialize_ms"],
                    "worker_ms_max": round(max(meta["worker_ms"]), 1) if meta["worker_ms"] else 0,
                    "pool_wait_ms": meta["pool_wait_ms"],
                })
                print(f"  W={w} T={t} (load={w*t}) c={c}: "
                      f"{rows[-1]['throughput_docs_per_min']} docs/min "
                      f"serialize={meta['serialize_ms']}ms wait={meta['pool_wait_ms']}ms")
            pool.shutdown()

    best = max(rows, key=lambda r: r["throughput_docs_per_min"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    (out / "bench_threads_raw.json").write_text(
        json.dumps({"rows": rows, "best": best}, indent=2))
    md = [f"# OCR thread/worker sweep {stamp}", "",
          f"- best: workers={best['workers']} intra={best['intra_threads']} "
          f"concurrency={best['concurrency']} -> {best['throughput_docs_per_min']} docs/min", "",
          "| workers | intra | load=W*T | concurrency | docs/min | serialize ms | worker ms (max) | pool wait ms |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['workers']} | {r['intra_threads']} | {r['effective_load']} "
                  f"| {r['concurrency']} | {r['throughput_docs_per_min']} "
                  f"| {r['serialize_ms']} | {r['worker_ms_max']} | {r['pool_wait_ms']} |")
    (out / f"bench_threads_{stamp}.md").write_text("\n".join(md))
    print(f"\nbest: {best}\nwrote {out / f'bench_threads_{stamp}.md'}")


if __name__ == "__main__":
    main()
