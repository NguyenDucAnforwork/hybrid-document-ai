"""Load test: measure throughput and latency under batch sizes 1 / 5 / 10.

Usage:
  # Sequential batch test (existing)
  python scripts/load_test.py --url http://localhost:8000 --img-dir $WS/data/sroie/test/images

  # Concurrent single-document stress test (Phase 1 SLA: p95 ≤ 3s at concurrency=5)
  python scripts/load_test.py --url http://localhost:8000 --img-dir $WS/data/sroie/test/images --concurrent

Output: docs/logs/load_test_TIMESTAMP.md
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import json
import random
import statistics
import time
from pathlib import Path

import httpx

BATCH_SIZES = [1, 5, 10]
ROUNDS = 3          # repeat each batch size N times


def run_batch(client: httpx.Client, url: str, imgs: list[Path], n: int) -> dict:
    sample = random.sample(imgs, min(n, len(imgs)))
    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in sample]
    t0 = time.perf_counter()
    r = client.post(f"{url}/batch_jobs", files=files, timeout=120)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    return {"latency_s": latency, "n": n, "response": r.json()}


async def _concurrent_stress(url: str, imgs: list[Path],
                             concurrency: int = 5, total: int = 20) -> dict:
    """Simulate `concurrency` simultaneous single-document requests.

    Measures per-request latency to verify Phase 1 SLA: p95 ≤ 3.0s under load.
    """
    sample = [random.choice(imgs) for _ in range(total)]

    async def one_request(client: httpx.AsyncClient, img: Path) -> float:
        data = img.read_bytes()
        t0 = time.perf_counter()
        r = await client.post(
            f"{url}/documents/extract",
            files={"file": (img.name, data, "image/jpeg")},
            timeout=30.0,
        )
        r.raise_for_status()
        return time.perf_counter() - t0

    sem = asyncio.Semaphore(concurrency)

    async def bounded(img: Path):
        async with sem:
            return await one_request(client, img)

    latencies, errors = [], []
    async with httpx.AsyncClient() as client:
        tasks = [bounded(img) for img in sample]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        else:
            latencies.append(r)

    latencies.sort()
    p50 = statistics.median(latencies) * 1000 if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] * 1000 if latencies else 0
    sla_pass = p95 <= 3000

    print(f"\n── Concurrent stress (c={concurrency}, n={total}) ──")
    print(f"  completed={len(latencies)}  errors={len(errors)}")
    print(f"  p50={p50:.0f}ms  p95={p95:.0f}ms")
    print(f"  Phase 1 SLA p95≤3000ms: {'✓ PASS' if sla_pass else '✗ FAIL'}")
    if errors:
        print(f"  sample error: {errors[0]}")

    return {
        "mode": "concurrent",
        "concurrency": concurrency,
        "total_requests": total,
        "completed": len(latencies),
        "errors": len(errors),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "sla_pass": sla_pass,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="docs/logs")
    ap.add_argument("--concurrent", action="store_true",
                    help="Run async concurrent stress test (Phase 1 SLA check)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Number of simultaneous requests for --concurrent mode")
    ap.add_argument("--total", type=int, default=20,
                    help="Total requests to fire for --concurrent mode")
    args = ap.parse_args()

    imgs = sorted(Path(args.img_dir).glob("*.jpg"))
    if not imgs:
        print("No images found"); return

    # Quick health check
    with httpx.Client() as hc:
        r = hc.get(f"{args.url}/health", timeout=10)
        print("Health:", r.json())

    if args.concurrent:
        concurrent_result = asyncio.run(
            _concurrent_stress(args.url, imgs, args.concurrency, args.total)
        )
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out) / f"concurrent_test_{stamp}.json"
        out_path.write_text(json.dumps(concurrent_result, indent=2))
        print(f"\nwrote {out_path}")
        return

    results = []
    with httpx.Client() as client:
        for batch_n in BATCH_SIZES:
            latencies = []
            for _ in range(ROUNDS):
                res = run_batch(client, args.url, imgs, batch_n)
                latencies.append(res["latency_s"])
                print(f"  batch={batch_n} latency={res['latency_s']:.2f}s")
            latencies.sort()
            throughput = batch_n / (sum(latencies) / len(latencies))
            results.append({
                "batch_size": batch_n,
                "rounds": ROUNDS,
                "latency_p50_s": round(latencies[len(latencies)//2], 3),
                "latency_p95_s": round(latencies[min(int(len(latencies)*0.95), len(latencies)-1)], 3),
                "throughput_docs_per_min": round(throughput * 60, 1),
            })

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    md = [f"# Load Test {stamp}", "",
          "| batch_size | latency_p50 | latency_p95 | throughput (docs/min) |",
          "|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['batch_size']} | {r['latency_p50_s']}s "
                  f"| {r['latency_p95_s']}s | {r['throughput_docs_per_min']} |")
    md += ["", f"Rounds per batch size: {ROUNDS}",
           f"Image dir: {args.img_dir}", f"Server: {args.url}"]

    out_path = Path(args.out) / f"load_test_{stamp}.md"
    out_path.write_text("\n".join(md))
    (Path(args.out) / "load_test_raw.json").write_text(json.dumps(results, indent=2))
    print("\n".join(md))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
