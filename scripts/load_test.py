"""Load test: measure throughput and latency under batch sizes 1 / 5 / 10.

Usage: python scripts/load_test.py --url http://localhost:8000 --img-dir $WS/data/sroie/test/images
Output: docs/logs/load_test_TIMESTAMP.md
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import random
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    imgs = sorted(Path(args.img_dir).glob("*.jpg"))
    if not imgs:
        print("No images found"); return

    results = []
    with httpx.Client() as client:
        # health check
        r = client.get(f"{args.url}/health", timeout=10)
        print("Health:", r.json())

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
