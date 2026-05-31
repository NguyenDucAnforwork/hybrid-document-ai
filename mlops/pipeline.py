"""Local training-pipeline DAG runner (MLOps - Training Pipeline).

Implements the lecture's principles on a single box so it is runnable in-budget:
- discrete steps (ingest -> validate -> train -> evaluate -> register)
- per-step CACHING + RETRY/RESUME (skip done steps; rerun failed)
- TRACEABILITY: every run writes a manifest with inputs/outputs/metrics/version
  (lineage), so you can answer "which data+code produced this model?".
The production target is Kubeflow Pipelines (see mlops/kfp_pipeline.py) — same
DAG, each step a container. This runner mirrors that contract.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

from docai.config import WORKSPACE, ARTIFACTS_DIR
from mlops.data_validation import validate

RUNS_DIR = ARTIFACTS_DIR / "pipeline_runs"
VENV_PY = sys.executable


class Pipeline:
    def __init__(self, run_id: str, resume: bool = True):
        self.run_dir = RUNS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.manifest = {"run_id": run_id, "steps": {}}

    def step(self, name: str, fn, retries: int = 1):
        marker = self.run_dir / f"{name}.json"
        if self.resume and marker.exists():
            out = json.loads(marker.read_text())
            self.manifest["steps"][name] = {**out, "cached": True}
            print(f"[skip] {name} (cached)")
            return out["result"]
        last = None
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                result = fn()
                rec = {"result": result, "dur_s": round(time.time() - t0, 2),
                       "attempt": attempt}
                marker.write_text(json.dumps(rec, indent=2, default=str))
                self.manifest["steps"][name] = rec
                print(f"[ok]   {name} ({rec['dur_s']}s)")
                return result
            except Exception as e:  # retry/resume on cheap unreliable compute
                last = str(e)
                print(f"[retry {attempt}] {name}: {e}")
                time.sleep(1)
        self.manifest["steps"][name] = {"error": last}
        raise RuntimeError(f"step {name} failed: {last}")

    def finish(self):
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str))
        print(f"\nmanifest -> {self.run_dir / 'manifest.json'}")


def run(run_id="kie_pipeline", version="v_pipe"):
    ws = Path(WORKSPACE)
    sroie_train = ws / "data/sroie/train/labels.json"
    synth = ws / "data/receipts/labels.json"
    p = Pipeline(run_id)

    def ingest():
        if not sroie_train.exists():
            subprocess.run([VENV_PY, "scripts/prepare_sroie.py"], check=True)
        if not synth.exists():
            from docai.synth import generate
            generate(str(ws / "data/receipts"), 120, 42)
        return {"sroie": sroie_train.exists(), "synth": synth.exists()}

    def validate_data():
        rep = validate(sroie_train)
        if not rep["passed"]:
            raise ValueError(f"data validation failed: {rep['checks']}")
        return {"coverage": rep["coverage"], "n": rep["n"]}

    def train():
        r = subprocess.run(
            [VENV_PY, "training/train_kie.py", "--data",
             str(ws / "data/sroie/train"), str(ws / "data/receipts"),
             "--version", version], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-500:])
        return {"version": version}

    def evaluate():
        r = subprocess.run(
            [VENV_PY, "scripts/run_benchmark.py", "--data",
             str(ws / "data/sroie/test"), "--f1-threshold", "0.2"],
            capture_output=True, text=True)
        return {"gate_pass": r.returncode == 0, "tail": r.stdout[-200:]}

    def register_prod():
        from docai import registry
        registry.transition("kie", version, "production")  # promote
        return {"promoted": version, "stage": "production"}

    p.step("ingest", ingest)
    p.step("validate_data", validate_data)
    p.step("train", train, retries=2)
    res = p.step("evaluate", evaluate)
    if res["gate_pass"]:
        p.step("register_production", register_prod)
    else:
        print("eval gate FAILED -> not promoting to production")
    p.finish()


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--run-id", default="kie_pipeline")
    a.add_argument("--version", default="v_pipe")
    a.add_argument("--no-resume", action="store_true")
    args = a.parse_args()
    run(args.run_id, args.version)
