"""Kubeflow Pipelines (KFP) definition — PRODUCTION target for the training DAG.

Artifact (not executed on the dev box; no k8s). Mirrors mlops/pipeline.py: each
step is a containerized component with typed inputs/outputs, so KFP gives
versioning, caching, scheduling, retries/resume and lineage visualization
(see "MLOps - Training Pipeline"). Compile with: kfp dsl compile.
"""
from __future__ import annotations
try:
    from kfp import dsl
except Exception:  # kfp not installed in the in-budget build
    dsl = None

IMAGE = "ghcr.io/nguyenducanforwork/hybrid-docai:latest"

if dsl:
    @dsl.container_component
    def ingest_op(out_dir: dsl.OutputPath()):
        return dsl.ContainerSpec(image=IMAGE,
                                 command=["python", "scripts/prepare_sroie.py"],
                                 args=["--out", out_dir])

    @dsl.container_component
    def validate_op(data_dir: dsl.InputPath()):
        return dsl.ContainerSpec(image=IMAGE,
                                 command=["python", "-m", "mlops.data_validation"],
                                 args=[data_dir])

    @dsl.container_component
    def train_op(data_dir: dsl.InputPath(), model_dir: dsl.OutputPath()):
        return dsl.ContainerSpec(image=IMAGE,
                                 command=["python", "training/train_kie.py"],
                                 args=["--data", data_dir, "--version", "kfp"])

    @dsl.container_component
    def eval_op(data_dir: dsl.InputPath()):
        return dsl.ContainerSpec(image=IMAGE,
                                 command=["python", "scripts/run_benchmark.py"],
                                 args=["--data", data_dir, "--f1-threshold", "0.2"])

    @dsl.pipeline(name="hybrid-docai-kie", description="KIE training DAG")
    def kie_pipeline():
        ing = ingest_op()
        val = validate_op(data_dir=ing.outputs["out_dir"])
        tr = train_op(data_dir=ing.outputs["out_dir"]).after(val)
        eval_op(data_dir=ing.outputs["out_dir"]).after(tr)
        # KFP caching + retries: set on tasks, e.g.
        # tr.set_caching_options(True); tr.set_retry(num_retries=2)
