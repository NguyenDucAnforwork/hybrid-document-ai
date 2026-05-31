"""Model registry (MLOps) — versioning, lifecycle STAGES, and lineage.

Maps to "MLOps - More and beyond" (Model Registry): a model moves through
developing -> staging -> production -> archived, and we record WHICH pipeline
run / data / metrics produced each version (lineage) for audit & reproducibility.
"""
from __future__ import annotations
import json
import yaml
from pathlib import Path
from .config import MODELS_DIR

REGISTRY_PATH = MODELS_DIR / "registry.yaml"
STAGES = ("developing", "staging", "production", "archived")


def _sanitize(obj):
    return json.loads(json.dumps(obj, default=float))


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    return {}


def _save(reg):
    REGISTRY_PATH.write_text(yaml.safe_dump(reg, sort_keys=False))


def register(model: str, version: str, path: str, metrics: dict, created: str,
             stage: str = "staging", lineage: dict | None = None,
             make_active: bool = True) -> None:
    """Register a model version with metrics + lineage (run id, data, params)."""
    reg = _load()
    entries = reg.setdefault(model, {"active": None, "production": None, "versions": {}})
    entries["versions"][version] = {
        "path": str(path), "stage": stage, "metrics": _sanitize(metrics),
        "created": created, "lineage": _sanitize(lineage or {}),
    }
    if make_active:
        entries["active"] = version
    if stage == "production":
        entries["production"] = version
    _save(reg)


def transition(model: str, version: str, stage: str) -> None:
    """Promote/rollback/archive a version (dev->staging->prod->archived)."""
    assert stage in STAGES, f"stage must be one of {STAGES}"
    reg = _load()
    entries = reg[model]
    entries["versions"][version]["stage"] = stage
    if stage == "production":
        # demote previous production to archived (single live prod model)
        for v, meta in entries["versions"].items():
            if v != version and meta.get("stage") == "production":
                meta["stage"] = "archived"
        entries["production"] = version
        entries["active"] = version
    _save(reg)


def active_version(model: str) -> str | None:
    e = _load().get(model, {})
    return e.get("production") or e.get("active")


def active_path(model: str) -> Path | None:
    reg = _load().get(model, {})
    v = reg.get("production") or reg.get("active")
    return Path(reg["versions"][v]["path"]) if v else None
