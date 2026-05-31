"""Minimal model registry (MLOps versioning). Tracks model->version->metrics."""
from __future__ import annotations
import json
import yaml
from pathlib import Path
from .config import MODELS_DIR

REGISTRY_PATH = MODELS_DIR / "registry.yaml"


def _sanitize(obj):
    """Convert numpy/exotic types to plain Python so yaml can serialize."""
    return json.loads(json.dumps(obj, default=float))


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    return {}


def register(model: str, version: str, path: str, metrics: dict, created: str,
             make_active: bool = True) -> None:
    reg = _load()
    entries = reg.setdefault(model, {"active": None, "versions": {}})
    entries["versions"][version] = {"path": str(path), "metrics": _sanitize(metrics),
                                    "created": created}
    if make_active:
        entries["active"] = version
    REGISTRY_PATH.write_text(yaml.safe_dump(reg, sort_keys=False))


def active_version(model: str) -> str | None:
    return _load().get(model, {}).get("active")


def active_path(model: str) -> Path | None:
    reg = _load().get(model, {})
    v = reg.get("active")
    if not v:
        return None
    return Path(reg["versions"][v]["path"])
