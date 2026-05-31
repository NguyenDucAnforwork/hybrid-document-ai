"""Tests for MLOps components: metrics, chaos resilience, data validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import cer, anls, f1, ece


def test_cer_anls():
    assert cer("abc", "abc") == 0.0
    assert cer("abx", "abc") == round(1 / 3, 10) or abs(cer("abx", "abc") - 1 / 3) < 1e-6
    assert anls("abc", "abc") == 1.0
    assert anls("xyz", "abc") == 0.0          # below 0.5 -> 0


def test_f1_ece():
    assert f1(5, 0, 0) == 1.0
    assert f1(0, 5, 5) == 0.0
    # perfectly calibrated -> low ECE
    assert ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) < 0.2


def test_chaos_graceful():
    from mlops.chaos import run_chaos
    assert run_chaos()["passed"]              # OOD/blank/corrupt/tiny -> graceful
