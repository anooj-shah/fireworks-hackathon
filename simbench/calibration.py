"""Optional add-on: blend a predicted distribution toward uniform to counter
overconfidence. TVD (the SimBench metric) punishes confident wrong answers harder
than it rewards confident right ones, so a small blend often helps.

Not wired into scripts/run_eval.py yet - call `blend()` on a row's
Response_Distribution if you want to try it.
"""
from __future__ import annotations


def uniform_distribution(n: int) -> list[float]:
    return [1.0 / n] * n if n > 0 else []


def blend(predicted: list[float], baseline: list[float], alpha: float = 0.0) -> list[float]:
    """alpha=0 returns `predicted` unchanged; alpha=1 returns `baseline` unchanged."""
    if len(predicted) != len(baseline):
        raise ValueError("predicted and baseline distributions must have the same length")
    blended = [(1 - alpha) * p + alpha * b for p, b in zip(predicted, baseline)]
    total = sum(blended)
    return [v / total for v in blended]
