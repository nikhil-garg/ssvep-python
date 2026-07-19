from __future__ import annotations

from typing import Any


def bits_per_selection(classes: int, accuracy: Any, *, clamp_below_chance: bool = False) -> Any:
    """Return Wolpaw information per selection in bits."""
    import numpy as np

    p = np.asarray(accuracy, dtype=float)
    if classes < 2 or np.any((p < 0) | (p > 1)):
        raise ValueError("classes must be at least 2 and accuracy must be in [0, 1]")
    if clamp_below_chance:
        p = np.maximum(p, 1.0 / classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            np.log2(classes) + p * np.log2(p)
            + (1 - p) * np.log2((1 - p) / (classes - 1))
        )
    result = np.where(p == 0, np.log2(classes / (classes - 1)), result)
    result = np.where(p == 1, np.log2(classes), result)
    if clamp_below_chance:
        result = np.maximum(result, 0.0)
    return float(result) if result.ndim == 0 else result


def information_transfer_rate(classes: int, accuracy: Any, selections_per_minute: float) -> Any:
    """MATLAB `ITR.m`, returning bits/minute."""
    import numpy as np

    p = np.asarray(accuracy, dtype=float)
    rate = np.asarray(selections_per_minute, dtype=float)
    if classes < 2 or np.any((p < 0) | (p > 1)) or np.any(rate < 0):
        raise ValueError("invalid ITR arguments")
    result = rate * bits_per_selection(classes, p)
    return float(result) if result.ndim == 0 else result


def latency_aware_itr(
    classes: int,
    accuracy: Any,
    decision_seconds: Any,
    *,
    onset_latency_seconds: float = 0.0,
    gaze_shift_seconds: float = 0.0,
    inter_command_seconds: float = 0.0,
    clamp_below_chance: bool = True,
) -> Any:
    """Return bits/minute using complete onset-to-next-command time."""
    import numpy as np

    decision = np.asarray(decision_seconds, dtype=float)
    overheads = (onset_latency_seconds, gaze_shift_seconds, inter_command_seconds)
    if np.any(decision <= 0) or min(overheads) < 0:
        raise ValueError("decision time must be positive and timing overheads nonnegative")
    result = 60.0 / (decision + sum(overheads)) * bits_per_selection(
        classes, accuracy, clamp_below_chance=clamp_below_chance,
    )
    return float(result) if np.asarray(result).ndim == 0 else result


def latency_itr_report(
    classes: int,
    accuracy: Any,
    decision_seconds: Any,
    *,
    onset_latency_seconds: float = 0.14,
    practical_overhead_seconds: float = 1.0,
) -> dict[str, Any]:
    """Return ideal neural-window and practical command-cycle ITR curves."""
    import numpy as np

    decision = np.asarray(decision_seconds, dtype=float)
    return {
        "decision_seconds": decision,
        "bits_per_selection": bits_per_selection(classes, accuracy, clamp_below_chance=True),
        "neural_window_itr_bits_per_minute": latency_aware_itr(
            classes, accuracy, decision, onset_latency_seconds=onset_latency_seconds,
        ),
        "practical_itr_bits_per_minute": latency_aware_itr(
            classes, accuracy, decision, onset_latency_seconds=onset_latency_seconds,
            gaze_shift_seconds=practical_overhead_seconds,
        ),
        "onset_latency_seconds": float(onset_latency_seconds),
        "practical_overhead_seconds": float(practical_overhead_seconds),
    }
