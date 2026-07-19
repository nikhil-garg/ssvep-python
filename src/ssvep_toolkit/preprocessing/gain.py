"""Fold-safe and streaming-safe amplitude calibration for spike encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GainCalibration:
    """A fitted branch gain with auditable training provenance."""

    gain_per_unit: Any
    reference_rms: Any
    target_rms: float
    method: str
    training_trials: int


def centered_rms(signals: Any, *, axis: int = -1, epsilon: float = 1e-6) -> Any:
    """Return RMS after removing the sample-axis mean."""
    import numpy as np

    values = np.asarray(signals, dtype=float)
    centered = values - values.mean(axis=axis, keepdims=True)
    return np.maximum(np.sqrt(np.mean(centered * centered, axis=axis)), float(epsilon))


def fit_training_branch_gain(
    signals: Any,
    training_mask: Any,
    *,
    target_rms: float = 0.75,
    method: str = "median",
    epsilon: float = 1e-6,
) -> GainCalibration:
    """Fit one gain per branch using training trials only.

    ``signals`` must be ``(trial, branch, sample)``.  The returned gain never
    depends on a held-out trial and therefore preserves between-trial amplitude.
    """
    import numpy as np

    values = np.asarray(signals, dtype=float)
    mask = np.asarray(training_mask, dtype=bool)
    if values.ndim != 3 or mask.shape != (values.shape[0],):
        raise ValueError("signals must be (trial, branch, sample) with a trial mask")
    if not np.any(mask) or target_rms <= 0:
        raise ValueError("training_mask cannot be empty and target_rms must be positive")
    rms = centered_rms(values[mask], epsilon=epsilon)
    if method == "median":
        reference = np.median(rms, axis=0)
    elif method == "geometric_mean":
        reference = np.exp(np.mean(np.log(np.maximum(rms, epsilon)), axis=0))
    else:
        raise ValueError("method must be 'median' or 'geometric_mean'")
    gain = float(target_rms) / np.maximum(reference, float(epsilon))
    return GainCalibration(gain, reference, float(target_rms), method, int(mask.sum()))


def apply_branch_gain(signals: Any, calibration: GainCalibration) -> Any:
    """Apply a fitted branch gain without changing trial-to-trial amplitudes."""
    import numpy as np

    values = np.asarray(signals, dtype=float)
    if values.ndim != 3 or np.asarray(calibration.gain_per_unit).shape != (values.shape[1],):
        raise ValueError("calibration does not match the branch axis")
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered * np.asarray(calibration.gain_per_unit)[None, :, None]


def fit_prestimulus_branch_gain(
    prestimulus: Any, *, target_rms: float = 0.75, epsilon: float = 1e-6,
) -> Any:
    """Return per-trial/branch gains estimated only from pre-stimulus samples."""
    import numpy as np

    values = np.asarray(prestimulus, dtype=float)
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError("prestimulus must be (trial, branch, sample)")
    return float(target_rms) / np.maximum(centered_rms(values, epsilon=epsilon), epsilon)


def causal_running_gain(
    signals: Any,
    sampling_rate_hz: float,
    *,
    target_rms: float = 0.75,
    tau_seconds: float = 0.5,
    initial_rms: Any | None = None,
    initial_mean: Any | None = None,
    minimum_rms: float = 1e-3,
    maximum_gain: float | None = None,
) -> tuple[Any, Any]:
    """Scale each sample using an RMS state containing past samples only.

    The gain applied at sample ``t`` is computed before incorporating sample
    ``t`` into the state.  This makes the transform strictly streaming-causal.
    Returns ``(adapted_signal, gain_trace)``.
    """
    import numpy as np

    values = np.asarray(signals, dtype=float)
    if values.ndim != 3 or sampling_rate_hz <= 0 or tau_seconds <= 0 or target_rms <= 0:
        raise ValueError("invalid signal shape or gain parameters")
    decay = float(np.exp(-1.0 / (sampling_rate_hz * tau_seconds)))
    if initial_rms is None:
        state = np.maximum(np.abs(values[..., 0]), minimum_rms) ** 2
    else:
        initial = np.asarray(initial_rms, dtype=float)
        if initial.shape != values.shape[:2]:
            raise ValueError("initial_rms must match (trial, branch)")
        state = np.maximum(initial, minimum_rms) ** 2
    if initial_mean is None:
        running_mean = values[..., 0].copy()
    else:
        running_mean = np.asarray(initial_mean, dtype=float)
        if running_mean.shape != values.shape[:2]:
            raise ValueError("initial_mean must match (trial, branch)")
    adapted = np.empty_like(values, dtype=float)
    trace = np.empty_like(values, dtype=float)
    for sample in range(values.shape[-1]):
        gain = float(target_rms) / np.sqrt(np.maximum(state, minimum_rms**2))
        if maximum_gain is not None:
            gain = np.minimum(gain, float(maximum_gain))
        trace[..., sample] = gain
        innovation = values[..., sample] - running_mean
        adapted[..., sample] = innovation * gain
        state = decay * state + (1.0 - decay) * innovation * innovation
        running_mean = decay * running_mean + (1.0 - decay) * values[..., sample]
    return adapted, trace
