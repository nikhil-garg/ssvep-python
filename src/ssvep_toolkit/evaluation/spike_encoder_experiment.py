"""Reusable feature extraction and apparent scoring for spike encoders."""
from __future__ import annotations

from typing import Any


def delta_count_features(filtered_bank: Any, threshold: float, asymmetry: float = 1.0,
                         *, preserve_channels: bool = False) -> Any:
    """Convert `(target, trial, channel, sample)` bands to `(trial, target*2)` counts."""
    import numpy as np

    if threshold <= 0 or asymmetry <= 0:
        raise ValueError("threshold and asymmetry must be positive")
    values = np.asarray(filtered_bank, dtype=float)
    differences = np.diff(values, axis=-1, prepend=values[..., :1])
    if preserve_channels:
        up = np.sum(differences > threshold, axis=-1)
        down = np.sum(differences < -(threshold * asymmetry), axis=-1)
        return np.stack((up, down), axis=-1).transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
    up = np.sum(differences > threshold, axis=(-2, -1))
    down = np.sum(differences < -(threshold * asymmetry), axis=(-2, -1))
    return np.stack((up, down), axis=-1).transpose(1, 0, 2).reshape(values.shape[1], -1)


def lif_count_features(
    filtered_bank: Any,
    sampling_rate_hz: float,
    threshold: float,
    tau_seconds: float,
    *,
    input_gain: float = 1.0,
    preserve_channels: bool = False,
) -> Any:
    """Vectorized LIF counts from target-filtered signals, returned trial-first."""
    import numpy as np

    if sampling_rate_hz <= 0 or threshold <= 0 or tau_seconds <= 0 or input_gain <= 0:
        raise ValueError("sampling rate, threshold, tau, and input gain must be positive")
    values = np.asarray(filtered_bank, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    membrane = np.zeros(flat.shape[0], dtype=float)
    counts = np.zeros(flat.shape[0], dtype=np.int32)
    decay = float(np.exp(-1.0 / (sampling_rate_hz * tau_seconds)))
    for sample in range(flat.shape[1]):
        previous = membrane.copy()
        membrane = decay * membrane + (1.0 - decay) * input_gain * flat[:, sample]
        crossing = (previous < threshold) & (membrane >= threshold)
        counts += crossing
        membrane[crossing] = 0.0
    target, trial, channel = values.shape[:-1]
    shaped = counts.reshape(target, trial, channel)
    return shaped.transpose(1, 0, 2).reshape(trial, -1) if preserve_channels else shaped.sum(axis=-1).T


def apparent_template_result(features: Any, labels: Any) -> tuple[float, Any, Any]:
    """Return same-data accuracy, predictions, and standardized template scores."""
    import numpy as np
    from ssvep_toolkit.algorithms.encoding import template_classification_scores

    truth = np.asarray(labels, dtype=int)
    scores = template_classification_scores(features, truth)
    prediction = np.argmax(scores, axis=1)
    return float(np.mean(prediction == truth)), prediction, scores
