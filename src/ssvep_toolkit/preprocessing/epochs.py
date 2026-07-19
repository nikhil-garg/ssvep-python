from __future__ import annotations

from typing import Any


def crop_latency(data: Any, seconds: float, sampling_rate_hz: float) -> Any:
    samples = round(seconds * sampling_rate_hz)
    if samples >= data.shape[-1]:
        raise ValueError("latency crop removes all samples")
    return data[..., samples:]


def phase_shifted_epochs(
    data: Any,
    stimulus_hz: float,
    *,
    sampling_rate_hz: int = 250,
    duration_seconds: float = 1.0,
) -> Any:
    """Reproduce the four MATLAB phase-shifted one-second epochs.

    Input ends in samples; output is `(phase, ..., epoch_sample)`.
    """
    import numpy as np

    length = round(duration_seconds * sampling_rate_hz)
    n1 = round(sampling_rate_hz / (4 * stimulus_hz))
    n2 = round(sampling_rate_hz / (2 * stimulus_hz))
    n3 = round(3 * sampling_rate_hz / (4 * stimulus_hz))
    starts = [0, sampling_rate_hz + n1, 2 * sampling_rate_hz + n2, 3 * sampling_rate_hz + n3]
    epochs = []
    for start in starts:
        stop = start + length
        if stop > data.shape[-1]:
            raise ValueError("recording is too short for phase-shifted epochs")
        epochs.append(data[..., start:stop])
    return np.stack(epochs)

