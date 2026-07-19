"""Inspectable internal-state traces for spike encoders."""
from __future__ import annotations

from typing import Any

from .spike_encoding import DeltaEncoderParameters, LIFEncoderParameters


def delta_state_trace(signal: Any, parameters: DeltaEncoderParameters) -> dict[str, Any]:
    import numpy as np

    parameters.validate()
    values = np.asarray(signal, dtype=float).reshape(-1)
    change = np.diff(values, prepend=values[:1])
    return {
        "signal": values, "change": change,
        "up_spikes": change > parameters.up_threshold,
        "down_spikes": change < -parameters.down_threshold,
        "up_threshold": parameters.up_threshold,
        "down_threshold": -parameters.down_threshold,
    }


def lif_state_trace(signal: Any, sampling_rate_hz: float,
                    parameters: LIFEncoderParameters) -> dict[str, Any]:
    import numpy as np

    parameters.validate()
    if sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be positive")
    values = np.asarray(signal, dtype=float).reshape(-1)
    membrane = np.empty(values.size, dtype=float)
    spikes = np.zeros(values.size, dtype=bool)
    state = float(parameters.reset_potential)
    decay = float(np.exp(-1 / (sampling_rate_hz * parameters.tau_seconds)))
    for sample, current in enumerate(values):
        previous = state
        state = decay * state + (1 - decay) * parameters.input_gain * current
        membrane[sample] = state
        crossing = previous < parameters.threshold <= state
        if crossing:
            spikes[sample] = True
            state = float(parameters.reset_potential)
    return {"signal": values, "membrane": membrane, "spikes": spikes,
            "threshold": parameters.threshold, "decay": decay}
