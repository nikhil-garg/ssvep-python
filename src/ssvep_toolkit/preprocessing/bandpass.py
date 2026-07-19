"""General Butterworth band-pass preprocessing for neural encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BandpassParameters:
    enabled: bool = False
    order: int = 5
    half_width_hz: float = 1.0
    zero_phase: bool = True
    minimum_hz: float = 0.1

    def validate(self, sampling_rate_hz: float | None = None) -> None:
        if self.order < 1:
            raise ValueError("band-pass order must be at least 1")
        if self.half_width_hz <= 0:
            raise ValueError("band-pass half-width must be positive")
        if self.minimum_hz <= 0:
            raise ValueError("band-pass minimum frequency must be positive")
        if sampling_rate_hz is not None and sampling_rate_hz <= 0:
            raise ValueError("sampling rate must be positive")


def butterworth_bandpass(
    data: Any,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 5,
    zero_phase: bool = True,
) -> Any:
    """Filter along the last axis using a numerically stable SOS design."""
    import numpy as np
    from scipy.signal import sosfiltfilt

    values = np.asarray(data, dtype=float)
    sos = butterworth_sos(sampling_rate_hz, low_hz, high_hz, order=order)
    return sosfiltfilt(sos, values, axis=-1) if zero_phase else butterworth_bandpass_stream(values, sos)[0]


def butterworth_sos(sampling_rate_hz: float, low_hz: float, high_hz: float, *, order: int = 5) -> Any:
    """Design a stable Butterworth band-pass for offline or streaming use."""
    from scipy.signal import butter
    nyquist = sampling_rate_hz / 2.0
    if order < 1: raise ValueError("band-pass order must be at least 1")
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"cutoffs must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)")
    return butter(order, (low_hz, high_hz), btype="bandpass", fs=sampling_rate_hz, output="sos")


def butterworth_bandpass_stream(data: Any, sos: Any, state: Any | None = None) -> tuple[Any, Any]:
    """Filter one causal chunk and return state for the next chunk.

    Leading dimensions are treated as independent streams. A zero state is
    used only for the first chunk; passing the returned state makes chunked and
    one-shot causal filtering numerically identical.
    """
    import numpy as np
    from scipy.signal import sosfilt
    values = np.asarray(data, dtype=float); coefficients = np.asarray(sos, dtype=float)
    if values.ndim < 1: raise ValueError("data must include a sample axis")
    flat = values.reshape(-1, values.shape[-1])
    if state is None:
        zi = np.zeros((coefficients.shape[0], flat.shape[0], 2), dtype=float)
    else:
        zi = np.asarray(state, dtype=float)
        if zi.shape != (coefficients.shape[0], flat.shape[0], 2):
            raise ValueError("streaming filter state does not match SOS sections and signal streams")
    filtered, final_state = sosfilt(coefficients, flat, axis=-1, zi=zi)
    return filtered.reshape(values.shape), final_state


def target_frequency_filter_bank(
    data: Any,
    stimulus_frequencies_hz: Sequence[float],
    sampling_rate_hz: float,
    parameters: BandpassParameters = BandpassParameters(enabled=True),
) -> Any:
    """Return `(target_frequency, ..., sample)` bands centered on each target."""
    import numpy as np

    parameters.validate(sampling_rate_hz)
    values = np.asarray(data, dtype=float)
    if not parameters.enabled:
        return np.broadcast_to(values, (len(stimulus_frequencies_hz),) + values.shape).copy()
    bands = []
    for frequency in stimulus_frequencies_hz:
        low = max(float(frequency) - parameters.half_width_hz, parameters.minimum_hz)
        high = float(frequency) + parameters.half_width_hz
        bands.append(butterworth_bandpass(
            values, sampling_rate_hz, low, high, order=parameters.order,
            zero_phase=parameters.zero_phase,
        ))
    return np.stack(bands)
