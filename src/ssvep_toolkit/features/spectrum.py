from __future__ import annotations

from typing import Any


def amplitude_spectrum(data: Any, sampling_rate_hz: float) -> tuple[Any, Any]:
    """One-sided amplitude spectrum along the final array axis."""
    import numpy as np

    values = np.asarray(data)
    n = values.shape[-1]
    transformed = np.fft.rfft(values, axis=-1)
    amplitude = np.abs(transformed) * (2.0 / n)
    amplitude[..., 0] *= 0.5
    if n % 2 == 0:
        amplitude[..., -1] *= 0.5
    frequencies = np.fft.rfftfreq(n, 1.0 / sampling_rate_hz)
    return frequencies, amplitude


def signal_to_noise_ratio(amplitude: Any, noise_bins: int = 5, epsilon: float = 1e-12) -> Any:
    """Spectral SNR in dB using neighboring bins on both sides.

    The target bin itself is excluded. Edge bins without a complete symmetric
    neighborhood are returned as NaN.
    """
    import numpy as np

    values = np.asarray(amplitude, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    for index in range(noise_bins, values.shape[-1] - noise_bins):
        left = values[..., index - noise_bins:index]
        right = values[..., index + 1:index + noise_bins + 1]
        noise = np.mean(np.concatenate((left, right), axis=-1), axis=-1)
        result[..., index] = 20.0 * np.log10((values[..., index] + epsilon) / (noise + epsilon))
    return result

