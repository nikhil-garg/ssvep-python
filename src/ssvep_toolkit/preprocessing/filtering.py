from __future__ import annotations

from typing import Any


def matlab_cheby_bandpass(
    low_pass_hz: float,
    sampling_rate_hz: float = 250.0,
    high_pass_hz: float = 90.0,
) -> tuple[Any, Any]:
    """Reproduce the Chebyshev-I design used by the MATLAB scripts."""
    from scipy.signal import cheb1ord, cheby1

    if low_pass_hz <= 0 or low_pass_hz >= high_pass_hz:
        raise ValueError("lower cutoff must be positive and below the upper cutoff")
    low_stop = low_pass_hz - (0.3 if low_pass_hz < 10 else 5.0)
    if low_stop <= 0:
        raise ValueError("lower cutoff produces a non-positive stop frequency")
    nyquist = sampling_rate_hz / 2.0
    wp = [low_pass_hz / nyquist, high_pass_hz / nyquist]
    ws = [low_stop / nyquist, 100.0 / nyquist]
    order, wn = cheb1ord(wp, ws, 3, 5)
    return cheby1(order, 0.5, wn, btype="bandpass")


def filter_zero_phase(data: Any, b: Any, a: Any) -> Any:
    from scipy.signal import filtfilt

    return filtfilt(b, a, data, axis=-1)


def harmonic_filter_bank(
    data: Any,
    stimulus_hz: float,
    *,
    first_low_hz: float,
    harmonics: int = 10,
    sampling_rate_hz: float = 250.0,
    maximum_hz: float = 90.0,
) -> Any:
    """Return `(subband, ..., sample)` filtered data as in FBCCA/FBTRCA."""
    import numpy as np

    bands = []
    for harmonic in range(1, harmonics + 1):
        low = first_low_hz if harmonic == 1 else harmonic * stimulus_hz - 0.2
        if harmonic * stimulus_hz >= maximum_hz or low >= maximum_hz:
            bands.append(np.zeros_like(data, dtype=float))
            continue
        b, a = matlab_cheby_bandpass(low, sampling_rate_hz, maximum_hz)
        bands.append(filter_zero_phase(data, b, a))
    return np.stack(bands)

