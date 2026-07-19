"""Interpretable parameterizations for normalized R&F oscillators."""
from __future__ import annotations

from typing import Any


def quality_factor(damping_alpha: Any) -> Any:
    """Approximate Q for normalized dynamics ``omega=2*pi``: ``Q=pi/alpha``."""
    import numpy as np

    alpha = np.asarray(damping_alpha, dtype=float)
    if np.any(alpha <= 0):
        raise ValueError("damping_alpha must be positive")
    return np.pi / alpha


def bandwidth_hz(resonance_frequency_hz: Any, damping_alpha: Any) -> Any:
    """Approximate half-power bandwidth implied by normalized damping."""
    import numpy as np

    frequency = np.asarray(resonance_frequency_hz, dtype=float)
    if np.any(frequency <= 0):
        raise ValueError("resonance frequencies must be positive")
    return frequency / quality_factor(damping_alpha)


def damping_from_bandwidth(resonance_frequency_hz: Any, bandwidth: Any) -> Any:
    """Convert a desired bandwidth in Hz to normalized damping alpha."""
    import numpy as np

    frequency = np.asarray(resonance_frequency_hz, dtype=float)
    width = np.asarray(bandwidth, dtype=float)
    if np.any(frequency <= 0) or np.any(width <= 0):
        raise ValueError("frequency and bandwidth must be positive")
    return np.pi * width / frequency


def effective_drive_threshold_ratio(input_gain: float, amplitude_scale: Any, threshold: float) -> Any:
    """Return the identifiable linear drive/threshold operating ratio."""
    import numpy as np

    if input_gain <= 0 or threshold <= 0:
        raise ValueError("input_gain and threshold must be positive")
    return float(input_gain) * np.asarray(amplitude_scale, dtype=float) / float(threshold)
