from __future__ import annotations

from typing import Any


def output_sample_count(samples: int, original_rate: int, target_rate: int, method: str) -> int:
    if method == "matlab_compatible":
        factor = original_rate // target_rate
        return (samples + factor - 1) // factor
    # resample_poly returns ceil(samples * up / down)
    return (samples * target_rate + original_rate - 1) // original_rate


def downsample(data: Any, original_rate: int, target_rate: int, method: str) -> Any:
    """Downsample a `(channel, sample)` array along its last axis."""
    if method == "matlab_compatible":
        if original_rate % target_rate:
            raise ValueError("matlab_compatible requires an integer factor")
        return data[..., :: original_rate // target_rate]
    if method == "polyphase":
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("SciPy is required for polyphase resampling") from exc
        return resample_poly(data, target_rate, original_rate, axis=-1)
    raise ValueError(f"unsupported downsampling method: {method}")

