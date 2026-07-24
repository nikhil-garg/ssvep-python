"""Typed data boundaries for SSVEP studies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class EpochBatch:
    """EEG epochs with shape ``(trial, channel, sample)``."""

    data: FloatArray
    labels: IntArray
    groups: IntArray
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    stimulus_frequencies_hz: tuple[float, ...]
    onset_sample: int

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("data must have shape (trial, channel, sample)")
        n_trials, n_channels, _ = self.data.shape
        if self.labels.shape != (n_trials,) or self.groups.shape != (n_trials,):
            raise ValueError("labels and groups must have shape (trial,)")
        if len(self.channel_names) != n_channels:
            raise ValueError("channel_names must match the channel axis")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if not 0 <= self.onset_sample < self.data.shape[-1]:
            raise ValueError("onset_sample must be within the sample axis")


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    signal_unit: str
    n_conditions: int
    n_blocks: int
