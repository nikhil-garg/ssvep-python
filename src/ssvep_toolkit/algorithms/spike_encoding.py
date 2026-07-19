"""Delta and leaky-integrate-and-fire EEG-to-spike encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from ssvep_toolkit.preprocessing.bandpass import BandpassParameters, target_frequency_filter_bank


@dataclass(frozen=True)
class DeltaEncoderParameters:
    threshold: float
    asymmetry: float = 1.0

    def validate(self) -> None:
        if self.threshold <= 0:
            raise ValueError("delta threshold must be positive")
        if self.asymmetry <= 0:
            raise ValueError("delta asymmetry must be positive")

    @property
    def up_threshold(self) -> float:
        return self.threshold

    @property
    def down_threshold(self) -> float:
        return self.threshold * self.asymmetry


@dataclass(frozen=True)
class LIFEncoderParameters:
    threshold: float
    tau_seconds: float = 0.02
    input_gain: float = 1.0
    reset_potential: float = 0.0

    def validate(self) -> None:
        if self.threshold <= 0:
            raise ValueError("LIF threshold must be positive")
        if self.tau_seconds <= 0:
            raise ValueError("LIF tau_seconds must be positive")
        if self.input_gain <= 0:
            raise ValueError("LIF input_gain must be positive")
        if self.reset_potential >= self.threshold:
            raise ValueError("LIF reset potential must be below threshold")


def delta_encode(data: Any, parameters: DeltaEncoderParameters) -> Any:
    """Return binary `(..., stream, sample)` spikes; stream 0=UP, 1=DN."""
    import numpy as np

    parameters.validate()
    values = np.asarray(data, dtype=float)
    differences = np.diff(values, axis=-1, prepend=values[..., :1])
    up = differences > parameters.up_threshold
    down = differences < -parameters.down_threshold
    return np.stack((up, down), axis=-2).astype(np.uint8)


def lif_encode(data: Any, sampling_rate_hz: float, parameters: LIFEncoderParameters) -> Any:
    """Return binary `(..., 1, sample)` spikes from upward threshold crossings.

    The exact discrete leak update is `v <- decay*v + (1-decay)*gain*x`.
    After a crossing the membrane is reset before the next EEG sample.
    """
    import numpy as np

    parameters.validate()
    if sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be positive")
    values = np.asarray(data, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    membrane = np.full(flat.shape[0], parameters.reset_potential, dtype=float)
    spikes = np.zeros(flat.shape, dtype=np.uint8)
    decay = float(np.exp(-1.0 / (sampling_rate_hz * parameters.tau_seconds)))
    for sample in range(flat.shape[1]):
        previous = membrane.copy()
        membrane = decay * membrane + (1.0 - decay) * parameters.input_gain * flat[:, sample]
        crossing = (previous < parameters.threshold) & (membrane >= parameters.threshold)
        spikes[crossing, sample] = 1
        membrane[crossing] = parameters.reset_potential
    return spikes.reshape(values.shape[:-1] + (1, values.shape[-1]))


def encode_target_frequency_bank(
    data: Any,
    stimulus_frequencies_hz: Sequence[float],
    sampling_rate_hz: float,
    *,
    encoder: Literal["delta", "lif"],
    delta_parameters: DeltaEncoderParameters | None = None,
    lif_parameters: LIFEncoderParameters | None = None,
    bandpass: BandpassParameters = BandpassParameters(enabled=True),
) -> Any:
    """Band-pass once per target, then give every band its own encoder."""
    filtered = target_frequency_filter_bank(data, stimulus_frequencies_hz, sampling_rate_hz, bandpass)
    if encoder == "delta":
        if delta_parameters is None:
            raise ValueError("delta_parameters are required for the delta encoder")
        return delta_encode(filtered, delta_parameters)
    if encoder == "lif":
        if lif_parameters is None:
            raise ValueError("lif_parameters are required for the LIF encoder")
        return lif_encode(filtered, sampling_rate_hz, lif_parameters)
    raise ValueError("encoder must be 'delta' or 'lif'")
