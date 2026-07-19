"""Uniform public interface for EEG-to-spike encoders.

Low-level encoder implementations remain independently testable.  This module
normalizes their outputs into `(trial, target)` spike-count features suitable
for classification and later multi-encoder fusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ssvep_toolkit.algorithms.resonate_and_fire import (
    OscillatorBankClassifier,
    ResonateAndFireParameters,
)
from ssvep_toolkit.algorithms.spike_encoding import (
    DeltaEncoderParameters,
    LIFEncoderParameters,
    encode_target_frequency_bank,
)
from ssvep_toolkit.preprocessing.bandpass import BandpassParameters


EncoderKind = Literal["resonate_fire", "delta", "lif"]


@dataclass(frozen=True)
class EncoderConfig:
    kind: EncoderKind
    bandpass: BandpassParameters | None = None
    resonate_fire: ResonateAndFireParameters | None = None
    delta: DeltaEncoderParameters | None = None
    lif: LIFEncoderParameters | None = None
    harmonics: tuple[int, ...] = (1,)
    spread_hz: tuple[float, ...] = (0.0,)

    def resolved_bandpass(self) -> BandpassParameters:
        if self.bandpass is not None:
            return self.bandpass
        return BandpassParameters(enabled=self.kind != "resonate_fire")

    def validate(self) -> None:
        self.resolved_bandpass().validate()
        if self.kind == "resonate_fire" and self.resonate_fire is None:
            raise ValueError("resonate_fire parameters are required")
        if self.kind == "delta" and self.delta is None:
            raise ValueError("delta parameters are required")
        if self.kind == "lif" and self.lif is None:
            raise ValueError("LIF parameters are required")


@dataclass(frozen=True)
class SpikeFeatures:
    counts: Any
    target_frequencies_hz: Any
    encoder: EncoderKind
    stream_names: tuple[str, ...]
    spikes: Any | None = field(default=None, repr=False)


def encode_spike_features(
    signals: Any,
    stimulus_frequencies_hz: Sequence[float],
    sampling_rate_hz: float,
    config: EncoderConfig,
    *,
    retain_spikes: bool = False,
) -> SpikeFeatures:
    """Encode `(trial, channel, sample)` signals into `(trial, target)` counts."""
    import numpy as np

    config.validate()
    values = np.asarray(signals, dtype=float)
    if values.ndim != 3:
        raise ValueError("signals must have shape (trial, channel, sample)")
    frequencies = np.asarray(stimulus_frequencies_hz, dtype=float)
    if config.kind == "resonate_fire":
        if config.resolved_bandpass().enabled:
            raise ValueError(
                "target-specific prefiltering is intentionally unsupported for R&F; "
                "leave bandpass disabled to avoid target-label leakage"
            )
        model = OscillatorBankClassifier(
            frequencies, sampling_rate_hz, config.resonate_fire,
            harmonics=config.harmonics, spread_hz=config.spread_hz,
        )
        model.channel_scale_ = np.ones(values.shape[1])
        counts = model.scores(values, (values.shape[-1],))[0]
        return SpikeFeatures(counts, frequencies, config.kind, ("R&F",), None)

    spikes = encode_target_frequency_bank(
        values, frequencies, sampling_rate_hz, encoder=config.kind,
        delta_parameters=config.delta, lif_parameters=config.lif,
        bandpass=config.resolved_bandpass(),
    )
    # target, trial, channel, stream, sample -> trial, target*stream.
    # Preserve delta polarity rather than erasing UP/DN asymmetry.
    stream_counts = np.sum(spikes, axis=(-3, -1)).transpose(1, 0, 2)
    counts = stream_counts.reshape(values.shape[0], -1)
    names = ("UP", "DN") if config.kind == "delta" else ("LIF",)
    return SpikeFeatures(counts, frequencies, config.kind, names, spikes if retain_spikes else None)


def template_classification_scores(features: Any, labels: Any) -> Any:
    """Same-data standardized class-template scores for an encoder feature map."""
    import numpy as np

    values = np.asarray(features, dtype=float)
    truth = np.asarray(labels, dtype=int)
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=1), 1e-6)
    standardized = (values - center) / scale
    classes = np.unique(truth)
    templates = np.stack([standardized[truth == label].mean(axis=0) for label in classes])
    return -np.mean((standardized[:, None, :] - templates[None, :, :]) ** 2, axis=-1)
