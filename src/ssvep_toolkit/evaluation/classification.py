from __future__ import annotations

from typing import Any, Sequence

from ssvep_toolkit.algorithms import (
    fit_fbtrca,
    fit_trca,
    predict_fbcca,
    predict_fbtrca,
    predict_trca,
    reference_signals,
)
from ssvep_toolkit.preprocessing.filtering import harmonic_filter_bank
from ssvep_toolkit.preprocessing.epochs import phase_shifted_epochs


def evaluate_fbcca(
    data: Any,
    frequencies_hz: Sequence[float],
    sampling_rate_hz: float,
    *,
    first_low_hz: float,
    harmonics: int = 10,
    subbands: int = 5,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
) -> tuple[Any, float]:
    """Evaluate `(class, trial, channel, sample)` data."""
    import numpy as np

    values = np.asarray(data, dtype=float)
    refs = reference_signals(frequencies_hz, values.shape[-1], sampling_rate_hz, harmonics)
    predictions = np.empty((values.shape[0], values.shape[1]), dtype=int)
    for cls in range(values.shape[0]):
        for trial in range(values.shape[1]):
            bands = harmonic_filter_bank(
                values[cls, trial], frequencies_hz[cls], first_low_hz=first_low_hz,
                harmonics=subbands, sampling_rate_hz=sampling_rate_hz,
            )
            predictions[cls, trial] = predict_fbcca(
                bands, refs, weight_a=weight_a, weight_b=weight_b, subband_count=subbands
            )
    truth = np.arange(values.shape[0])[:, None]
    return predictions, float(np.mean(predictions == truth))


def evaluate_trca(data: Any) -> tuple[Any, float]:
    """Leave-one-block-out TRCA for `(class, trial, channel, sample)` data."""
    import numpy as np

    values = np.asarray(data, dtype=float)
    predictions = np.empty((values.shape[0], values.shape[1]), dtype=int)
    for held_out in range(values.shape[1]):
        model = fit_trca(np.delete(values, held_out, axis=1))
        for cls in range(values.shape[0]):
            predictions[cls, held_out] = predict_trca(model, values[cls, held_out])
    truth = np.arange(values.shape[0])[:, None]
    return predictions, float(np.mean(predictions == truth))


def evaluate_fbtrca(
    data: Any,
    frequencies_hz: Sequence[float],
    sampling_rate_hz: float,
    *,
    first_low_hz: float,
    subbands: int = 5,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
) -> tuple[Any, float]:
    """Leave-one-block-out FBTRCA for `(class, trial, channel, sample)` data."""
    import numpy as np

    values = np.asarray(data, dtype=float)
    filtered = np.empty((subbands,) + values.shape, dtype=float)
    for cls, frequency in enumerate(frequencies_hz):
        for trial in range(values.shape[1]):
            filtered[:, cls, trial] = harmonic_filter_bank(
                values[cls, trial], frequency, first_low_hz=first_low_hz,
                harmonics=subbands, sampling_rate_hz=sampling_rate_hz,
            )
    predictions = np.empty((values.shape[0], values.shape[1]), dtype=int)
    for held_out in range(values.shape[1]):
        model = fit_fbtrca(
            np.delete(filtered, held_out, axis=2), weight_a=weight_a, weight_b=weight_b
        )
        for cls in range(values.shape[0]):
            predictions[cls, held_out] = predict_fbtrca(model, filtered[:, cls, held_out])
    truth = np.arange(values.shape[0])[:, None]
    return predictions, float(np.mean(predictions == truth))


def evaluate_phase_fbtrca(
    recordings: Any,
    stimulus_hz: float,
    sampling_rate_hz: float,
    *,
    first_low_hz: float,
    subbands: int = 5,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
) -> tuple[Any, float]:
    """Paper-compatible four-phase FBTRCA for `(block, channel, sample)` data."""
    import numpy as np

    values = np.asarray(recordings, dtype=float)
    if values.ndim != 3:
        raise ValueError("recordings must have shape (block, channel, sample)")
    # filter -> (block, subband, channel, sample), then epoch -> phase
    filtered = np.stack([
        harmonic_filter_bank(
            block, stimulus_hz, first_low_hz=first_low_hz, harmonics=subbands,
            sampling_rate_hz=sampling_rate_hz,
        )
        for block in values
    ])
    phase_data = np.stack([
        phase_shifted_epochs(bands, stimulus_hz, sampling_rate_hz=round(sampling_rate_hz))
        for bands in filtered
    ])
    # block, phase, subband, channel, sample -> subband, phase, block, channel, sample
    subband_trials = phase_data.transpose(2, 1, 0, 3, 4)
    predictions = np.empty((4, values.shape[0]), dtype=int)
    for held_out in range(values.shape[0]):
        model = fit_fbtrca(
            np.delete(subband_trials, held_out, axis=2), weight_a=weight_a, weight_b=weight_b
        )
        for phase in range(4):
            predictions[phase, held_out] = predict_fbtrca(model, subband_trials[:, phase, held_out])
    truth = np.arange(4)[:, None]
    return predictions, float(np.mean(predictions == truth))
