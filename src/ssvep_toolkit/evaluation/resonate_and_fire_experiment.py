from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier, ResonateAndFireParameters


def load_resonate_and_fire_data(
    inputs: Sequence[str | Path], frequencies_hz: Sequence[int], condition: int = 2,
    channel_positions: Sequence[int] = (6, 7, 8),
) -> tuple[object, float]:
    """Load `(subject, class, trial, channel, sample)` from preprocessed HDF5."""
    import h5py
    import numpy as np

    subjects = []
    sampling_rate = None
    for item in inputs:
        with h5py.File(item, "r") as source:
            sampling_rate = float(source.attrs["sampling_rate_hz"])
            frequency_axis = list(map(int, source.attrs["frequencies_hz"]))
            selected = [frequency_axis.index(int(frequency)) for frequency in frequencies_hz]
            # channel, sample, class, trial -> class, trial, channel, sample
            values = np.asarray(source["data"][condition - 1, :, :, selected, :])
            values = values.transpose(2, 3, 0, 1)[:, :, channel_positions, :]
            subjects.append(values)
    return np.asarray(subjects), float(sampling_rate)


def load_raw_resonate_and_fire_data(
    inputs: Sequence[str | Path], frequencies_hz: Sequence[int], condition: int = 2,
    *, latency_seconds: float = 0.14, duration_seconds: float = 5.0,
    progress: Callable[[str], None] = print,
) -> tuple[object, float]:
    """Load raw 1000 Hz O1/Oz/O2 epochs without downsampling."""
    import numpy as np

    from ssvep_toolkit.data.matlab import Matlab73Dataset

    sampling_rate = 1000.0
    start = round(latency_seconds * sampling_rate)
    stop = start + round(duration_seconds * sampling_rate)
    subjects = []
    for index, item in enumerate(inputs, 1):
        progress(f"Raw R&F data subject {index}/{len(inputs)}")
        with Matlab73Dataset(item) as source:
            # Source channels 61–64 contain O1, Oz, O2 and one unused channel.
            chunk = source.read_channel_chunk(60, 64)
        # condition, channel, sample, frequency, block
        selected = chunk[condition - 1, :3, start:stop, :, :]
        selected = np.take(selected, np.asarray(frequencies_hz) - 1, axis=2)
        subjects.append(selected.transpose(2, 3, 0, 1))
    return np.asarray(subjects), sampling_rate


def _flatten(data, subject_indices):
    import numpy as np

    selected = data[subject_indices]
    signals = selected.reshape(-1, selected.shape[-2], selected.shape[-1])
    labels = np.tile(np.repeat(np.arange(data.shape[1]), data.shape[2]), len(subject_indices))
    groups = np.repeat(subject_indices, data.shape[1] * data.shape[2])
    return signals, labels, groups


def _accuracy(predictions, labels) -> float:
    import numpy as np

    return float(np.mean(np.asarray(predictions) == labels))


def _fft_baseline(data, frequencies, sampling_rate, durations):
    import numpy as np

    result = np.empty((len(durations),) + data.shape[:3], dtype=int)
    for di, duration in enumerate(durations):
        samples = round(duration * sampling_rate)
        spectrum = np.fft.rfft(data[..., :samples], axis=-1)
        bins = np.fft.rfftfreq(samples, 1 / sampling_rate)
        scores = []
        for frequency in frequencies:
            fundamental = np.argmin(np.abs(bins - frequency))
            harmonic = np.argmin(np.abs(bins - 2 * frequency))
            scores.append((np.abs(spectrum[..., fundamental]) + np.abs(spectrum[..., harmonic])).sum(axis=-1))
        result[di] = np.argmax(np.stack(scores, axis=-1), axis=-1)
    return result


def run_grouped_resonate_and_fire_experiment(
    data,
    sampling_rate_hz: float,
    frequencies_hz: Sequence[float],
    output: str | Path,
    *,
    durations_seconds: Sequence[float] = tuple(x / 2 for x in range(1, 11)),
    damping_grid: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
    threshold_grid: Sequence[float] = (0.002, 0.005, 0.01, 0.02, 0.05),
    outer_folds: int = 5,
    inner_folds: int = 3,
    tuning_duration_seconds: float = 1.0,
    spread_hz: Sequence[float] = (-0.5, 0.0, 0.5),
    harmonics: Sequence[int] = (1, 2, 3),
    harmonic_weights: Sequence[float] | None = None,
    integration_substeps: int = 4,
    refractory_cycles: float = 0.5,
    progress: Callable[[str], None] = print,
) -> Path:
    """Nested subject-grouped evaluation with no test-subject parameter tuning."""
    import numpy as np

    values = np.asarray(data, dtype=float)
    durations = np.asarray(durations_seconds, dtype=float)
    stops = np.asarray(np.round(durations * sampling_rate_hz), dtype=int)
    subject_count, class_count, trial_count = values.shape[:3]
    predictions = np.full((len(durations), subject_count, class_count, trial_count), -1, dtype=int)
    spike_scores = np.zeros((len(durations), subject_count, class_count, trial_count, class_count), dtype=np.float32)
    selected_parameters = np.zeros((outer_folds, 2))
    parameter_scores = np.zeros((outer_folds, len(damping_grid), len(threshold_grid)))
    subjects = np.arange(subject_count)
    for outer in range(outer_folds):
        test_subjects = subjects[subjects % outer_folds == outer]
        train_subjects = subjects[subjects % outer_folds != outer]
        progress(f"R&F outer fold {outer + 1}/{outer_folds}; held out subjects {(test_subjects + 1).tolist()}")
        for ai, damping in enumerate(damping_grid):
            for ti, threshold in enumerate(threshold_grid):
                fold_scores = []
                parameters = ResonateAndFireParameters(damping_alpha=damping, threshold=threshold, integration_substeps=integration_substeps,refractory_cycles=refractory_cycles)
                for inner in range(inner_folds):
                    validation_subjects = train_subjects[train_subjects % inner_folds == inner]
                    inner_train = train_subjects[train_subjects % inner_folds != inner]
                    train_x, train_y, _ = _flatten(values, inner_train)
                    validation_x, validation_y, _ = _flatten(values, validation_subjects)
                    model = OscillatorBankClassifier(frequencies_hz, sampling_rate_hz, parameters, spread_hz=spread_hz,
                                                     harmonics=harmonics, harmonic_weights=harmonic_weights).fit_scaler(train_x)
                    model.fit_calibration(train_x, train_y, round(tuning_duration_seconds * sampling_rate_hz))
                    prediction = model.predict(validation_x, [round(tuning_duration_seconds * sampling_rate_hz)])[0]
                    fold_scores.append(_accuracy(prediction, validation_y))
                parameter_scores[outer, ai, ti] = np.mean(fold_scores)
        best = np.unravel_index(np.argmax(parameter_scores[outer]), parameter_scores[outer].shape)
        parameters = ResonateAndFireParameters(
            damping_alpha=float(damping_grid[best[0]]), threshold=float(threshold_grid[best[1]]),
            integration_substeps=integration_substeps,
            refractory_cycles=refractory_cycles,
        )
        selected_parameters[outer] = (parameters.damping_alpha, parameters.threshold)
        train_x, train_y, _ = _flatten(values, train_subjects)
        test_x, _, _ = _flatten(values, test_subjects)
        model = OscillatorBankClassifier(frequencies_hz, sampling_rate_hz, parameters, spread_hz=spread_hz,
                                         harmonics=harmonics, harmonic_weights=harmonic_weights).fit_scaler(train_x)
        model.fit_calibration(train_x, train_y, stops)
        scores = model.decision_scores(test_x, stops)
        predicted = np.argmax(scores, axis=-1)
        reshaped_prediction = predicted.reshape(len(durations), len(test_subjects), class_count, trial_count)
        reshaped_scores = scores.reshape(len(durations), len(test_subjects), class_count, trial_count, class_count)
        predictions[:, test_subjects] = reshaped_prediction
        spike_scores[:, test_subjects] = reshaped_scores
    truth = np.broadcast_to(np.arange(class_count)[None, None, :, None], predictions.shape)
    subject_accuracy = np.mean(predictions == truth, axis=(2, 3)).T  # subject, duration
    accuracy = subject_accuracy.mean(axis=0)
    fft_predictions = _fft_baseline(values, frequencies_hz, sampling_rate_hz, durations)
    fft_subject_accuracy = np.mean(fft_predictions == truth, axis=(2, 3)).T
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frequencies_hz=np.asarray(frequencies_hz), durations_seconds=durations,
        predictions=predictions, spike_scores=spike_scores,
        subject_accuracy=subject_accuracy, accuracy=accuracy,
        fft_predictions=fft_predictions, fft_subject_accuracy=fft_subject_accuracy,
        selected_parameters=selected_parameters,
        parameter_scores=parameter_scores,
        damping_grid=np.asarray(damping_grid), threshold_grid=np.asarray(threshold_grid),
        sampling_rate_hz=sampling_rate_hz,
        spread_hz=np.asarray(spread_hz), harmonics=np.asarray(harmonics),
        harmonic_weights=np.asarray(harmonic_weights if harmonic_weights is not None else [1/h for h in harmonics]),
        integration_substeps=integration_substeps, refractory_cycles=refractory_cycles, normalized_dynamics=True,
    )
    return output
