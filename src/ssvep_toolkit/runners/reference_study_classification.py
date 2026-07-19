from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence


def _mat73(path: Path, variable: str):
    import h5py
    import numpy as np

    with h5py.File(path, "r") as source:
        value = np.asarray(source[variable])
    return value.transpose(tuple(range(value.ndim - 1, -1, -1)))


def compute_figure10_cohort(
    inputs: Sequence[str | Path], parameter_dir: str | Path, output: str | Path,
    *, checkpoint_dir: str | Path | None = None, progress: Callable[[str], None] = print,
) -> Path:
    """Reference-study FBCCA sweep at 0.1–5.0 s for Figures 10 and 11."""
    import h5py
    import numpy as np

    from ssvep_toolkit.algorithms import canonical_correlations, information_transfer_rate, reference_signals
    from ssvep_toolkit.preprocessing.filtering import harmonic_filter_bank

    parameter_dir = Path(parameter_dir)
    cutoffs = _mat73(parameter_dir / "ccabp_m.mat", "ccabp_m")
    parameters = _mat73(parameter_dir / "fbcca_abn.mat", "fbcca_abn")
    bands = np.asarray([[2, 3, 4, 5], [12, 13, 14, 15], [40, 41, 42, 43]])
    durations = np.arange(1, 51) / 10
    checkpoint = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint:
        checkpoint.mkdir(parents=True, exist_ok=True)
    subject_accuracy = []
    for subject_index, item in enumerate(inputs, 1):
        saved = checkpoint / f"subject_{subject_index:02d}.npy" if checkpoint else None
        if saved and saved.exists():
            subject_accuracy.append(np.load(saved))
            progress(f"Figure 10 subject {subject_index}/{len(inputs)}: resumed")
            continue
        progress(f"Figure 10 subject {subject_index}/{len(inputs)}")
        with h5py.File(item, "r") as source:
            all_data = source["data"]
            fs = float(source.attrs["sampling_rate_hz"])
            accuracy = np.zeros((2, 3, 50))
            for condition in range(2):
                for band_index, frequencies in enumerate(bands):
                    # class, trial, channel, sample
                    data = np.asarray(all_data[condition, :, :, frequencies - 1, :]).transpose(2, 3, 0, 1)
                    count = int(parameters[condition, band_index, 2])
                    filtered = []
                    for cls, frequency in enumerate(frequencies):
                        filtered.append(harmonic_filter_bank(
                            data[cls], float(frequency), first_low_hz=float(cutoffs[condition, band_index]),
                            harmonics=count, sampling_rate_hz=fs,
                        ))
                    filtered = np.asarray(filtered)  # class, subband, trial, channel, sample
                    refs = reference_signals(frequencies, 1250, fs, 10)
                    weights = np.arange(1, count + 1) ** (-parameters[condition, band_index, 0]) + parameters[condition, band_index, 1]
                    for time_index, duration in enumerate(durations):
                        samples = round(duration * fs)
                        correct = 0
                        for target in range(4):
                            for trial in range(12):
                                correlations = np.zeros((count, 4))
                                for subband in range(count):
                                    test = filtered[target, subband, trial, :, :samples]
                                    for candidate in range(4):
                                        correlations[subband, candidate] = canonical_correlations(test, refs[candidate, :, :samples])[0]
                                scores = weights @ (correlations ** 2)
                                correct += int(np.argmax(scores) == target)
                        accuracy[condition, band_index, time_index] = correct / 48
        if saved:
            np.save(saved, accuracy)
        subject_accuracy.append(accuracy)
    accuracy = np.asarray(subject_accuracy)
    itr = information_transfer_rate(4, accuracy, 60 / (durations[None, None, None, :] + 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=durations, accuracy=accuracy, itr=itr, bands=bands)
    return output


def compute_figure12_cohort(
    inputs: Sequence[str | Path], parameter_dir: str | Path, output: str | Path,
    *, checkpoint_dir: str | Path, progress: Callable[[str], None] = print,
) -> Path:
    """Reference-study four-phase FBTRCA sweep for Figures 12 and 13."""
    import h5py
    import numpy as np

    from ssvep_toolkit.algorithms import information_transfer_rate
    from ssvep_toolkit.evaluation import evaluate_phase_fbtrca

    parameter_dir = Path(parameter_dir)
    cutoffs = _mat73(parameter_dir / "parabp_m.mat", "parabp_m")
    parameters = _mat73(parameter_dir / "para_abn.mat", "para_abn")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_accuracy = []
    for subject_index, item in enumerate(inputs, 1):
        saved = checkpoint_dir / f"subject_{subject_index:02d}.npz"
        accuracy = np.zeros((2, 60))
        completed = np.zeros((2, 60), dtype=bool)
        if saved.exists():
            with np.load(saved) as prior:
                accuracy = prior["accuracy"]
                completed = prior["completed"]
        with h5py.File(item, "r") as source:
            for condition in range(2):
                for frequency_index in range(60):
                    if completed[condition, frequency_index]:
                        continue
                    frequency = frequency_index + 1
                    progress(f"Figure 12 subject {subject_index}/{len(inputs)}, condition {condition + 1}, {frequency} Hz")
                    # block, channel, sample
                    recordings = np.asarray(source["data"][condition, :, :, frequency_index, :]).transpose(2, 0, 1)
                    a, b, count = parameters[condition, frequency_index]
                    _, accuracy[condition, frequency_index] = evaluate_phase_fbtrca(
                        recordings, frequency, 250, first_low_hz=float(cutoffs[condition, frequency_index]),
                        subbands=int(count), weight_a=float(a), weight_b=float(b),
                    )
                    completed[condition, frequency_index] = True
                    np.savez_compressed(saved, accuracy=accuracy, completed=completed)
        all_accuracy.append(accuracy)
    accuracy = np.asarray(all_accuracy)
    itr = information_transfer_rate(4, accuracy, 30.0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, frequencies_hz=np.arange(1, 61), accuracy=accuracy, itr=itr)
    return output
