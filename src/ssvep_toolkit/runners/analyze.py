from __future__ import annotations

from pathlib import Path
from typing import Callable

from ssvep_toolkit.algorithms import information_transfer_rate
from ssvep_toolkit.evaluation import evaluate_fbcca, evaluate_fbtrca, evaluate_trca


def run_classification(
    input_path: str | Path,
    output_path: str | Path,
    algorithm: str,
    *,
    duration_seconds: float = 1.0,
    first_low_hz: float = 6.0,
    subbands: int = 5,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
    rest_seconds: float = 1.0,
    progress: Callable[[str], None] = print,
) -> Path:
    import h5py
    import numpy as np

    input_path = Path(input_path)
    output_path = Path(output_path)
    with h5py.File(input_path, "r") as source:
        sampling_rate = float(source.attrs["sampling_rate_hz"])
        frequencies = np.asarray(source.attrs["frequencies_hz"], dtype=float)
        conditions = np.asarray(source.attrs["conditions"], dtype=int)
        samples = round(duration_seconds * sampling_rate)
        if samples <= 0 or samples > source["data"].shape[2]:
            raise ValueError("duration is outside the available recording")
        if len(frequencies) < 2:
            raise ValueError("classification requires at least two selected frequencies")
        predictions = []
        accuracies = []
        for condition_index, condition in enumerate(conditions):
            # Stored: channel,sample,class,trial; evaluator: class,trial,channel,sample.
            values = np.asarray(source["data"][condition_index, :, :samples, :, :])
            values = values.transpose(2, 3, 0, 1)
            progress(f"condition {condition}: {algorithm}")
            if algorithm == "fbcca":
                pred, accuracy = evaluate_fbcca(
                    values, frequencies, sampling_rate, first_low_hz=first_low_hz,
                    subbands=subbands, weight_a=weight_a, weight_b=weight_b,
                )
            elif algorithm == "trca":
                pred, accuracy = evaluate_trca(values)
            elif algorithm == "fbtrca":
                pred, accuracy = evaluate_fbtrca(
                    values, frequencies, sampling_rate, first_low_hz=first_low_hz,
                    subbands=subbands, weight_a=weight_a, weight_b=weight_b,
                )
            else:
                raise ValueError("algorithm must be fbcca, trca, or fbtrca")
            predictions.append(pred)
            accuracies.append(accuracy)
    selections = 60.0 / (duration_seconds + rest_seconds)
    itr = information_transfer_rate(len(frequencies), np.asarray(accuracies), selections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        algorithm=algorithm,
        conditions=conditions,
        frequencies_hz=frequencies,
        predictions=np.asarray(predictions),
        accuracy=np.asarray(accuracies),
        itr=np.asarray(itr),
        duration_seconds=duration_seconds,
        sampling_rate_hz=sampling_rate,
    )
    return output_path

