"""Independent same-data R&F optimization for all 30 subjects.

This intentionally performs no held-out evaluation. Each subject's scaler,
templates, parameters, and reported predictions use that same subject's data.
The result is therefore apparent/training accuracy, not generalization accuracy.
Checkpoints make the long run safely resumable.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import time

import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import (
    OscillatorBankClassifier,
    ResonateAndFireParameters,
)
from ssvep_toolkit.data.matlab import Matlab73Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
OUT = ROOT / "outputs/experiments/resonate_and_fire_30_subject_apparent_accuracy"
CHECKPOINTS = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

SUBJECT_IDS = np.arange(1, 31)
CLASS_COUNTS = np.array((2, 4, 8, 16, 32))
POOL = np.arange(8, 40)
CLASS_SETS = [np.rint(np.linspace(8, 39, count)).astype(int) for count in CLASS_COUNTS]
DURATIONS = np.arange(0.5, 5.01, 0.5)
STOPS = np.rint(1000 * DURATIONS).astype(int)
HARMONIC_SETS = ((1,), (1, 2), (1, 2, 3))
DAMPING_GRID = (0.05, 0.1)
THRESHOLD_GRID = (0.02, 0.05, 0.1)
CANDIDATES = list(product(DAMPING_GRID, THRESHOLD_GRID, range(len(HARMONIC_SETS))))


def load_subject(subject_id: int) -> np.ndarray:
    """Return frequency, block, channel, sample at 1000 Hz."""
    path = DATA_ROOT / f"data_s{subject_id}_64.mat"
    with Matlab73Dataset(path) as source:
        chunk = source.read_channel_chunk(60, 64)
    start = 140
    selected = chunk[1, :3, start : start + 5000, :, :]
    return selected.transpose(2, 3, 0, 1).astype(np.float32, copy=False)


def flatten(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_count, block_count, channel_count, sample_count = values.shape
    return values.reshape(-1, channel_count, sample_count), np.repeat(np.arange(class_count), block_count)


def overlap_coefficient(target: np.ndarray, impostor: np.ndarray, bins: int = 30) -> float:
    edges = np.histogram_bin_edges(np.r_[target, impostor], bins=bins)
    a, _ = np.histogram(target, edges, density=True)
    b, _ = np.histogram(impostor, edges, density=True)
    return float(np.sum(np.minimum(a, b) * np.diff(edges)))


def evaluate_candidate(signals, labels, frequencies, alpha, threshold, harmonics):
    parameters = ResonateAndFireParameters(
        damping_alpha=float(alpha), threshold=float(threshold),
        integration_substeps=4, refractory_cycles=0.5, solver="exact",
        reset_mode="zero", spike_detection="upward_crossing",
    )
    model = OscillatorBankClassifier(
        frequencies, 1000.0, parameters, spread_hz=(0.0,),
        harmonics=harmonics, harmonic_weights=tuple(1 / h for h in harmonics),
    ).fit_scaler(signals)
    model.fit_calibration(signals, labels, STOPS)
    scores = model.decision_scores(signals, STOPS)
    predictions = np.argmax(scores, axis=-1)
    accuracy = np.mean(predictions == labels[None, :], axis=1)
    return accuracy, predictions, scores


def run_cell(subject_id: int, class_count: int, subject_data: np.ndarray) -> Path:
    checkpoint = CHECKPOINTS / f"subject_{subject_id:02d}_{class_count:02d}_classes.npz"
    if checkpoint.exists():
        print(f"S{subject_id:02d} {class_count:02d} classes: resumed", flush=True)
        return checkpoint
    frequencies = CLASS_SETS[np.flatnonzero(CLASS_COUNTS == class_count)[0]]
    values = subject_data[frequencies - 8]
    signals, labels = flatten(values)
    candidate_accuracy = np.zeros((len(CANDIDATES), len(DURATIONS)), dtype=np.float32)
    best_predictions = None
    best_scores = None
    best_index = -1
    best_value = -np.inf
    started = time.perf_counter()
    for index, (alpha, threshold, harmonic_index) in enumerate(CANDIDATES):
        accuracy, predictions, scores = evaluate_candidate(
            signals, labels, frequencies, alpha, threshold, HARMONIC_SETS[harmonic_index]
        )
        candidate_accuracy[index] = accuracy
        # Optimize the full duration-accuracy curve, with 5 s breaking ties.
        objective = float(accuracy.mean() + 1e-3 * accuracy[-1])
        if objective > best_value:
            best_value = objective
            best_index = index
            best_predictions = predictions.copy()
            best_scores = scores.copy()
        print(
            f"S{subject_id:02d} C{class_count:02d} candidate {index+1:02d}/{len(CANDIDATES)} "
            f"alpha={alpha:g} threshold={threshold:g} H={HARMONIC_SETS[harmonic_index]} "
            f"5s={100*accuracy[-1]:.2f}%",
            flush=True,
        )
    target = np.take_along_axis(best_scores[-1], labels[:, None], axis=1)[:, 0]
    masked = best_scores[-1].copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    impostor = masked.max(axis=1)
    alpha, threshold, harmonic_index = CANDIDATES[best_index]
    np.savez_compressed(
        checkpoint,
        subject_id=subject_id, class_count=class_count, frequencies_hz=frequencies,
        durations_seconds=DURATIONS, candidate_accuracy=candidate_accuracy,
        candidates=np.asarray(CANDIDATES, dtype=float), harmonic_sets=np.array(("f", "f+2f", "f+2f+3f")),
        selected_candidate_index=best_index, selected_damping_alpha=alpha,
        selected_threshold=threshold, selected_harmonic_index=harmonic_index,
        selected_harmonics=np.asarray(HARMONIC_SETS[harmonic_index]),
        accuracy=candidate_accuracy[best_index], predictions=best_predictions,
        target_scores_5s=target, best_impostor_scores_5s=impostor,
        target_margin_5s=target-impostor,
        overlap_coefficient_5s=overlap_coefficient(target, impostor),
        evaluation_design="same_subject_same_data_optimization_and_accuracy_no_holdout",
        elapsed_seconds=time.perf_counter()-started,
    )
    print(f"S{subject_id:02d} {class_count:02d} classes complete: {100*candidate_accuracy[best_index,-1]:.2f}%", flush=True)
    return checkpoint


def aggregate() -> Path:
    shape = (len(SUBJECT_IDS), len(CLASS_COUNTS), len(DURATIONS))
    accuracy = np.full(shape, np.nan)
    overlap = np.full(shape[:2], np.nan)
    margins = np.full(shape[:2], np.nan)
    selected = np.full(shape[:2] + (3,), np.nan)
    harmonic_index = np.full(shape[:2], -1, dtype=int)
    for si, subject_id in enumerate(SUBJECT_IDS):
        for ci, class_count in enumerate(CLASS_COUNTS):
            path = CHECKPOINTS / f"subject_{subject_id:02d}_{class_count:02d}_classes.npz"
            if not path.exists():
                continue
            with np.load(path) as result:
                accuracy[si, ci] = result["accuracy"]
                overlap[si, ci] = result["overlap_coefficient_5s"]
                margins[si, ci] = np.mean(result["target_margin_5s"] > 0)
                harmonic_index[si, ci] = result["selected_harmonic_index"]
                selected[si, ci] = (
                    result["selected_damping_alpha"], result["selected_threshold"],
                    result["selected_harmonic_index"],
                )
    output = OUT / "all_30_subjects_apparent_accuracy.npz"
    np.savez_compressed(
        output, subject_ids=SUBJECT_IDS, class_counts=CLASS_COUNTS,
        durations_seconds=DURATIONS, accuracy=accuracy,
        overlap_coefficient_5s=overlap, positive_margin_fraction_5s=margins,
        selected_parameters=selected, selected_harmonic_index=harmonic_index,
        harmonic_sets=np.array(("f", "f+2f", "f+2f+3f")),
        evaluation_design="same_subject_same_data_optimization_and_accuracy_no_holdout",
    )
    return output


def main() -> None:
    for subject_id in SUBJECT_IDS:
        print(f"Loading subject {subject_id}/30", flush=True)
        subject_data = load_subject(int(subject_id))
        for class_count in CLASS_COUNTS:
            run_cell(int(subject_id), int(class_count), subject_data)
        aggregate()
    print(aggregate(), flush=True)


if __name__ == "__main__":
    main()
