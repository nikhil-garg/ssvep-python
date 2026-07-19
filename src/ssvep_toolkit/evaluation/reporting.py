"""Comparable accuracy, ITR, latency, robustness, and spike-cost reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ssvep_toolkit.algorithms.itr import latency_itr_report


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    correct: int
    trials: int
    classes: int
    decision_seconds: float
    onset_latency_seconds: float
    practical_overhead_seconds: float
    neural_window_itr_bits_per_minute: float
    practical_itr_bits_per_minute: float
    mean_spikes_per_trial: float | None
    spikes_per_correct_selection: float | None
    robustness_accuracy_drop: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluation_report(
    labels: Any,
    predictions: Any,
    *,
    classes: int,
    decision_seconds: float,
    onset_latency_seconds: float = .14,
    practical_overhead_seconds: float = 1.0,
    spike_counts: Any | None = None,
    perturbed_predictions: Any | None = None,
) -> EvaluationReport:
    import numpy as np

    truth = np.asarray(labels)
    predicted = np.asarray(predictions)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError("labels and predictions must be aligned non-empty vectors")
    accuracy = float(np.mean(predicted == truth))
    itr = latency_itr_report(
        classes, accuracy, decision_seconds, onset_latency_seconds=onset_latency_seconds,
        practical_overhead_seconds=practical_overhead_seconds,
    )
    mean_spikes = spike_efficiency = None
    if spike_counts is not None:
        counts = np.asarray(spike_counts, dtype=float)
        if counts.shape[0] != truth.size:
            raise ValueError("spike_counts must have one row per trial")
        per_trial = counts.reshape(truth.size, -1).sum(axis=1)
        mean_spikes = float(per_trial.mean())
        correct = int(np.sum(predicted == truth))
        spike_efficiency = float(per_trial.sum() / correct) if correct else None
    robustness_drop = None
    if perturbed_predictions is not None:
        perturbed = np.asarray(perturbed_predictions)
        if perturbed.shape != truth.shape:
            raise ValueError("perturbed_predictions must match labels")
        robustness_drop = accuracy - float(np.mean(perturbed == truth))
    return EvaluationReport(
        accuracy=accuracy, correct=int(np.sum(predicted == truth)), trials=int(truth.size),
        classes=int(classes), decision_seconds=float(decision_seconds),
        onset_latency_seconds=float(onset_latency_seconds),
        practical_overhead_seconds=float(practical_overhead_seconds),
        neural_window_itr_bits_per_minute=float(itr["neural_window_itr_bits_per_minute"]),
        practical_itr_bits_per_minute=float(itr["practical_itr_bits_per_minute"]),
        mean_spikes_per_trial=mean_spikes, spikes_per_correct_selection=spike_efficiency,
        robustness_accuracy_drop=robustness_drop,
    )


def earliest_near_optimal_endpoint(
    accuracies: Sequence[float], durations_seconds: Sequence[float], *, fraction: float = .95,
) -> tuple[float, float]:
    """Return earliest duration reaching `fraction` of maximum observed accuracy."""
    import numpy as np

    accuracy = np.asarray(accuracies, dtype=float)
    durations = np.asarray(durations_seconds, dtype=float)
    if accuracy.ndim != 1 or durations.shape != accuracy.shape or accuracy.size == 0:
        raise ValueError("accuracies and durations must be aligned vectors")
    if not 0 < fraction <= 1 or np.any(durations <= 0):
        raise ValueError("fraction must be in (0,1] and durations must be positive")
    valid = np.isfinite(accuracy)
    if not np.any(valid):
        raise ValueError("at least one accuracy must be finite")
    target = fraction * float(np.max(accuracy[valid]))
    eligible = np.flatnonzero(valid & (accuracy >= target))
    index = int(eligible[np.argmin(durations[eligible])])
    return float(durations[index]), float(accuracy[index])
