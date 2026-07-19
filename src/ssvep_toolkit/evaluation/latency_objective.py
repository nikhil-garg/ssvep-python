"""Latency, spike-cost, and Pareto summaries for endpoint experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class EndpointResult:
    seconds: float
    accuracy: float
    itr_bits_per_minute: float
    mean_spikes: float
    utility: float


def endpoint_results(
    labels: Any, predictions: Any, endpoints_seconds: Sequence[float], spike_counts: Any, *,
    classes: int, overhead_seconds: float = 0.0, spike_penalty: float = 0.0,
    latency_penalty: float = 0.0,
) -> tuple[EndpointResult, ...]:
    """Calculate transparent endpoint utilities; predictions are endpoint x trial."""
    import numpy as np
    from ssvep_toolkit.algorithms.itr import information_transfer_rate

    truth = np.asarray(labels); predicted = np.asarray(predictions); spikes = np.asarray(spike_counts, dtype=float)
    endpoints = np.asarray(endpoints_seconds, dtype=float)
    if predicted.shape != (endpoints.size, truth.size) or spikes.shape != predicted.shape:
        raise ValueError("predictions and spike_counts must be endpoint x trial")
    if np.any(endpoints <= 0):
        raise ValueError("endpoints must be positive")
    output = []
    for index, seconds in enumerate(endpoints):
        accuracy = float(np.mean(predicted[index] == truth)); mean_spikes = float(np.mean(spikes[index]))
        itr = float(information_transfer_rate(classes, accuracy, seconds + overhead_seconds))
        utility = accuracy - spike_penalty * mean_spikes - latency_penalty * float(seconds)
        output.append(EndpointResult(float(seconds), accuracy, itr, mean_spikes, utility))
    return tuple(output)


def pareto_endpoints(results: Sequence[EndpointResult]) -> tuple[EndpointResult, ...]:
    """Return endpoints not dominated in accuracy, latency, and spike cost."""
    frontier = []
    for candidate in results:
        dominated = any(
            other.accuracy >= candidate.accuracy and other.seconds <= candidate.seconds
            and other.mean_spikes <= candidate.mean_spikes
            and (other.accuracy > candidate.accuracy or other.seconds < candidate.seconds
                 or other.mean_spikes < candidate.mean_spikes)
            for other in results
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: item.seconds))
