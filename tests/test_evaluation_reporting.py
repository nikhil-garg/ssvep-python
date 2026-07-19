import pytest

from ssvep_toolkit.evaluation import earliest_near_optimal_endpoint, evaluation_report


def test_evaluation_report_includes_latency_spike_cost_and_robustness():
    report = evaluation_report(
        [0, 1, 2, 0], [0, 1, 1, 0], classes=3, decision_seconds=.5,
        spike_counts=[[2, 3], [1, 1], [4, 0], [2, 2]], perturbed_predictions=[0, 2, 1, 0],
    )
    assert report.accuracy == .75
    assert report.mean_spikes_per_trial == 3.75
    assert report.spikes_per_correct_selection == 5
    assert report.robustness_accuracy_drop == .25
    assert report.neural_window_itr_bits_per_minute > report.practical_itr_bits_per_minute


def test_earliest_endpoint_within_95_percent_of_peak():
    duration, accuracy = earliest_near_optimal_endpoint([.70, .82, .85, .84], [.25, .5, .75, 1.0])
    assert duration == .5
    assert accuracy == .82
