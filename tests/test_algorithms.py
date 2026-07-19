import numpy as np

from ssvep_toolkit.algorithms import (
    canonical_correlations,
    bits_per_selection,
    fit_trca,
    information_transfer_rate,
    latency_aware_itr,
    latency_itr_report,
    predict_trca,
    reference_signals,
    trca_intertrial_covariance,
)


def test_itr_boundaries_match_formula() -> None:
    assert information_transfer_rate(4, 1.0, 30) == 60
    assert np.isfinite(information_transfer_rate(4, 0.0, 30))


def test_latency_aware_itr_distinguishes_neural_and_practical_timing() -> None:
    ideal = latency_aware_itr(4, 1.0, 0.5, onset_latency_seconds=0.14)
    practical = latency_aware_itr(
        4, 1.0, 0.5, onset_latency_seconds=0.14, gaze_shift_seconds=1.0,
    )
    assert ideal > practical > 0
    assert bits_per_selection(4, 0.25, clamp_below_chance=True) == 0
    report = latency_itr_report(4, np.array((0.8, 0.9)), np.array((0.5, 1.0)))
    assert report["neural_window_itr_bits_per_minute"].shape == (2,)


def test_reference_signal_shape_and_timing() -> None:
    refs = reference_signals([8, 12], 250, 250, harmonics=3)
    assert refs.shape == (2, 6, 250)
    assert np.isclose(refs[0, 0, 0], np.sin(2 * np.pi * 8 / 250))


def test_canonical_correlation_is_one_for_linear_copy() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 500))
    y = np.vstack((2 * x[0], -3 * x[1]))
    assert canonical_correlations(x, y)[0] > 0.999999


def test_trca_covariance_includes_same_trials() -> None:
    trials = np.ones((2, 3, 4))
    with_same = trca_intertrial_covariance(trials, True)
    without_same = trca_intertrial_covariance(trials, False)
    assert np.all(with_same == 2 * without_same)


def test_trca_separates_synthetic_classes() -> None:
    rng = np.random.default_rng(3)
    time = np.arange(250) / 250
    trials = np.empty((2, 5, 3, 250))
    for cls, frequency in enumerate((8, 15)):
        base = np.sin(2 * np.pi * frequency * time)
        for trial in range(5):
            trials[cls, trial] = np.vstack((base, 0.7 * base, -0.3 * base)) + rng.normal(0, 0.02, (3, 250))
    model = fit_trca(trials)
    assert predict_trca(model, trials[0, 0]) == 0
    assert predict_trca(model, trials[1, 0]) == 1
