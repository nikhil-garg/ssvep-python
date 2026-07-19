import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import bandwidth_hz, damping_from_bandwidth, quality_factor
from ssvep_toolkit.preprocessing import (apply_branch_gain, causal_running_gain,
                                         fit_training_branch_gain)


def test_training_gain_ignores_held_out_amplitude_and_preserves_trial_ratios() -> None:
    t = np.linspace(0, 1, 100, endpoint=False)
    base = np.sin(2 * np.pi * 10 * t)
    signals = np.stack([base, 2 * base, 100 * base])[:, None, :]
    calibration = fit_training_branch_gain(signals, np.array([True, True, False]), target_rms=1)
    adapted = apply_branch_gain(signals, calibration)
    # The extreme held-out trial cannot affect the fitted reference.
    assert calibration.training_trials == 2
    assert np.isclose(np.sqrt(np.mean(adapted[1, 0] ** 2)) / np.sqrt(np.mean(adapted[0, 0] ** 2)), 2)


def test_running_gain_is_strictly_causal() -> None:
    signal = np.ones((1, 1, 20)); altered = signal.copy(); altered[..., 10:] = 100
    _, first = causal_running_gain(signal, 100, tau_seconds=.2)
    _, second = causal_running_gain(altered, 100, tau_seconds=.2)
    assert np.allclose(first[..., :11], second[..., :11])
    assert np.all(np.isfinite(second))


def test_bandwidth_parameterization_round_trips() -> None:
    alpha = damping_from_bandwidth(20, .5)
    assert np.isclose(bandwidth_hz(20, alpha), .5)
    assert np.isclose(quality_factor(alpha), 40)
