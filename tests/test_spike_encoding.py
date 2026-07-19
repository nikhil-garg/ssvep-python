import numpy as np
import pytest

from ssvep_toolkit.algorithms import (
    DeltaEncoderParameters,
    EncoderConfig,
    LIFEncoderParameters,
    delta_encode,
    encode_target_frequency_bank,
    lif_encode,
    encode_spike_features,
    template_classification_scores,
)
from ssvep_toolkit.preprocessing import BandpassParameters, target_frequency_filter_bank


def test_delta_encoder_produces_separate_up_and_down_streams() -> None:
    signal = np.array([0.0, 0.6, 0.8, -0.1, -0.7])
    spikes = delta_encode(signal, DeltaEncoderParameters(threshold=0.5))
    assert spikes.shape == (2, 5)
    np.testing.assert_array_equal(spikes[0], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(spikes[1], [0, 0, 0, 1, 1])


def test_delta_asymmetry_scales_only_down_threshold() -> None:
    signal = np.array([0.0, 0.6, 0.0])
    spikes = delta_encode(signal, DeltaEncoderParameters(threshold=0.5, asymmetry=2.0))
    np.testing.assert_array_equal(spikes[0], [0, 1, 0])
    assert not spikes[1].any()


def test_lif_encoder_leaks_integrates_spikes_and_resets() -> None:
    signal = np.ones(1000)
    slow = lif_encode(signal, 1000, LIFEncoderParameters(threshold=0.5, tau_seconds=0.02))
    fast = lif_encode(signal, 1000, LIFEncoderParameters(threshold=0.5, tau_seconds=0.005))
    assert slow.shape == (1, 1000)
    assert slow.sum() > 0
    assert fast.sum() > slow.sum()
    assert not lif_encode(np.full(1000, 0.2), 1000, LIFEncoderParameters(0.5)).any()


def test_target_filter_bank_separates_sinusoidal_frequencies() -> None:
    sampling_rate = 1000
    time = np.arange(4000) / sampling_rate
    signal = np.sin(2 * np.pi * 10 * time) + 0.2 * np.sin(2 * np.pi * 20 * time)
    bands = target_frequency_filter_bank(
        signal, (10, 20), sampling_rate, BandpassParameters(enabled=True, order=5),
    )
    assert bands.shape == (2, signal.size)
    assert np.sqrt(np.mean(bands[0] ** 2)) > 3 * np.sqrt(np.mean(bands[1] ** 2))


@pytest.mark.parametrize("encoder,streams", [("delta", 2), ("lif", 1)])
def test_each_target_band_has_its_own_encoder(encoder: str, streams: int) -> None:
    sampling_rate = 1000
    time = np.arange(2000) / sampling_rate
    signal = np.stack((np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 12 * time)))
    spikes = encode_target_frequency_bank(
        signal, (10, 12), sampling_rate, encoder=encoder,
        delta_parameters=DeltaEncoderParameters(0.01),
        lif_parameters=LIFEncoderParameters(0.05, input_gain=10),
        bandpass=BandpassParameters(enabled=True),
    )
    assert spikes.shape == (2, 2, streams, signal.shape[-1])
    assert spikes.dtype == np.uint8


def test_one_hz_target_uses_positive_lower_cutoff() -> None:
    signal = np.zeros(4000)
    result = target_frequency_filter_bank(signal, (1,), 1000, BandpassParameters(enabled=True))
    assert result.shape == (1, signal.size)


def test_uniform_encoder_interface_returns_trial_target_counts() -> None:
    sampling_rate = 1000
    time = np.arange(2000) / sampling_rate
    signals = np.stack((
        np.sin(2 * np.pi * 10 * time),
        np.sin(2 * np.pi * 12 * time),
    ))[:, None, :]
    result = encode_spike_features(
        signals, (10, 12), sampling_rate,
        EncoderConfig(kind="delta", delta=DeltaEncoderParameters(0.01)),
    )
    assert result.counts.shape == (2, 4)
    assert result.stream_names == ("UP", "DN")
    scores = template_classification_scores(result.counts, np.array((0, 1)))
    assert scores.shape == (2, 2)
