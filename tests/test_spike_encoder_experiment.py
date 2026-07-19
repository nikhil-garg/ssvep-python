import numpy as np

from ssvep_toolkit.evaluation import apparent_template_result, delta_count_features, lif_count_features


def test_count_feature_shapes_and_template_scoring() -> None:
    sampling_rate = 1000
    time = np.arange(1000) / sampling_rate
    bank = np.empty((2, 4, 1, time.size))
    bank[0] = np.sin(2 * np.pi * 10 * time)
    bank[1] = np.sin(2 * np.pi * 12 * time)
    delta = delta_count_features(bank, 0.01)
    lif = lif_count_features(bank, sampling_rate, 0.05, 0.01, input_gain=10)
    assert delta.shape == (4, 4)
    assert lif.shape == (4, 2)
    accuracy, prediction, scores = apparent_template_result(delta, np.array((0, 0, 1, 1)))
    assert 0 <= accuracy <= 1
    assert prediction.shape == (4,)
    assert scores.shape == (4, 2)


def test_count_features_can_retain_channel_and_stream_identity() -> None:
    bank = np.zeros((3, 6, 2, 20))
    bank[..., 1::2] = 1
    delta = delta_count_features(bank, .5, preserve_channels=True)
    lif = lif_count_features(bank, 1000, .1, .01, input_gain=10, preserve_channels=True)
    assert delta.shape == (6, 3 * 2 * 2)
    assert lif.shape == (6, 3 * 2)
