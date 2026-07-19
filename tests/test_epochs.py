import numpy as np

from ssvep_toolkit.preprocessing.epochs import phase_shifted_epochs


def test_phase_shifted_epoch_shape_and_first_epoch() -> None:
    data = np.arange(1285)[None, :]
    epochs = phase_shifted_epochs(data, 8)
    assert epochs.shape == (4, 1, 250)
    np.testing.assert_array_equal(epochs[0, 0], np.arange(250))

