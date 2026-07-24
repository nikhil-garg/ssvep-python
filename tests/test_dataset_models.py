import numpy as np
import pytest

from ssvep_toolkit.datasets import EpochBatch


def test_epoch_batch_requires_documented_axes() -> None:
    batch = EpochBatch(
        np.zeros((4, 3, 20)), np.array((0, 1, 0, 1)), np.array((0, 0, 1, 1)),
        1000.0, ("O1", "Oz", "O2"), (8.0, 12.0), 4,
    )
    assert batch.data.shape == (4, 3, 20)
    with pytest.raises(ValueError, match="shape"):
        EpochBatch(np.zeros((4, 20)), np.zeros(4, int), np.zeros(4, int), 1000.0, ("O1",), (8.0,), 0)
