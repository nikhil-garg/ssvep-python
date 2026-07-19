from pathlib import Path

import h5py
import numpy as np

from ssvep_toolkit.data.matlab import Matlab73Dataset


def test_reader_converts_storage_axes(tmp_path: Path) -> None:
    path = tmp_path / "subject.mat"
    logical = np.arange(2 * 3 * 5 * 4 * 2).reshape(2, 3, 5, 4, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("datas", data=logical.transpose(4, 3, 2, 1, 0))
    with Matlab73Dataset(path) as reader:
        assert reader.logical_shape == logical.shape
        np.testing.assert_array_equal(reader.read_trial(2, 3, 1), logical[1, :, :, 2, 0])

