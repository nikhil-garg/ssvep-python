import numpy as np

from ssvep_toolkit.preprocessing.downsampling import downsample, output_sample_count


def test_matlab_compatible_selects_every_fourth_sample() -> None:
    data = np.arange(20).reshape(2, 10)
    result = downsample(data, 1000, 250, "matlab_compatible")
    np.testing.assert_array_equal(result, data[:, [0, 4, 8]])


def test_output_count_matches_slice() -> None:
    assert output_sample_count(5140, 1000, 250, "matlab_compatible") == 1285

