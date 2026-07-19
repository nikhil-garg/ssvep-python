import numpy as np

from ssvep_toolkit.cli import main


def test_encode_spikes_cli_writes_reproducible_metadata(tmp_path) -> None:
    sampling_rate = 1000
    time = np.arange(2000) / sampling_rate
    input_path = tmp_path / "eeg.npy"
    output_path = tmp_path / "delta.npz"
    np.save(input_path, np.sin(2 * np.pi * 10 * time))

    status = main([
        "encode-spikes", "delta", "--input", str(input_path), "--output", str(output_path),
        "--frequencies", "10", "12", "--sampling-rate", "1000", "--threshold", "0.01",
    ])

    assert status == 0
    with np.load(output_path) as result:
        assert result["spikes"].shape == (2, 1, 1, 2, time.size)
        assert result["counts"].shape == (1, 4)
        np.testing.assert_array_equal(result["stream_names"], ["UP", "DN"])
        assert bool(result["bandpass_enabled"])
        assert int(result["bandpass_order"]) == 5


def test_resonate_fire_is_available_through_uniform_cli(tmp_path) -> None:
    sampling_rate = 1000
    time = np.arange(1000) / sampling_rate
    input_path = tmp_path / "eeg.npy"
    output_path = tmp_path / "rf.npz"
    data = np.stack((np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 12 * time)))[:, None, :]
    np.save(input_path, data)

    status = main([
        "encode-spikes", "resonate_fire", "--input", str(input_path), "--output", str(output_path),
        "--frequencies", "10", "12", "--sampling-rate", "1000", "--threshold", "0.01",
    ])

    assert status == 0
    with np.load(output_path) as result:
        assert result["counts"].shape == (2, 2)
        assert not bool(result["bandpass_enabled"])
        assert not bool(result["normalize_input_by_resonance"])
        assert "spikes" not in result.files
