import numpy as np

from ssvep_toolkit.features import amplitude_spectrum, signal_to_noise_ratio


def test_amplitude_spectrum_finds_unit_sine() -> None:
    fs = 250
    time = np.arange(fs) / fs
    frequencies, amplitude = amplitude_spectrum(np.sin(2 * np.pi * 8 * time), fs)
    peak = int(np.argmax(amplitude))
    assert frequencies[peak] == 8
    assert np.isclose(amplitude[peak], 1.0)


def test_snr_has_peak_at_stimulus() -> None:
    fs = 250
    time = np.arange(fs) / fs
    frequencies, amplitude = amplitude_spectrum(np.sin(2 * np.pi * 8 * time), fs)
    snr = signal_to_noise_ratio(amplitude)
    assert frequencies[np.nanargmax(snr)] == 8
