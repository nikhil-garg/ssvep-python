from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence


def compute_figure8_cohort(
    inputs: Sequence[str | Path], output: str | Path, progress: Callable[[str], None] = print
) -> Path:
    """Reproduce `savefor_Figure8.m` arrays for preprocessed subject files."""
    import h5py
    import numpy as np
    from scipy.signal import cheb1ord, cheby1, filtfilt

    nyquist = 125.0
    order, wn = cheb1ord([0.8 / nyquist, 90 / nyquist], [0.5 / nyquist, 100 / nyquist], 3, 40)
    b, a = cheby1(order, 0.5, wn, btype="bandpass")
    amplitude_harmonics = []
    snr_harmonics = []
    amplitude_maps = []
    snr_maps = []
    for index, item in enumerate(inputs, 1):
        path = Path(item)
        progress(f"Figure 8 subject {index}/{len(inputs)}: {path.name}")
        with h5py.File(path, "r") as source:
            data = np.asarray(source["data"], dtype=float)
        filtered = filtfilt(b, a, data, axis=2)
        averaged = filtered.mean(axis=4)
        transformed = np.fft.rfft(averaged, axis=2)
        spectrum = np.abs(transformed / averaged.shape[2])
        spectrum[:, :, 1:-1, :] *= 2
        spectrum = spectrum.mean(axis=1).transpose(0, 2, 1)  # condition, stimulus, bin
        snr = np.full_like(spectrum, np.nan)
        for spectral_bin in range(4, spectrum.shape[-1] - 4):
            neighbors = [spectral_bin + delta for delta in (-4, -3, -2, -1, 1, 2, 3, 4)]
            noise = spectrum[..., neighbors].mean(axis=-1)
            snr[..., spectral_bin] = 20 * np.log10(spectrum[..., spectral_bin] / noise)
        amp_h = np.zeros((2, 60, 10))
        snr_h = np.zeros_like(amp_h)
        amp_map = np.zeros((2, 60, 90))
        snr_map = np.zeros_like(amp_map)
        for stimulus in range(1, 61):
            for harmonic in range(1, 11):
                response = stimulus * harmonic
                if response <= 90:
                    bin_index = response * 5
                    amp_h[:, stimulus - 1, harmonic - 1] = spectrum[:, stimulus - 1, bin_index]
                    snr_h[:, stimulus - 1, harmonic - 1] = snr[:, stimulus - 1, bin_index]
                    amp_map[:, stimulus - 1, response - 1] = spectrum[:, stimulus - 1, bin_index]
                    snr_map[:, stimulus - 1, response - 1] = snr[:, stimulus - 1, bin_index]
        amplitude_harmonics.append(amp_h)
        snr_harmonics.append(snr_h)
        amplitude_maps.append(amp_map)
        snr_maps.append(snr_map)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frequencies_hz=np.arange(1, 61),
        amplitude_harmonics=np.asarray(amplitude_harmonics),
        snr_harmonics=np.asarray(snr_harmonics),
        amplitude_maps=np.asarray(amplitude_maps),
        snr_maps=np.asarray(snr_maps),
    )
    return output

