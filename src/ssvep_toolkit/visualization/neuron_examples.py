"""Real-segment signal, internal-state, and spike evidence for every encoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence


ELECTRODE_DEFINITION = {
    "O1": (60,), "Oz": (61,), "O2": (62,),
    "O1-Oz": (60, 61), "O2-Oz": (62, 61),
}


@dataclass(frozen=True)
class NeuronExampleConfig:
    subject: int = 1
    condition: int = 2
    frequency_hz: int = 8
    block: int = 1
    electrode: str = "Oz"
    start_ms: int = 140
    duration_ms: int = 1000
    sampling_rate_hz: float = 1000.0
    filter_order: int = 5
    filter_half_width_hz: float = 1.0
    rf_alpha: float = .025
    rf_threshold: float = .01
    rf_operating_rms: float = .75
    rf_input_gain: float = .05
    rf_normalize_input_by_resonance: bool = False
    delta_threshold_scale: float = 1.0
    delta_asymmetry: float = 1.0
    lif_threshold_scale: float = .5
    lif_tau_seconds: float = .02
    lif_input_gain: float = 1.0

    def validate(self) -> None:
        if not 1 <= self.subject <= 30 or not 1 <= self.frequency_hz <= 60:
            raise ValueError("subject must be 1–30 and frequency 1–60 Hz")
        if self.condition not in (1, 2) or not 1 <= self.block <= 12:
            raise ValueError("condition must be 1–2 and block 1–12")
        if self.electrode not in ELECTRODE_DEFINITION:
            raise ValueError(f"electrode must be one of {tuple(ELECTRODE_DEFINITION)}")
        if self.start_ms < 0 or self.duration_ms <= 0 or self.start_ms + self.duration_ms > 5140:
            raise ValueError("requested segment is outside the 5.14 s trial")


def compute_neuron_example(data_directory: str | Path, config: NeuronExampleConfig) -> dict[str, object]:
    import numpy as np

    from ssvep_toolkit.algorithms import (
        DeltaEncoderParameters, LIFEncoderParameters, delta_state_trace, lif_state_trace,
    )
    from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters, simulate_trace
    from ssvep_toolkit.data.matlab import Matlab73Dataset
    from ssvep_toolkit.preprocessing import butterworth_bandpass

    config.validate()
    path = Path(data_directory) / f"data_s{config.subject}_64.mat"
    with Matlab73Dataset(path) as source:
        trial = source.read_trial(config.condition, config.frequency_hz, config.block)
    definition = ELECTRODE_DEFINITION[config.electrode]
    raw_full = np.asarray(trial[definition[0]], dtype=float)
    if len(definition) == 2:
        raw_full = raw_full - np.asarray(trial[definition[1]], dtype=float)
    start = round(config.start_ms * config.sampling_rate_hz / 1000)
    stop = start + round(config.duration_ms * config.sampling_rate_hz / 1000)
    raw = raw_full[start:stop]
    filtered_full = butterworth_bandpass(
        raw_full, config.sampling_rate_hz,
        max(.1, config.frequency_hz - config.filter_half_width_hz),
        config.frequency_hz + config.filter_half_width_hz,
        order=config.filter_order, zero_phase=True,
    )
    filtered = filtered_full[start:stop]
    centered = raw - raw.mean()
    raw_rms = float(np.sqrt(np.mean(centered ** 2)))
    gain = config.rf_operating_rms / max(raw_rms, 1e-9)
    rf_input = centered * gain
    rf_parameters = ResonateAndFireParameters(
        damping_alpha=config.rf_alpha, threshold=config.rf_threshold,
        input_gain=config.rf_input_gain, integration_substeps=4,
        normalize_input_by_resonance=config.rf_normalize_input_by_resonance,
        refractory_cycles=.5, solver="exact", reset_mode="zero",
        spike_detection="upward_crossing",
    )
    rf_spikes, rf_u, rf_v = simulate_trace(
        rf_input, config.frequency_hz, config.sampling_rate_hz, rf_parameters,
    )
    # A robust high quantile gives a readable event raster across low and high
    # stimulus frequencies without normalizing the displayed EEG amplitude.
    derivative_scale = float(np.quantile(np.abs(np.diff(filtered)), .9))
    delta_threshold = max(derivative_scale * config.delta_threshold_scale, 1e-9)
    delta = delta_state_trace(filtered, DeltaEncoderParameters(delta_threshold, config.delta_asymmetry))
    band_rms = float(np.sqrt(np.mean(filtered ** 2)))
    lif_threshold = max(band_rms * config.lif_threshold_scale, 1e-9)
    lif = lif_state_trace(
        filtered, config.sampling_rate_hz,
        LIFEncoderParameters(lif_threshold, config.lif_tau_seconds, config.lif_input_gain),
    )
    return {
        "config": config, "time_ms": np.arange(raw.size) / config.sampling_rate_hz * 1000,
        "raw_uV": raw, "filtered_uV": filtered,
        "rf_input": rf_input, "rf_u": rf_u, "rf_v": rf_v, "rf_spikes": rf_spikes,
        "rf_threshold": config.rf_threshold, "rf_gain_per_uV": gain, "raw_rms_uV": raw_rms,
        "delta_change_uV": delta["change"], "delta_up": np.flatnonzero(delta["up_spikes"]),
        "delta_down": np.flatnonzero(delta["down_spikes"]),
        "delta_up_threshold_uV": delta["up_threshold"],
        "delta_down_threshold_uV": delta["down_threshold"],
        "lif_membrane": lif["membrane"], "lif_spikes": np.flatnonzero(lif["spikes"]),
        "lif_threshold": lif["threshold"], "band_rms_uV": band_rms,
    }


def render_neuron_example(
    data_directory: str | Path,
    config: NeuronExampleConfig,
    output: str | Path,
    *,
    encoders: Sequence[str] = ("resonate_fire", "delta", "lif"),
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    valid = {"resonate_fire", "delta", "lif"}
    if not encoders or any(item not in valid for item in encoders):
        raise ValueError(f"encoders must be drawn from {valid}")
    data = compute_neuron_example(data_directory, config)
    time = data["time_ms"]
    fig, axes = plt.subplots(3, len(encoders), figsize=(5.3 * len(encoders), 8.5),
                             sharex=True, squeeze=False, constrained_layout=True)
    for column, encoder in enumerate(encoders):
        if encoder == "resonate_fire":
            axes[0, column].plot(time, data["raw_uV"], color="#355f8d", linewidth=.8)
            axes[0, column].set_title(f"R&F · {len(data['rf_spikes'])} spikes")
            axes[0, column].set_ylabel("Raw EEG (µV)")
            axes[1, column].plot(time, data["rf_u"], label="u", linewidth=.9)
            axes[1, column].plot(time, data["rf_v"], label="v", linewidth=.9)
            axes[1, column].axhline(data["rf_threshold"], color="#bf4b4b", linewidth=.8, label="threshold")
            axes[1, column].set_ylabel("Oscillator state\n(dimensionless)"); axes[1, column].legend(frameon=False, ncol=3)
            _raster(axes[2, column], time, data["rf_spikes"], ("R&F",), (0,))
            axes[2, column].set_title(f"gain={data['rf_gain_per_uV']:.3g}/µV, α={config.rf_alpha:g}")
        elif encoder == "delta":
            axes[0, column].plot(time, data["filtered_uV"], color="#355f8d", linewidth=.8)
            axes[0, column].set_title(f"Delta · {len(data['delta_up'])} UP, {len(data['delta_down'])} DN")
            axes[0, column].set_ylabel("Filtered EEG (µV)")
            axes[1, column].plot(time, data["delta_change_uV"], linewidth=.8)
            axes[1, column].axhline(data["delta_up_threshold_uV"], color="#16856b", linewidth=.8)
            axes[1, column].axhline(data["delta_down_threshold_uV"], color="#bf4b4b", linewidth=.8)
            axes[1, column].set_ylabel("Successive Δ (µV)")
            _raster(axes[2, column], time, np.r_[data["delta_up"], data["delta_down"]],
                    ("UP", "DN"), (len(data["delta_up"]), len(data["delta_down"])))
            axes[2, column].set_title(f"threshold={data['delta_up_threshold_uV']:.3g} µV")
        else:
            axes[0, column].plot(time, data["filtered_uV"], color="#355f8d", linewidth=.8)
            axes[0, column].set_title(f"LIF · {len(data['lif_spikes'])} spikes")
            axes[0, column].set_ylabel("Filtered EEG (µV)")
            axes[1, column].plot(time, data["lif_membrane"], linewidth=.9)
            axes[1, column].axhline(data["lif_threshold"], color="#bf4b4b", linewidth=.8)
            axes[1, column].set_ylabel("Membrane state\n(µV-equivalent)")
            _raster(axes[2, column], time, data["lif_spikes"], ("LIF",), (0,))
            axes[2, column].set_title(f"threshold={data['lif_threshold']:.3g}, τ={1000*config.lif_tau_seconds:g} ms")
        axes[2, column].set_xlabel("Time after segment start (ms)")
        for row in range(3):
            axes[row, column].grid(alpha=.18)
    fig.suptitle(
        f"S{config.subject:02d} · {config.frequency_hz} Hz class · block {config.block} · "
        f"{config.electrode} · {config.duration_ms} ms",
        fontsize=15,
    )
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight"); plt.close(fig)
    metadata = target.with_suffix(".json")
    metadata.write_text(json.dumps({
        **asdict(config), "image": target.name, "encoders": list(encoders),
        "image_dpi": 200, "time_unit": "milliseconds", "eeg_amplitude_unit": "microvolts",
        "rf_state_unit": "dimensionless", "lif_state_unit": "microvolt_equivalent",
    }, indent=2), encoding="utf-8")
    np.savez_compressed(
        target.with_suffix(".npz"), time_ms=time, raw_uV=data["raw_uV"],
        filtered_uV=data["filtered_uV"], rf_u=data["rf_u"], rf_v=data["rf_v"],
        rf_spikes=data["rf_spikes"], delta_change_uV=data["delta_change_uV"],
        delta_up=data["delta_up"], delta_down=data["delta_down"],
        lif_membrane=data["lif_membrane"], lif_spikes=data["lif_spikes"],
    )
    return target


def _raster(axis, time, indices, labels, counts):
    import numpy as np

    indices = np.asarray(indices, dtype=int)
    if len(labels) == 1:
        if indices.size:
            axis.vlines(time[indices], .1, .9, color="#482878", linewidth=.8)
        axis.set(ylim=(0, 1), yticks=(.5,), yticklabels=labels, ylabel="Output spikes")
        return
    split = int(counts[0]); groups = (indices[:split], indices[split:])
    for row, (group, color) in enumerate(zip(groups, ("#16856b", "#bf4b4b"))):
        if group.size:
            axis.vlines(time[group], row+.1, row+.9, color=color, linewidth=.8)
    axis.set(ylim=(0, 2), yticks=(.5, 1.5), yticklabels=labels, ylabel="Output spikes")


def plot_rf_input_compensation_ablation(labels, compensated_counts, raw_counts, output: str | Path) -> Path:
    """Compare raw R&F counts with and without input-frequency compensation."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = tuple(labels); compensated = np.asarray(compensated_counts); raw = np.asarray(raw_counts)
    if compensated.shape != raw.shape or compensated.ndim != 1 or len(labels) != raw.size:
        raise ValueError("labels and count vectors must be aligned")
    x = np.arange(raw.size); width = .36
    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    first = axis.bar(x-width/2, compensated, width, label="Input drive ÷ resonance frequency", color="#8da0b6")
    second = axis.bar(x+width/2, raw, width, label="Uncompensated input drive", color="#482878")
    axis.bar_label(first, padding=3); axis.bar_label(second, padding=3)
    axis.set(xticks=x, xticklabels=labels, ylabel="Raw R&F spikes in 1 s",
             title="Input-frequency compensation ablation on real EEG segments")
    axis.legend(frameon=False); axis.grid(axis="y", alpha=.2)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight"); plt.close(fig); return target
