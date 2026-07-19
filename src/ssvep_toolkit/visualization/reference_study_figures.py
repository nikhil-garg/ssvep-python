from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .plots import _finish, _pyplot, plot_accuracy_itr, plot_frequency_heatmap, plot_shaded_series, plot_spectrum


REFERENCE_STUDY_FIGURE_CONTRACTS = {
    4: "signal, sampling_rate_hz",
    5: "frequencies_hz, amplitude",
    6: "frequencies_hz, amplitude",
    7: "matrix; optional xlabels, ylabels",
    8: "frequencies_hz, amplitude, snr",
    9: "frequencies_hz, scores",
    10: "x, accuracy, itr",
    11: "scores, accuracy",
    12: "frequencies_hz, accuracy, itr",
    13: "scores, accuracy",
}


def _require(data: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"missing figure inputs {missing}; available: {sorted(data)}")


def _scatter_relationship(data: Mapping[str, Any], number: int, output: Path) -> Any:
    import numpy as np
    from scipy.stats import linregress

    _require(data, "scores", "accuracy")
    plt = _pyplot()
    x = np.asarray(data["scores"], dtype=float).reshape(-1)
    y = np.asarray(data["accuracy"], dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size != y.size or x.size < 2:
        raise ValueError("scores and accuracy need at least two paired values")
    fit = linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, alpha=0.7)
    ax.plot(line_x, fit.intercept + fit.slope * line_x, color="tab:red")
    ax.set(xlabel="Subjective score", ylabel="Accuracy", title=f"Figure {number}: score relationship")
    ax.text(0.03, 0.97, f"r = {fit.rvalue:.3f}, p = {fit.pvalue:.3g}", transform=ax.transAxes, va="top")
    ax.grid(alpha=0.2)
    return _finish(fig, output)


def render_reference_study_figure(number: int, data: Mapping[str, Any], output: str | Path) -> Any:
    """Render reference-study Figure 4–13 without inventing missing results."""
    import numpy as np

    if number not in REFERENCE_STUDY_FIGURE_CONTRACTS:
        raise ValueError("figure number must be from 4 through 13")
    output = Path(output)
    if number == 4:
        _require(data, "signal", "sampling_rate_hz")
        signal = np.asarray(data["signal"], dtype=float).reshape(-1)
        fs = float(np.asarray(data["sampling_rate_hz"]).reshape(-1)[0])
        frequency = np.fft.rfftfreq(signal.size, 1 / fs)
        amplitude = np.abs(np.fft.rfft(signal)) * 2 / signal.size
        plt = _pyplot()
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        axes[0].plot(np.arange(signal.size) / fs, signal)
        axes[0].set(xlabel="Time (s)", ylabel="Signal", title="Figure 4: photodiode timing")
        axes[1].plot(frequency, amplitude)
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude", xlim=(0, min(100, fs / 2)))
        for ax in axes:
            ax.grid(alpha=0.2)
        return _finish(fig, output)
    if number in (5, 6):
        _require(data, "frequencies_hz", "amplitude")
        return plot_spectrum(data["frequencies_hz"], data["amplitude"], title=f"Figure {number}", output=output)
    if number == 7:
        _require(data, "matrix")
        return plot_frequency_heatmap(
            data["matrix"], xlabels=data.get("xlabels"), ylabels=data.get("ylabels"),
            title="Figure 7: frequency topography summary", output=output,
        )
    if number == 8:
        if "amplitude_harmonics" in data:
            return _render_figure8_cohort(data, output)
        _require(data, "frequencies_hz", "amplitude", "snr")
        return plot_spectrum(
            data["frequencies_hz"], data["amplitude"], snr_db=data["snr"],
            title="Figure 8: amplitude and SNR", output=output,
        )
    if number == 9:
        _require(data, "frequencies_hz", "scores")
        if np.asarray(data["scores"]).ndim == 4:
            return _render_figure9_cohort(data, output)
        return plot_shaded_series(
            data["frequencies_hz"], data["scores"], ylabel="Subjective score",
            title="Figure 9: subjective evaluation", output=output,
        )
    if number == 10:
        _require(data, "x", "accuracy", "itr")
        if np.asarray(data["accuracy"]).ndim == 4:
            return _render_figure10_cohort(data, output)
        return plot_accuracy_itr(data["x"], data["accuracy"], data["itr"], xlabel="Data length (s)", title="Figure 10", output=output)
    if number in (11, 13):
        if "composite_scores" in data:
            return _render_composite_scores(number, data, output)
        return _scatter_relationship(data, number, output)
    if np.asarray(data["accuracy"]).ndim == 3:
        return _render_figure12_cohort(data, output)
    _require(data, "frequencies_hz", "accuracy", "itr")
    return plot_accuracy_itr(
        data["frequencies_hz"], data["accuracy"], data["itr"],
        title="Figure 12: FBTRCA performance", output=output,
    )


def _render_figure8_cohort(data: Mapping[str, Any], output: Path) -> Any:
    import numpy as np

    _require(data, "frequencies_hz", "amplitude_harmonics", "snr_harmonics", "amplitude_maps", "snr_maps")
    plt = _pyplot()
    frequency = np.asarray(data["frequencies_hz"])
    amp = np.asarray(data["amplitude_harmonics"])[..., 0]
    snr = np.asarray(data["snr_harmonics"])[..., 0]
    amp_maps = np.asarray(data["amplitude_maps"]).mean(axis=0)
    snr_maps = np.asarray(data["snr_maps"]).mean(axis=0)
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for condition, label in enumerate(("Low depth", "High depth")):
        for ax, values, ylabel in ((axes[0, 0], amp[:, condition], "Amplitude (µV)"), (axes[0, 1], snr[:, condition], "SNR (dB)")):
            mean = np.nanmean(values, axis=0)
            sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            line, = ax.plot(frequency, mean, label=label)
            ax.fill_between(frequency, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
            ax.set(xlabel="Stimulus frequency (Hz)", ylabel=ylabel)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    for row, maps, label in ((1, amp_maps, "Amplitude (µV)"), (2, snr_maps, "SNR (dB)")):
        for condition in range(2):
            image = axes[row, condition].imshow(maps[condition], aspect="auto", origin="lower", extent=(1, 90, 1, 60))
            axes[row, condition].set(
                xlabel="Response frequency (Hz)", ylabel="Stimulus frequency (Hz)",
                title=("Low depth" if condition == 0 else "High depth") + f" — {label}",
            )
            fig.colorbar(image, ax=axes[row, condition], label=label)
    fig.suptitle("Figure 8: cohort amplitude and SNR")
    return _finish(fig, output)


def _render_figure9_cohort(data: Mapping[str, Any], output: Path) -> Any:
    import numpy as np
    from scipy.signal import savgol_filter

    scores = np.asarray(data["scores"], dtype=float)  # subject, condition, frequency, category
    frequency = np.asarray(data["frequencies_hz"])
    if scores.shape[1:] != (2, len(frequency), 3):
        raise ValueError("cohort scores must have shape (subject, 2, frequency, 3)")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for category, (ax, title) in enumerate(zip(axes, ("Comfort level", "Flicker perception", "Preference"))):
        for condition, label in enumerate(("Low depth", "High depth")):
            values = scores[:, condition, :, category]
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            smooth = savgol_filter(mean, 9, 3)
            line, = ax.plot(frequency, smooth, label=label)
            ax.fill_between(frequency, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
        ax.set(title=title, xlabel="Stimulus frequency (Hz)", ylim=(0.5, 5.2))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False)
    fig.suptitle("Figure 9: subjective evaluation")
    return _finish(fig, output)


def _render_figure10_cohort(data: Mapping[str, Any], output: Path) -> Any:
    import numpy as np

    x = np.asarray(data["x"])
    accuracy = np.asarray(data["accuracy"])
    itr = np.asarray(data["itr"])
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for band in range(3):
        for condition, label in enumerate(("Low depth", "High depth")):
            for row, values, ylabel in ((0, accuracy[:, condition, band] * 100, "Accuracy (%)"), (1, itr[:, condition, band], "ITR (bits/min)")):
                mean = values.mean(axis=0); sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
                line, = axes[row, band].plot(x, mean, label=label)
                axes[row, band].fill_between(x, mean-sem, mean+sem, color=line.get_color(), alpha=.2)
                axes[row, band].set(ylabel=ylabel, xlabel="Data length (s)")
                axes[row, band].grid(alpha=.2)
        axes[0, band].set_title(("Low", "Medium", "High")[band] + " frequency band")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Reference-study Figure 10: FBCCA performance")
    return _finish(fig, output)


def _render_figure12_cohort(data: Mapping[str, Any], output: Path) -> Any:
    import numpy as np

    frequency = np.asarray(data["frequencies_hz"]); accuracy = np.asarray(data["accuracy"]); itr = np.asarray(data["itr"])
    plt = _pyplot(); fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for condition, label in enumerate(("Low depth", "High depth")):
        for ax, values, ylabel in ((axes[0], accuracy[:, condition] * 100, "Accuracy (%)"), (axes[1], itr[:, condition], "ITR (bits/min)")):
            mean=values.mean(0); sem=values.std(0,ddof=1)/np.sqrt(values.shape[0]); line,=ax.plot(frequency,mean,label=label)
            ax.fill_between(frequency,mean-sem,mean+sem,color=line.get_color(),alpha=.2); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[1].set_xlabel("Stimulus frequency (Hz)"); axes[0].legend(frameon=False)
    fig.suptitle("Reference-study Figure 12: FBTRCA performance")
    return _finish(fig, output)


def _render_composite_scores(number: int, data: Mapping[str, Any], output: Path) -> Any:
    import numpy as np

    values=np.asarray(data["composite_scores"]); plt=_pyplot()
    if number == 11:
        # ratio, subject, condition, band, duration
        fig, axes=plt.subplots(2,4,figsize=(17,8),sharey=True)
        for ratio in range(2):
            for duration in range(4):
                mean=values[ratio,:,:,:,duration].mean(0); sem=values[ratio,:,:,:,duration].std(0,ddof=1)/np.sqrt(values.shape[1])
                x=np.arange(3); width=.34
                for condition,label in enumerate(("Low depth","High depth")):
                    axes[ratio,duration].bar(x+(condition-.5)*width,mean[condition],width,yerr=sem[condition],label=label)
                axes[ratio,duration].set_xticks(x,("Low","Medium","High")); axes[ratio,duration].set_ylim(0,1); axes[ratio,duration].set_title(f"{duration+1} s"); axes[ratio,duration].grid(axis="y",alpha=.2)
        axes[0,-1].legend(frameon=False); fig.suptitle("Reference-study Figure 11: composite performance and comfort")
    else:
        # ratio, subject, condition, frequency
        fig, axes=plt.subplots(2,1,figsize=(11,8),sharex=True); frequency=np.arange(1,values.shape[-1]+1)
        for ratio in range(2):
            for condition,label in enumerate(("Low depth","High depth")):
                series=values[ratio,:,condition]; mean=series.mean(0); sem=series.std(0,ddof=1)/np.sqrt(series.shape[0]); line,=axes[ratio].plot(frequency,mean,label=label)
                axes[ratio].fill_between(frequency,mean-sem,mean+sem,color=line.get_color(),alpha=.2)
            axes[ratio].set_ylim(0,1); axes[ratio].set_ylabel("Score"); axes[ratio].grid(alpha=.2)
        axes[1].set_xlabel("Stimulus frequency (Hz)"); axes[0].legend(frameon=False); fig.suptitle("Reference-study Figure 13: composite performance and comfort")
    return _finish(fig, output)


def load_reference_study_figure_data(path: str | Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


# Compatibility aliases for configurations created before the naming split.
FIGURE_CONTRACTS = REFERENCE_STUDY_FIGURE_CONTRACTS
render_paper_figure = render_reference_study_figure
load_figure_data = load_reference_study_figure_data
