from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def _pyplot():
    import os
    import tempfile
    from pathlib import Path

    cache = Path(tempfile.gettempdir()) / "ssvep-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig: Any, output: str | Path | None, dpi: int = 160) -> Any:
    fig.tight_layout()
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_shaded_series(
    x: Any,
    values: Any,
    *,
    labels: Sequence[str] | None = None,
    ylabel: str = "Value",
    xlabel: str = "Frequency (Hz)",
    title: str | None = None,
    output: str | Path | None = None,
) -> Any:
    """Plot mean ± standard error for arrays shaped `(series, observation, x)`."""
    import numpy as np

    plt = _pyplot()
    data = np.asarray(values, dtype=float)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or data.shape[-1] != len(x):
        raise ValueError("values must have shape (observation, x) or (series, observation, x)")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, series in enumerate(data):
        mean = np.nanmean(series, axis=0)
        count = np.maximum(np.sum(np.isfinite(series), axis=0), 1)
        sem = np.nanstd(series, axis=0, ddof=1) / np.sqrt(count)
        label = labels[index] if labels else f"Series {index + 1}"
        line, = ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.2)
    if data.shape[0] > 1 or labels:
        ax.legend(frameon=False)
    return _finish(fig, output)


def plot_spectrum(
    frequencies_hz: Any,
    amplitude: Any,
    *,
    snr_db: Any | None = None,
    stimulus_hz: float | None = None,
    maximum_hz: float = 100.0,
    title: str | None = None,
    output: str | Path | None = None,
) -> Any:
    """Plot amplitude and optional spectral SNR."""
    import numpy as np

    plt = _pyplot()
    f = np.asarray(frequencies_hz)
    amp = np.asarray(amplitude)
    amp = np.nanmean(amp.reshape(-1, amp.shape[-1]), axis=0) if amp.ndim > 1 else amp
    mask = f <= maximum_hz
    rows = 2 if snr_db is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.5 * rows), sharex=True, squeeze=False)
    axes[0, 0].plot(f[mask], amp[mask])
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title(title or "Amplitude spectrum")
    if stimulus_hz is not None:
        for harmonic in range(1, int(maximum_hz // stimulus_hz) + 1):
            axes[0, 0].axvline(harmonic * stimulus_hz, color="tab:red", alpha=0.25, linewidth=0.8)
    if snr_db is not None:
        snr = np.asarray(snr_db)
        if snr.ndim > 1:
            flat = snr.reshape(-1, snr.shape[-1])
            counts = np.sum(np.isfinite(flat), axis=0)
            snr = np.divide(np.nansum(flat, axis=0), counts, out=np.full(flat.shape[-1], np.nan), where=counts > 0)
        axes[1, 0].plot(f[mask], snr[mask])
        axes[1, 0].set_ylabel("SNR (dB)")
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.grid(alpha=0.2)
    return _finish(fig, output)


def plot_frequency_heatmap(
    matrix: Any,
    *,
    xlabels: Sequence[Any] | None = None,
    ylabels: Sequence[Any] | None = None,
    colorbar_label: str = "Value",
    title: str | None = None,
    output: str | Path | None = None,
) -> Any:
    import numpy as np

    plt = _pyplot()
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    if xlabels is not None:
        ticks = np.linspace(0, len(xlabels) - 1, min(10, len(xlabels)), dtype=int)
        ax.set_xticks(ticks, [str(xlabels[x]) for x in ticks])
    if ylabels is not None:
        ticks = np.arange(len(ylabels))
        ax.set_yticks(ticks, [str(y) for y in ylabels])
    ax.set(title=title, xlabel="Frequency", ylabel="Channel / condition")
    return _finish(fig, output)


def plot_topography(
    values: Any,
    xy: Any,
    *,
    channel_names: Sequence[str] | None = None,
    title: str | None = None,
    colorbar_label: str = "Value",
    output: str | Path | None = None,
) -> Any:
    """Interpolate channel values over normalized 2-D electrode coordinates."""
    import numpy as np
    from scipy.interpolate import griddata

    plt = _pyplot()
    z = np.asarray(values, dtype=float).reshape(-1)
    points = np.asarray(xy, dtype=float)
    if points.shape != (z.size, 2):
        raise ValueError("xy must have shape (channels, 2)")
    gx, gy = np.mgrid[-1:1:200j, -1:1:200j]
    gz = griddata(points, z, (gx, gy), method="cubic")
    mask = gx**2 + gy**2 > 1
    gz[mask] = np.nan
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(gz.T, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1))
    ax.scatter(points[:, 0], points[:, 1], s=12, color="black")
    if channel_names:
        for (x, y), name in zip(points, channel_names):
            ax.text(x, y, name, fontsize=7, ha="center", va="bottom")
    ax.set(title=title, aspect="equal", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    ax.axis("off")
    fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8)
    return _finish(fig, output)


def plot_accuracy_itr(
    x: Any,
    accuracy: Any,
    itr: Any | None = None,
    *,
    labels: Sequence[str] | None = None,
    xlabel: str = "Frequency (Hz)",
    title: str | None = None,
    output: str | Path | None = None,
) -> Any:
    import numpy as np

    plt = _pyplot()
    acc = np.asarray(accuracy, dtype=float)
    if acc.ndim == 1:
        acc = acc[None, :]
    fig, axes = plt.subplots(2 if itr is not None else 1, 1, figsize=(9, 7 if itr is not None else 4), sharex=True, squeeze=False)
    for i, row in enumerate(acc):
        axes[0, 0].plot(x, 100 * row, label=labels[i] if labels else f"Series {i + 1}")
    axes[0, 0].set(ylabel="Accuracy (%)", title=title)
    axes[0, 0].set_ylim(0, 105)
    if itr is not None:
        rates = np.asarray(itr, dtype=float)
        if rates.ndim == 1:
            rates = rates[None, :]
        for i, row in enumerate(rates):
            axes[1, 0].plot(x, row, label=labels[i] if labels else f"Series {i + 1}")
        axes[1, 0].set_ylabel("ITR (bits/min)")
    axes[-1, 0].set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.grid(alpha=0.2)
        if acc.shape[0] > 1 or labels:
            ax.legend(frameon=False)
    return _finish(fig, output)


def plot_score_distribution(
    scores: Any,
    *,
    category_names: Sequence[str] = ("Comfort", "Flicker perception", "Preference"),
    title: str | None = None,
    output: str | Path | None = None,
) -> Any:
    """Box plots for scores shaped `(observation, category)` or `(group, observation, category)`."""
    import numpy as np

    plt = _pyplot()
    values = np.asarray(scores, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != len(category_names):
        raise ValueError("scores must end with a category axis")
    fig, axes = plt.subplots(1, values.shape[0], figsize=(5 * values.shape[0], 4), sharey=True, squeeze=False)
    for group, ax in enumerate(axes[0]):
        ax.boxplot([values[group, :, i] for i in range(values.shape[-1])], tick_labels=category_names)
        ax.set_title(f"Group {group + 1}" if values.shape[0] > 1 else (title or "Scores"))
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Score")
    if title and values.shape[0] > 1:
        fig.suptitle(title)
    return _finish(fig, output)
