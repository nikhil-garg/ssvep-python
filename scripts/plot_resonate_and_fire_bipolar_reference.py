"""Plot the 30-subject comparison of monopolar and Oz-referenced R&F readouts."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "outputs/experiments/resonate_and_fire_bipolar_reference/checkpoints"
FIGURES = ROOT / "outputs/experiments/resonate_and_fire_bipolar_reference/figures"
CLASS_COUNTS = np.array([2, 4, 8, 16, 32])
SUBJECTS = np.arange(1, 31)


def load_results() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.empty((len(SUBJECTS), len(CLASS_COUNTS)))
    template = np.empty_like(baseline)
    direct = np.empty_like(baseline)
    for row, subject in enumerate(SUBJECTS):
        for col, classes in enumerate(CLASS_COUNTS):
            path = CHECKPOINTS / f"subject_{subject:02d}_{classes:02d}_classes.npz"
            with np.load(path) as result:
                baseline[row, col] = result["baseline_three_channel_accuracy"]
                template[row, col] = result["bipolar_template_accuracy"]
                direct[row, col] = result["bipolar_direct_spike_accuracy"]
    return 100 * baseline, 100 * template, 100 * direct


def main() -> None:
    baseline, template, direct = load_results()
    delta = template - baseline
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1])

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(CLASS_COUNTS))
    width = 0.25
    colors = ("#315a7d", "#db7c26", "#8b96a1")
    for offset, values, label, color in (
        (-width, baseline, "O1 + Oz + O2 (template)", colors[0]),
        (0, template, "O1-Oz + O2-Oz (template)", colors[1]),
        (width, direct, "O1-Oz + O2-Oz (direct spikes)", colors[2]),
    ):
        means = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        bars = ax.bar(x + offset, means, width, yerr=sem, capsize=3, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, CLASS_COUNTS)
    ax.set_xlabel("Number of classes")
    ax.set_ylabel("Apparent accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("A  Oz referencing reduces mean accuracy")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    rng = np.random.default_rng(7)
    for col, classes in enumerate(CLASS_COUNTS):
        jitter = rng.normal(0, 0.025, len(SUBJECTS))
        ax.scatter(np.full(len(SUBJECTS), col) + jitter, delta[:, col], s=23,
                   color=np.where(delta[:, col] > 0, "#16856b", "#bf4b4b"), alpha=0.75)
        ax.plot([col - 0.22, col + 0.22], [np.mean(delta[:, col])] * 2,
                color="black", linewidth=2.5)
        ax.text(col, 6, f"{np.sum(delta[:, col] > 0)}/30 improve", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, CLASS_COUNTS)
    ax.set_xlabel("Number of classes")
    ax.set_ylabel("Bipolar-template minus 3-channel accuracy (points)")
    ax.set_ylim(min(-55, delta.min() - 4), max(10, delta.max() + 4))
    ax.set_title("B  A minority of subjects benefit")

    ax = fig.add_subplot(grid[1, 0])
    limit = max(10, np.ceil(np.max(np.abs(delta)) / 5) * 5)
    image = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(x, CLASS_COUNTS)
    ax.set_yticks(np.arange(len(SUBJECTS)), SUBJECTS, fontsize=7)
    ax.set_xlabel("Number of classes")
    ax.set_ylabel("Subject")
    ax.set_title("C  Subject-specific accuracy change (percentage points)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    colorbar.set_label("Bipolar − 3-channel")

    ax = fig.add_subplot(grid[1, 1])
    markers = ("o", "s", "^", "D", "P")
    class_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(CLASS_COUNTS)))
    for col, (classes, marker, color) in enumerate(zip(CLASS_COUNTS, markers, class_colors)):
        ax.scatter(baseline[:, col], delta[:, col], label=f"{classes} classes", marker=marker,
                   color=color, edgecolor="white", linewidth=0.4, s=46, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Original 3-channel apparent accuracy (%)")
    ax.set_ylabel("Bipolar-template change (points)")
    ax.set_title("D  Benefit is individual, not a general replacement")
    ax.legend(frameon=False, ncols=2, fontsize=8)

    fig.suptitle(
        "Resonate-and-fire spatial reference test: O1−Oz and O2−Oz\n"
        "30 subjects, 1 s endpoint; bipolar readout reuses parameters selected for the original three channels",
        fontsize=15,
    )
    output = FIGURES / "01_bipolar_reference_accuracy_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
