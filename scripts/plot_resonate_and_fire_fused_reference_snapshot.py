"""Plot a completion-aware snapshot of the resumable fused-reference search."""
from pathlib import Path
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/experiments/resonate_and_fire_fused_reference_search/checkpoints"
OUT = ROOT / "outputs/experiments/resonate_and_fire_fused_reference_search/figures"
COUNTS = np.array((2, 4, 8, 16, 32))
CHANNELS = ("O1", "Oz", "O2", "O1-Oz", "O2-Oz")


def load() -> dict[int, list[dict[str, np.ndarray | float | int]]]:
    grouped = {int(count): [] for count in COUNTS}
    for path in SOURCE.glob("subject_*_classes.npz"):
        match = re.match(r"subject_(\d+)_(\d+)_classes", path.stem)
        if match is None:
            continue
        with np.load(path) as result:
            grouped[int(match.group(2))].append({
                "subject": int(match.group(1)),
                "baseline": float(result["baseline_three_channel_accuracy"]),
                "bipolar": float(result["optimized_bipolar_accuracy"]),
                "fused": float(result["fused_accuracy"]),
                "weights": np.asarray(result["selected_channel_weights"], float),
            })
    for values in grouped.values():
        values.sort(key=lambda item: int(item["subject"]))
    return grouped


def main() -> None:
    data = load()
    completed = sum(map(len, data.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    x = np.arange(len(COUNTS)); width = 0.25
    methods = (("baseline", "Original O1/Oz/O2", "#315a7d"),
               ("bipolar", "Optimized O1-Oz/O2-Oz", "#db7c26"),
               ("fused", "Five-channel fusion", "#16856b"))
    for offset, (key, label, color) in zip((-width, 0, width), methods):
        means = [100*np.mean([row[key] for row in data[int(count)]]) for count in COUNTS]
        sem = [100*np.std([row[key] for row in data[int(count)]], ddof=1)/np.sqrt(len(data[int(count)]))
               if len(data[int(count)]) > 1 else 0 for count in COUNTS]
        bars = axes[0, 0].bar(x+offset, means, width, yerr=sem, capsize=3, label=label, color=color)
        axes[0, 0].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[0, 0].set(xticks=x, xticklabels=[f"{c}\n(n={len(data[int(c)])})" for c in COUNTS],
                         xlabel="Classes and completed subjects", ylabel="Apparent accuracy (%)",
                         ylim=(0, 105), title="A  Completed-checkpoint mean")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for col, count in enumerate(COUNTS):
        rows = data[int(count)]
        delta = 100*np.array([row["fused"]-row["baseline"] for row in rows])
        axes[0, 1].scatter(np.full(len(delta), col), delta, s=28, alpha=.75)
        axes[0, 1].plot([col-.2, col+.2], [delta.mean(), delta.mean()], color="black", linewidth=2)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set(xticks=x, xticklabels=COUNTS, xlabel="Classes",
                         ylabel="Fusion minus original (points)", title="B  Subject-wise apparent gain")

    weights = np.stack([np.mean(np.stack([row["weights"] for row in data[int(count)]]), axis=0) for count in COUNTS])
    bottom = np.zeros(len(COUNTS))
    palette = ("#315a7d", "#6f8fad", "#9fb5c8", "#db7c26", "#f2b36f")
    for channel, color, values in zip(CHANNELS, palette, weights.T):
        axes[1, 0].bar(x, values, bottom=bottom, label=channel, color=color)
        bottom += values
    axes[1, 0].set(xticks=x, xticklabels=COUNTS, xlabel="Classes", ylabel="Mean selected weight",
                         ylim=(0, 1), title="C  Learned encoder-channel mixture")
    axes[1, 0].legend(frameon=False, ncols=3, fontsize=8)

    subject_ids = sorted({int(row["subject"]) for rows in data.values() for row in rows})
    matrix = np.full((len(subject_ids), len(COUNTS)), np.nan)
    for col, count in enumerate(COUNTS):
        lookup = {int(row["subject"]): 100*(row["fused"]-row["baseline"]) for row in data[int(count)]}
        for row, subject in enumerate(subject_ids):
            matrix[row, col] = lookup.get(subject, np.nan)
    limit = max(5, np.nanmax(np.abs(matrix)))
    image = axes[1, 1].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1, 1].set(xticks=x, xticklabels=COUNTS, yticks=np.arange(len(subject_ids)), yticklabels=subject_ids,
                         xlabel="Classes", ylabel="Subject", title="D  Fusion gain by completed cell")
    fig.colorbar(image, ax=axes[1, 1], label="Percentage points")
    fig.suptitle(f"R&F five-channel fusion — interim snapshot ({completed}/150 cells; apparent accuracy)", fontsize=15)
    path = OUT / "01_fused_reference_interim_snapshot.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {key: np.array([100*np.mean([row[key] for row in data[int(count)]]) for count in COUNTS])
               for key in ("baseline", "bipolar", "fused")}
    np.savez_compressed(OUT.parent / "interim_summary.npz", class_counts=COUNTS,
                        completed_per_class=np.array([len(data[int(count)]) for count in COUNTS]),
                        mean_weights=weights, channel_names=np.asarray(CHANNELS), **summary)
    print(path)
    print("completed", completed, "per_class", [len(data[int(count)]) for count in COUNTS])
    for index, count in enumerate(COUNTS):
        print(count, *(f"{key}={summary[key][index]:.2f}" for key in summary))


if __name__ == "__main__":
    main()
