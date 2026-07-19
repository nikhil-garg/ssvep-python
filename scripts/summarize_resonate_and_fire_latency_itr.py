"""Create completion-aware accuracy and latency-ITR curves from R&F checkpoints."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from ssvep_toolkit.algorithms import latency_itr_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = ROOT / "outputs/experiments/resonate_and_fire_decision_endpoints"
DEEP = ROOT / "outputs/experiments/resonate_and_fire_deep_gain_search/checkpoints"
COUNTS = np.array((2, 4, 8, 16, 32))
ENDPOINT_MS = np.array((250, 500, 750, 1000))


def load_accuracy() -> np.ndarray:
    accuracy = np.full((30, len(COUNTS), len(ENDPOINT_MS)), np.nan)
    for subject in range(1, 31):
        for class_index, count in enumerate(COUNTS):
            for endpoint_index, endpoint in enumerate(ENDPOINT_MS[:-1]):
                path = ENDPOINT / "checkpoints" / f"endpoint_{endpoint:04d}ms_subject_{subject:02d}_{count:02d}_classes.npz"
                if path.exists():
                    with np.load(path) as result:
                        accuracy[subject-1, class_index, endpoint_index] = float(result["accuracy"])
            path = DEEP / f"subject_{subject:02d}_{count:02d}_classes.npz"
            if path.exists():
                with np.load(path) as result:
                    accuracy[subject-1, class_index, -1] = float(result["accuracy"])
    return accuracy


def main() -> None:
    accuracy = load_accuracy()
    mean = np.nanmean(accuracy, axis=0)
    completed = np.sum(np.isfinite(accuracy), axis=0)
    neural_itr = np.full_like(mean, np.nan)
    practical_itr = np.full_like(mean, np.nan)
    for class_index, count in enumerate(COUNTS):
        report = latency_itr_report(
            int(count), mean[class_index], ENDPOINT_MS / 1000,
            onset_latency_seconds=0.14, practical_overhead_seconds=1.0,
        )
        neural_itr[class_index] = report["neural_window_itr_bits_per_minute"]
        practical_itr[class_index] = report["practical_itr_bits_per_minute"]

    output = ENDPOINT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(.08, .92, len(COUNTS)))
    for index, (count, color) in enumerate(zip(COUNTS, colors)):
        labels = [f"{value:.0f}\n(n={n})" for value, n in zip(100*mean[index], completed[index])]
        axes[0].plot(ENDPOINT_MS, 100*mean[index], marker="o", color=color, label=f"{count} classes")
        for x, y, label in zip(ENDPOINT_MS, 100*mean[index], labels):
            if np.isfinite(y):
                axes[0].annotate(label, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        axes[1].plot(ENDPOINT_MS, neural_itr[index], marker="o", color=color)
        axes[2].plot(ENDPOINT_MS, practical_itr[index], marker="o", color=color)
    axes[0].set(xlabel="Decision window (ms)", ylabel="Apparent accuracy (%)", title="Accuracy")
    axes[1].set(xlabel="Decision window (ms)", ylabel="ITR (bits/min)", title="Neural-window ITR (+140 ms latency)")
    axes[2].set(xlabel="Decision window (ms)", ylabel="ITR (bits/min)", title="Practical ITR (+1 s command overhead)")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(ENDPOINT_MS)
    figure = output / "01_accuracy_and_latency_itr_snapshot.png"
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        ENDPOINT / "latency_itr_snapshot.npz", class_counts=COUNTS,
        endpoint_ms=ENDPOINT_MS, subject_accuracy=accuracy, mean_accuracy=mean,
        completed_subjects=completed, neural_window_itr=neural_itr,
        practical_itr=practical_itr, onset_latency_seconds=.14,
        practical_overhead_seconds=1.0, evaluation_design="apparent_accuracy_completion_aware_snapshot",
    )
    print(figure)
    for index, count in enumerate(COUNTS):
        best_neural = int(np.nanargmax(neural_itr[index]))
        best_practical = int(np.nanargmax(practical_itr[index]))
        print(
            f"{count} classes: neural best={ENDPOINT_MS[best_neural]}ms {neural_itr[index,best_neural]:.2f} "
            f"practical best={ENDPOINT_MS[best_practical]}ms {practical_itr[index,best_practical]:.2f}",
        )


if __name__ == "__main__":
    main()
