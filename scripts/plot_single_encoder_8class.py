"""Create a compact six-panel report for the focused single-encoder study."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load(root: Path, encoder: str) -> list[dict[str, object]]:
    result = []
    for path in sorted((root / "checkpoints").glob(f"{encoder}_subject_*_08_classes.npz")):
        with np.load(path, allow_pickle=False) as data:
            result.append({key: data[key] for key in data.files})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/experiments/single_encoder_8class")
    args = parser.parse_args(); output = args.root / "figures"; output.mkdir(parents=True, exist_ok=True)
    rf, lif = load(args.root, "resonate_fire"), load(args.root, "lif")
    if not rf and not lif: raise SystemExit("No single-encoder checkpoints found.")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for encoder, records, color in (("R&F", rf, "#1769aa"), ("BPF→LIF", lif, "#16856b")):
        if not records: continue
        subjects = [int(item["subject_id"]) for item in records]
        accuracy = [100 * float(item["accuracy"]) for item in records]
        axes[0, 0].plot(subjects, accuracy, "o-", label=encoder, color=color)
        block_values = np.concatenate([item["block_accuracy"] for item in records]) * 100
        axes[0, 1].hist(block_values, bins=np.arange(-6.25, 106.26, 12.5), alpha=.45, label=encoder, color=color)
        selected = np.concatenate([item["selected_parameters_per_block"] for item in records])
        names = [str(x) for x in records[0]["parameter_names"]]
        axes[0, 2].boxplot(selected, positions=np.arange(len(names)) + (.15 if encoder == "BPF→LIF" else -.15), widths=.25)
        axes[1, 0].hist(np.concatenate([item["candidate_oof_accuracy"] for item in records]) * 100, bins=20, alpha=.45, label=encoder, color=color)
        first = records[0]; grid, surface = first["parameter_grid"], first["candidate_oof_accuracy"] * 100
        axes[1, 1].scatter(grid[:, 0], surface, s=18, alpha=.65, label=encoder, color=color)
        axes[1, 2].scatter(grid[:, 1], surface, s=18, alpha=.65, label=encoder, color=color)
        axes[1, 1].set_xlabel(names[0]); axes[1, 2].set_xlabel(names[1])
    axes[0, 0].set(title="Subject-level held-block accuracy", xlabel="Subject", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0, 1].set(title="Held-block accuracy distribution", xlabel="Accuracy (%)", ylabel="Blocks")
    axes[0, 2].set(title="Selected parameter distributions", ylabel="Selected value")
    axes[0, 2].set_xticks(range(len([str(x) for x in (rf or lif)[0]["parameter_names"]])))
    axes[0, 2].set_xticklabels([str(x) for x in (rf or lif)[0]["parameter_names"]], rotation=25, ha="right")
    axes[1, 0].set(title="All candidate held-block accuracies", xlabel="Candidate OOF accuracy (%)", ylabel="Candidates")
    axes[1, 1].set(title="Parameter 1 effect (first completed subject)", ylabel="OOF accuracy (%)")
    axes[1, 2].set(title="Parameter 2 effect (first completed subject)", ylabel="OOF accuracy (%)")
    for ax in axes.flat: ax.grid(alpha=.2)
    axes[0, 0].legend(); axes[0, 1].legend(); axes[1, 0].legend()
    fig.savefig(output / "single_encoder_8class_overview.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__": main()
