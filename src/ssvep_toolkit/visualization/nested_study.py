"""Evidence plots for validation-safe multi-encoder checkpoint comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def plot_nested_checkpoint_comparison(checkpoints: Sequence[str | Path], output: str | Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    records = []
    for checkpoint in checkpoints:
        with np.load(checkpoint) as source:
            records.append({
                "mode": str(source["filter_mode"]), "labels": source["labels"],
                "predictions": source["predictions"], "block": source["accuracy_by_block"],
                "report": json.loads(str(source["report_json"])),
                "dropout": float(source["rotating_branch_dropout_accuracy"]),
                "noise": float(source["feature_count_noise_accuracy"]),
                "gain": source["oof_selected_gain_per_uV"].reshape(-1),
            })
    records.sort(key=lambda item: (item["mode"] != "offline", item["mode"]))
    modes = [item["mode"].capitalize() for item in records]
    colors = ["#1769aa", "#d97706"][:len(records)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    width = .23; x = np.arange(len(records))
    clean = [100 * item["report"]["accuracy"] for item in records]
    dropout = [100 * item["dropout"] for item in records]
    noise = [100 * item["noise"] for item in records]
    axes[0, 0].bar(x-width, clean, width, label="Clean", color="#16856b")
    axes[0, 0].bar(x, dropout, width, label="Rotating branch dropout", color="#bf4b4b")
    axes[0, 0].bar(x+width, noise, width, label="10% count noise", color="#7c5cbf")
    axes[0, 0].set(xticks=x, xticklabels=modes, ylim=(0, 105), ylabel="Outer-block accuracy (%)",
                   title="A  Clean and perturbed accuracy")
    axes[0, 0].legend(frameon=False, fontsize=8)
    neural = [item["report"]["neural_window_itr_bits_per_minute"] for item in records]
    practical = [item["report"]["practical_itr_bits_per_minute"] for item in records]
    axes[0, 1].bar(x-.17, neural, .34, label="Neural window", color="#21918c")
    axes[0, 1].bar(x+.17, practical, .34, label="Practical", color="#355f8d")
    axes[0, 1].set(xticks=x, xticklabels=modes, ylabel="ITR (bits/min)", title="B  Latency-aware ITR")
    axes[0, 1].legend(frameon=False)
    for item, color in zip(records, colors):
        axes[0, 2].plot(np.arange(1, len(item["block"])+1), 100*item["block"], "o-",
                        label=item["mode"].capitalize(), color=color)
    axes[0, 2].set(xlabel="Held-out block", ylabel="Accuracy (%)", ylim=(0, 105),
                   title="C  Outer-fold variability")
    axes[0, 2].legend(frameon=False)
    for panel, item in zip((axes[1, 0], axes[1, 1]), records[:2]):
        labels = item["labels"].astype(int); predictions = item["predictions"].astype(int)
        classes = int(max(labels.max(), predictions.max()) + 1)
        confusion = np.zeros((classes, classes), int)
        np.add.at(confusion, (labels, predictions), 1)
        image = panel.imshow(confusion, cmap="Blues", vmin=0)
        for row in range(classes):
            for column in range(classes):
                panel.text(column, row, str(confusion[row, column]), ha="center", va="center",
                           color="white" if confusion[row, column] > confusion.max()/2 else "#17212b")
        panel.set(xlabel="Predicted class", ylabel="True class", title=f"{'D' if panel is axes[1,0] else 'E'}  {item['mode'].capitalize()} confusion")
        panel.set_xticks(range(classes)); panel.set_yticks(range(classes))
    axes[1, 2].boxplot([item["gain"] for item in records], tick_labels=modes, showfliers=False)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set(ylabel="Fold-selected gain per µV (log scale)", title="F  Segment × branch gain distribution")
    fig.suptitle("Nested multi-encoder pilot · 500 ms · outer held-out blocks", fontsize=15)
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight"); plt.close(fig)
    return target
