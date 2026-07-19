"""Render PCA/t-SNE/UMAP diagnostics from a fused R&F checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ssvep_toolkit.visualization import compute_feature_embedding, plot_feature_embedding


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--method", choices=("pca", "tsne", "umap"), default="pca")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()
    source = ROOT / "outputs/experiments/resonate_and_fire_fused_reference_search/checkpoints" / (
        f"subject_{args.subject:02d}_{args.classes:02d}_classes.npz"
    )
    with np.load(source) as result:
        # channel, trial, target -> trial, channel*target
        features = np.asarray(result["channel_template_scores"]).transpose(1, 0, 2).reshape(-1, 5 * args.classes)
        prediction = np.asarray(result["fused_prediction"])
    labels = np.repeat(np.arange(args.classes), 12)
    embedding, metadata = compute_feature_embedding(
        features, args.method, random_state=args.seed, perplexity=args.perplexity,
    )
    output = ROOT / "outputs/experiments/resonate_and_fire_fused_reference_search/figures/feature_space" / (
        f"subject_{args.subject:02d}_{args.classes:02d}_classes_{args.method}_seed_{args.seed}.png"
    )
    plot_feature_embedding(
        embedding, labels, output,
        title=f"Subject {args.subject}: {args.classes}-class fused R&F {args.method.upper()} feature space",
        correctness=prediction == labels,
    )
    np.savez_compressed(output.with_suffix(".npz"), embedding=embedding, labels=labels,
                        prediction=prediction, features=features, **metadata)
    print(output)
    print(metadata)


if __name__ == "__main__":
    main()
