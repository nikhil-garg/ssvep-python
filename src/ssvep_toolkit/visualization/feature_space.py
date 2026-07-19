"""PCA/t-SNE/UMAP feature-space diagnostics for encoder outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


def standardized_features(features: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("features must be a 2D array with at least three trials")
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=1), 1e-8)
    return (values - center) / scale, center, scale


def compute_feature_embedding(
    features: Any,
    method: Literal["pca", "tsne", "umap"] = "pca",
    *,
    random_state: int = 0,
    perplexity: float = 30.0,
    neighbors: int = 15,
) -> tuple[Any, dict[str, Any]]:
    """Return a two-dimensional unsupervised embedding and audit metadata."""
    import numpy as np

    standardized, _, _ = standardized_features(features)
    if method == "pca":
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        embedding = standardized @ right[:2].T
        variance = singular ** 2
        ratio = variance[:2] / max(float(variance.sum()), 1e-12)
        return embedding, {"method": method, "explained_variance_ratio": ratio, "random_state": None}
    if method == "tsne":
        if not 1 <= perplexity < standardized.shape[0]:
            raise ValueError("t-SNE perplexity must be at least 1 and below the trial count")
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise RuntimeError("t-SNE requires the optional 'analysis' dependencies") from exc
        pre_dimensions = min(50, standardized.shape[0] - 1, standardized.shape[1])
        pre_pca = compute_pca_components(standardized, pre_dimensions)
        model = TSNE(n_components=2, perplexity=perplexity, init="pca",
                     learning_rate="auto", random_state=random_state)
        return model.fit_transform(pre_pca), {
            "method": method, "perplexity": perplexity, "random_state": random_state,
            "pre_pca_dimensions": pre_dimensions,
        }
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP requires the optional 'analysis' dependencies") from exc
        model = umap.UMAP(n_components=2, n_neighbors=neighbors, random_state=random_state)
        return model.fit_transform(standardized), {
            "method": method, "neighbors": neighbors, "random_state": random_state,
        }
    raise ValueError("method must be pca, tsne, or umap")


def compute_pca_components(standardized: Any, dimensions: int) -> Any:
    import numpy as np

    values = np.asarray(standardized, dtype=float)
    _, _, right = np.linalg.svd(values, full_matrices=False)
    return values @ right[:dimensions].T


def plot_feature_embedding(
    embedding: Any,
    labels: Any,
    output: str | Path,
    *,
    title: str,
    correctness: Any | None = None,
) -> Path:
    """Plot class identity by color and optional correctness by marker."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(embedding, dtype=float)
    truth = np.asarray(labels)
    if points.shape != (truth.size, 2):
        raise ValueError("embedding must have shape (trial, 2) and match labels")
    correct = np.ones(truth.size, dtype=bool) if correctness is None else np.asarray(correctness, dtype=bool)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    classes = np.unique(truth)
    colors = plt.cm.turbo(np.linspace(0, 1, classes.size))
    if classes.size > 16 and np.issubdtype(classes.dtype, np.number):
        color_values = np.searchsorted(classes, truth)
        shown = ax.scatter(points[correct, 0], points[correct, 1], c=color_values[correct],
                           cmap="turbo", marker="o", s=28, alpha=.75,
                           vmin=0, vmax=classes.size-1)
        if np.any(~correct):
            ax.scatter(points[~correct, 0], points[~correct, 1], c=color_values[~correct],
                       cmap="turbo", marker="x", s=42, linewidths=1.2,
                       vmin=0, vmax=classes.size-1)
        colorbar = fig.colorbar(shown, ax=ax)
        colorbar.set_label("Class index")
    else:
        for class_value, color in zip(classes, colors):
            selected = truth == class_value
            ax.scatter(points[selected & correct, 0], points[selected & correct, 1], color=color,
                       marker="o", s=28, alpha=.75, label=str(class_value))
            if np.any(selected & ~correct):
                ax.scatter(points[selected & ~correct, 0], points[selected & ~correct, 1],
                           color=color, marker="x", s=42, linewidths=1.2)
    ax.set(xlabel="Embedding dimension 1", ylabel="Embedding dimension 2", title=title)
    if classes.size <= 16:
        ax.legend(title="Class", frameon=False, ncols=2)
    ax.grid(alpha=.2)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
