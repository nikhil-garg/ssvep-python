from __future__ import annotations

from typing import Any

from .cca import canonical_correlations


def fbcca_scores(
    subbands: Any,
    references: Any,
    *,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
    subband_count: int | None = None,
) -> Any:
    """Scores for `(subband, channel, sample)` and `(class, ref, sample)`."""
    import numpy as np

    data = np.asarray(subbands)
    refs = np.asarray(references)
    count = data.shape[0] if subband_count is None else min(subband_count, data.shape[0])
    correlations = np.zeros((count, refs.shape[0]))
    for band in range(count):
        for cls in range(refs.shape[0]):
            correlations[band, cls] = canonical_correlations(data[band], refs[cls])[0]
    weights = np.arange(1, count + 1, dtype=float) ** (-weight_a) + weight_b
    return weights @ (np.sign(correlations) * np.abs(correlations) ** 2)


def predict_fbcca(subbands: Any, references: Any, **kwargs: Any) -> int:
    """Return a zero-based class index."""
    import numpy as np

    return int(np.argmax(fbcca_scores(subbands, references, **kwargs)))

