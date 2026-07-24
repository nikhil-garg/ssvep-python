from __future__ import annotations

from typing import Any, Sequence


def reference_signals(
    frequencies_hz: Sequence[float],
    samples: int,
    sampling_rate_hz: float,
    harmonics: int = 10,
) -> Any:
    """Return `(class, reference_component, sample)` sine/cosine references."""
    import numpy as np

    time = np.arange(1, samples + 1, dtype=float) / sampling_rate_hz
    references = []
    for frequency in frequencies_hz:
        components = []
        for harmonic in range(1, harmonics + 1):
            components.extend((
                np.sin(2 * np.pi * harmonic * frequency * time),
                np.cos(2 * np.pi * harmonic * frequency * time),
            ))
        references.append(components)
    return np.asarray(references)


def canonical_correlations(x: Any, y: Any, regularization: float = 1e-10) -> Any:
    """Return CCA correlations for ``(variable, sample)`` inputs.

    ``regularization`` is a non-negative ridge value applied to both
    covariance matrices before whitening; it makes rank-deficient inputs
    well-defined without changing the unregularized API.
    """
    import numpy as np
    from scipy.linalg import eigh

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be variables-by-samples with equal samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    scale = max(x.shape[1] - 1, 1)
    cxx = x @ x.T / scale + regularization * np.eye(x.shape[0])
    cyy = y @ y.T / scale + regularization * np.eye(y.shape[0])
    cxy = x @ y.T / scale
    x_values, x_vectors = eigh(cxx)
    y_values, y_vectors = eigh(cyy)
    if np.any(x_values <= 0) or np.any(y_values <= 0):
        raise ValueError("regularization must make covariance matrices positive definite")
    inv_root_x = (x_vectors / np.sqrt(x_values)) @ x_vectors.T
    inv_root_y = (y_vectors / np.sqrt(y_values)) @ y_vectors.T
    values = np.linalg.svd(inv_root_x @ cxy @ inv_root_y, compute_uv=False)
    return np.clip(np.real(values), 0, 1)
