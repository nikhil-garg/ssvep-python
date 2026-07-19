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
    """Canonical correlations for variables-by-samples arrays."""
    import numpy as np
    from scipy.linalg import qr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be variables-by-samples with equal samples")
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    qx, _ = qr(x.T, mode="economic")
    qy, _ = qr(y.T, mode="economic")
    values = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return np.clip(np.real(values), 0, 1)
