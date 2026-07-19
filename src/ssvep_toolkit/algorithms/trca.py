from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def trca_intertrial_covariance(trials: Any, include_same_trial: bool = True) -> Any:
    """MATLAB `trca_S` for `(trial, channel, sample)` arrays."""
    import numpy as np

    x = np.asarray(trials, dtype=float)
    result = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if include_same_trial or i != j:
                result += x[i] @ x[j].T
    return result


@dataclass(frozen=True)
class TRCAModel:
    templates: Any  # class, channel, sample
    filters: Any  # class, channel


def fit_trca(trials: Any, *, include_same_trial: bool = True, regularization: float = 1e-8) -> TRCAModel:
    """Fit classes from `(class, trial, channel, sample)` data."""
    import numpy as np
    from scipy.linalg import eig

    data = np.asarray(trials, dtype=float)
    if data.ndim != 4:
        raise ValueError("trials must have shape (class, trial, channel, sample)")
    combined = data.transpose(2, 0, 1, 3).reshape(data.shape[2], -1)
    combined -= combined.mean(axis=1, keepdims=True)
    q = combined @ combined.T + regularization * np.eye(data.shape[2])
    filters = []
    for cls in range(data.shape[0]):
        s = trca_intertrial_covariance(data[cls], include_same_trial)
        values, vectors = eig(s, q)
        filters.append(np.real(vectors[:, int(np.argmax(np.real(values)))]))
    return TRCAModel(templates=data.mean(axis=1), filters=np.asarray(filters))


def _normalized_dot(a: Any, b: Any) -> float:
    import numpy as np

    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return 0.0 if denominator == 0 else float(np.sum(a * b) / denominator)


def predict_trca(model: TRCAModel, trial: Any, ensemble: bool = True) -> int:
    """Return a zero-based predicted class for `(channel, sample)` data."""
    import numpy as np

    x = np.asarray(trial, dtype=float)
    scores = []
    for cls, template in enumerate(model.templates):
        filters = model.filters.T if ensemble else model.filters[cls][:, None]
        scores.append(_normalized_dot(x.T @ filters, template.T @ filters))
    return int(np.argmax(scores))
