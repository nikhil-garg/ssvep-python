from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trca import TRCAModel, fit_trca


@dataclass(frozen=True)
class FBTRCAModel:
    models: tuple[TRCAModel, ...]
    weights: Any


def fit_fbtrca(
    subband_trials: Any,
    *,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
    include_same_trial: bool = True,
) -> FBTRCAModel:
    """Fit `(subband, class, trial, channel, sample)` training data."""
    import numpy as np

    data = np.asarray(subband_trials, dtype=float)
    if data.ndim != 5:
        raise ValueError("subband_trials must be 5-D")
    models = tuple(fit_trca(band, include_same_trial=include_same_trial) for band in data)
    weights = np.arange(1, len(models) + 1, dtype=float) ** (-weight_a) + weight_b
    return FBTRCAModel(models, weights)


def _normalized_dot(a: Any, b: Any) -> float:
    import numpy as np

    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return 0.0 if denominator == 0 else float(np.sum(a * b) / denominator)


def fbtrca_scores(model: FBTRCAModel, subband_trial: Any, ensemble: bool = True) -> Any:
    import numpy as np

    data = np.asarray(subband_trial, dtype=float)
    if data.shape[0] != len(model.models):
        raise ValueError("test subband count does not match model")
    correlations = np.zeros((len(model.models), model.models[0].templates.shape[0]))
    for band, band_model in enumerate(model.models):
        filters = band_model.filters.T if ensemble else band_model.filters[0][:, None]
        for cls, template in enumerate(band_model.templates):
            correlations[band, cls] = _normalized_dot(data[band].T @ filters, template.T @ filters)
    return model.weights @ (np.sign(correlations) * np.abs(correlations) ** 2)


def predict_fbtrca(model: FBTRCAModel, subband_trial: Any, ensemble: bool = True) -> int:
    import numpy as np

    return int(np.argmax(fbtrca_scores(model, subband_trial, ensemble)))

