"""Leakage-aware linear fusion for channel and encoder feature blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LinearFusionModel:
    classes: Any
    feature_mean: Any
    feature_scale: Any
    coefficients: Any
    intercept: Any
    l2: float
    feature_names: tuple[str, ...]

    def decision_scores(self, features: Any) -> Any:
        import numpy as np

        values = np.asarray(features, dtype=float)
        standardized = (values - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept

    def predict(self, features: Any) -> Any:
        import numpy as np

        return self.classes[np.argmax(self.decision_scores(features), axis=1)]


def concatenate_feature_blocks(blocks: Mapping[str, Any]) -> tuple[Any, tuple[str, ...]]:
    """Concatenate trial-aligned blocks while retaining auditable feature names."""
    import numpy as np

    if not blocks:
        raise ValueError("at least one feature block is required")
    matrices = []
    names = []
    trials = None
    for block_name, block in blocks.items():
        values = np.asarray(block, dtype=float)
        if values.ndim < 2:
            raise ValueError(f"feature block {block_name!r} must be trial-first and at least 2D")
        matrix = values.reshape(values.shape[0], -1)
        if trials is None:
            trials = matrix.shape[0]
        elif matrix.shape[0] != trials:
            raise ValueError("all feature blocks must contain the same trials")
        matrices.append(matrix)
        names.extend(f"{block_name}:{index}" for index in range(matrix.shape[1]))
    return np.concatenate(matrices, axis=1), tuple(names)


def fit_linear_fusion(
    features: Any,
    labels: Any,
    *,
    l2: float = 1.0,
    feature_names: Sequence[str] | None = None,
) -> LinearFusionModel:
    """Fit standardized multiclass ridge least-squares scores.

    Scaling statistics and coefficients are learned only from the supplied
    training rows. The intercept is not regularized.
    """
    import numpy as np

    values = np.asarray(features, dtype=float)
    truth = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] != truth.size:
        raise ValueError("features must be 2D and trial-aligned with labels")
    if l2 < 0:
        raise ValueError("l2 must be nonnegative")
    classes, encoded = np.unique(truth, return_inverse=True)
    if classes.size < 2:
        raise ValueError("at least two classes are required")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=1), 1e-8)
    standardized = (values - mean) / scale
    design = np.column_stack((standardized, np.ones(values.shape[0])))
    targets = np.eye(classes.size)[encoded]
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0.0
    solution = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    names = tuple(feature_names or (f"feature:{i}" for i in range(values.shape[1])))
    if len(names) != values.shape[1]:
        raise ValueError("feature_names length must equal the feature count")
    return LinearFusionModel(classes, mean, scale, solution[:-1], solution[-1], float(l2), names)


def select_l2_grouped(
    features: Any,
    labels: Any,
    groups: Any,
    l2_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    *,
    feature_names: Sequence[str] | None = None,
    selection_rule: str = "max_mean",
) -> tuple[LinearFusionModel, Any]:
    """Select regularization by leave-one-group-out inner validation."""
    import numpy as np

    values = np.asarray(features, dtype=float)
    truth = np.asarray(labels)
    fold_groups = np.asarray(groups)
    if truth.size != fold_groups.size or values.shape[0] != truth.size:
        raise ValueError("features, labels, and groups must be trial-aligned")
    grid = np.asarray(l2_grid, dtype=float)
    if grid.size == 0 or np.any(grid < 0):
        raise ValueError("l2_grid must contain nonnegative values")
    fold_values = np.unique(fold_groups)
    accuracy = np.empty((grid.size, fold_values.size), dtype=float)
    for parameter_index, l2 in enumerate(grid):
        for fold_index, held_out in enumerate(fold_values):
            test = fold_groups == held_out
            train = ~test
            model = fit_linear_fusion(values[train], truth[train], l2=float(l2), feature_names=feature_names)
            accuracy[parameter_index, fold_index] = np.mean(model.predict(values[test]) == truth[test])
    mean_accuracy = accuracy.mean(axis=1)
    best_mean = int(np.flatnonzero(mean_accuracy == mean_accuracy.max())[-1])
    if selection_rule == "max_mean":
        eligible = np.flatnonzero(mean_accuracy == mean_accuracy.max())
    elif selection_rule == "one_standard_error":
        standard_error = accuracy.std(axis=1, ddof=1) / np.sqrt(fold_values.size)
        cutoff = mean_accuracy[best_mean] - np.nan_to_num(standard_error[best_mean])
        eligible = np.flatnonzero(mean_accuracy >= cutoff)
    else:
        raise ValueError("selection_rule must be 'max_mean' or 'one_standard_error'")
    # Among statistically indistinguishable models prefer stronger ridge
    # regularization, reducing variance without consulting the outer test fold.
    best = int(eligible[np.argmax(grid[eligible])])
    return fit_linear_fusion(values, truth, l2=float(grid[best]), feature_names=feature_names), accuracy
