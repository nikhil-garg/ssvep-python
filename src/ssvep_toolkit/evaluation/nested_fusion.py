"""Nested grouped evaluation for validation-safe multi-encoder fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ssvep_toolkit.algorithms.linear_fusion import (
    fit_linear_fusion,
    select_l2_grouped,
)


@dataclass(frozen=True)
class CandidateFeatureBlock:
    """Trial-aligned raw features for alternative encoder parameter choices."""
    name: str
    values: Any  # candidate, trial, feature...
    candidate_ids: tuple[Any, ...]
    feature_names: tuple[str, ...]
    candidate_costs: Any | None = None

    def validate(self, trials: int) -> None:
        import numpy as np

        values = np.asarray(self.values)
        if values.ndim < 3 or values.shape[1] != trials:
            raise ValueError(f"{self.name} values must be (candidate, trial, feature...)")
        if values.shape[0] != len(self.candidate_ids):
            raise ValueError(f"{self.name} candidate_ids do not match candidate axis")
        features = int(np.prod(values.shape[2:]))
        if features != len(self.feature_names):
            raise ValueError(f"{self.name} feature_names do not match flattened features")
        if self.candidate_costs is not None and np.asarray(self.candidate_costs).shape not in {
            (values.shape[0],), (values.shape[0], trials),
        }:
            raise ValueError(f"{self.name} candidate_costs must be candidate or candidate-by-trial")


@dataclass(frozen=True)
class OuterFoldSelection:
    held_out_group: Any
    selected_candidate_indices: tuple[int, ...]
    selected_candidate_ids: tuple[Any, ...]
    selected_l2: float
    inner_candidate_accuracy: tuple[Any, ...]
    inner_candidate_diagnostics: tuple[dict[str, Any], ...]
    inner_l2_accuracy: Any
    outer_accuracy: float


@dataclass(frozen=True)
class NestedFusionResult:
    predictions: Any
    decision_scores: Any
    out_of_fold_features: Any
    feature_names: tuple[str, ...]
    folds: tuple[OuterFoldSelection, ...]
    accuracy: float
    accuracy_by_group: Any
    perturbed_predictions: dict[str, Any]
    perturbed_accuracy: dict[str, float]


def _select_candidate(
    block: CandidateFeatureBlock,
    labels: Any,
    groups: Any,
    outer_train: Any,
    *,
    selection_l2: float,
    selection_rule: str = "max_mean",
    reference: dict[str, float] | None = None,
    candidate_progress: Callable[[], None] | None = None,
    fidelity: dict[str, Any] | None = None,
    fidelity_seed: int = 0,
) -> tuple[int, Any, dict[str, Any]]:
    import numpy as np

    values = np.asarray(block.values, dtype=float).reshape(len(block.candidate_ids), len(labels), -1)
    inner_groups = np.unique(groups[outer_train])
    rng = np.random.default_rng(int(fidelity_seed)); inner_groups = inner_groups[rng.permutation(inner_groups.size)]
    accuracy = np.full((values.shape[0], inner_groups.size), np.nan, dtype=float)
    use_fidelity = bool((fidelity or {}).get("enabled", False))
    requested = [int(value) for value in (fidelity or {}).get("folds", ())]
    stages = sorted({min(inner_groups.size, value) for value in requested if value > 0} | {inner_groups.size}) if use_fidelity else [inner_groups.size]
    retain = [float(value) for value in (fidelity or {}).get("retain_fractions", ())]
    active = np.arange(values.shape[0]); previous = 0; stage_history = []
    for stage_index, fold_limit in enumerate(stages):
        evaluated = active.copy()
        for candidate in evaluated:
            for fold in range(previous, fold_limit):
                held_out = inner_groups[fold]
                validation = outer_train & (groups == held_out)
                training = outer_train & (groups != held_out)
                model = fit_linear_fusion(
                    values[candidate, training], labels[training], l2=selection_l2,
                    feature_names=block.feature_names,
                )
                accuracy[candidate, fold] = np.mean(
                    model.predict(values[candidate, validation]) == labels[validation]
                )
            if candidate_progress is not None: candidate_progress()
        retained = evaluated
        if stage_index < len(stages) - 1:
            fraction = retain[min(stage_index, len(retain) - 1)] if retain else 0.5
            if not 0 < fraction <= 1: raise ValueError("retain_fractions must be in (0, 1]")
            count = max(1, int(np.ceil(evaluated.size * fraction)))
            partial_mean = np.nanmean(accuracy[evaluated, :fold_limit], axis=1)
            retained = evaluated[np.lexsort((evaluated, -partial_mean))[:count]]
        stage_history.append({
            "folds_evaluated": int(fold_limit), "candidates_evaluated": evaluated.astype(int).tolist(),
            "candidates_retained": retained.astype(int).tolist(),
        })
        active = retained; previous = fold_limit
    finalists = active
    mean = np.nanmean(accuracy, axis=1)
    evaluated_folds = np.sum(np.isfinite(accuracy), axis=1)
    standard_error = np.nanstd(accuracy, axis=1, ddof=1) / np.sqrt(np.maximum(evaluated_folds, 1))
    standard_error = np.nan_to_num(standard_error)
    finalist_mean = mean[finalists]
    best = int(finalists[np.flatnonzero(finalist_mean == finalist_mean.max())[0]])
    cutoff = float(mean[best] - standard_error[best])
    eligible = finalists[mean[finalists] >= cutoff] if selection_rule == "one_standard_error" else np.asarray([best])
    if selection_rule not in {"max_mean", "one_standard_error"}:
        raise ValueError("selection_rule must be 'max_mean' or 'one_standard_error'")

    distances = np.zeros(values.shape[0], dtype=float)
    if reference:
        distances.fill(np.inf)
        for index, candidate_id in enumerate(block.candidate_ids):
            if not isinstance(candidate_id, dict):
                continue
            terms = []
            for name, target in reference.items():
                if name not in candidate_id:
                    continue
                value = float(candidate_id[name]); target = float(target)
                terms.append((np.log(value / target) if value > 0 and target > 0 else value - target) ** 2)
            if terms:
                distances[index] = float(np.sqrt(np.sum(terms)))
    # The reference is a predeclared physiologically reasonable operating point.
    # It is used only among candidates statistically indistinguishable from the
    # best inner-fold mean; it never sees the outer test block.
    selected = int(eligible[np.lexsort((eligible, standard_error[eligible], distances[eligible]))[0]])

    boundary_hits: dict[str, str] = {}
    searched_parameter_count = 0
    searched_parameters: list[str] = []
    chosen = block.candidate_ids[selected]
    if isinstance(chosen, dict):
        for name, value in chosen.items():
            numeric = [float(item[name]) for item in block.candidate_ids
                       if isinstance(item, dict) and name in item and isinstance(item[name], (int, float))]
            if len(set(numeric)) <= 1 or not isinstance(value, (int, float)):
                continue
            searched_parameter_count += 1
            searched_parameters.append(name)
            at_low = np.isclose(float(value), min(numeric)); at_high = np.isclose(float(value), max(numeric))
            if at_low or at_high:
                boundary_hits[name] = "lower" if at_low else "upper"
    diagnostic = {
        "selection_rule": selection_rule,
        "best_mean_index": best,
        "selected_index": selected,
        "best_mean_accuracy": float(mean[best]),
        "selected_mean_accuracy": float(mean[selected]),
        "selected_standard_error": float(standard_error[selected]),
        "one_standard_error_cutoff": cutoff,
        "eligible_indices": eligible.astype(int).tolist(),
        "boundary_hits": boundary_hits,
        "searched_parameter_count": searched_parameter_count,
        "searched_parameters": searched_parameters,
        "multi_fidelity_enabled": use_fidelity,
        "multi_fidelity_stages": stage_history,
        "finalist_indices": finalists.astype(int).tolist(),
        "model_fits": int(np.isfinite(accuracy).sum()),
    }
    return selected, accuracy, diagnostic


def nested_grouped_linear_fusion(
    blocks: Sequence[CandidateFeatureBlock],
    labels: Any,
    groups: Any,
    *,
    l2_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    candidate_selection_l2: float = 1.0,
    candidate_selection_rule: str = "max_mean",
    candidate_reference_by_block: dict[str, dict[str, float]] | None = None,
    candidate_fidelity: dict[str, Any] | None = None,
    l2_selection_rule: str = "max_mean",
    outer_test_transform: Callable[[Any, Any, tuple[str, ...], Any], Any] | None = None,
    outer_test_transforms: dict[str, Callable[[Any, Any, tuple[str, ...], Any], Any]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NestedFusionResult:
    """Select encoder candidates and fusion only inside each outer fold.

    Feature arrays must be raw deterministic encodings. Any data-derived scale,
    template, channel weight, or normalization must already have been generated
    fold-safely; same-data calibrated score matrices are not valid inputs.
    """
    import numpy as np

    truth = np.asarray(labels)
    fold_groups = np.asarray(groups)
    if truth.ndim != 1 or fold_groups.shape != truth.shape:
        raise ValueError("labels and groups must be aligned one-dimensional arrays")
    if not blocks:
        raise ValueError("at least one encoder feature block is required")
    for block in blocks:
        block.validate(truth.size)
    classes = np.unique(truth)
    outer_values = np.unique(fold_groups)
    predictions = np.empty(truth.shape, dtype=classes.dtype)
    scores = np.empty((truth.size, classes.size), dtype=float)
    combined_names = tuple(name for block in blocks for name in block.feature_names)
    out_of_fold = np.empty((truth.size, len(combined_names)), dtype=float)
    perturbations = dict(outer_test_transforms or {})
    perturbed_predictions = {name: np.empty(truth.shape, dtype=classes.dtype) for name in perturbations}
    fold_results = []
    accuracy_by_group = np.empty(outer_values.size, dtype=float)
    def candidate_stage_work(count: int) -> int:
        if not bool((candidate_fidelity or {}).get("enabled", False)): return count
        stages = sorted({min(outer_values.size - 1, int(value)) for value in
                         (candidate_fidelity or {}).get("folds", ()) if int(value) > 0} | {outer_values.size - 1})
        fractions = [float(value) for value in (candidate_fidelity or {}).get("retain_fractions", ())]
        active = count; work = 0
        for index, _ in enumerate(stages):
            work += active
            if index < len(stages) - 1:
                fraction = fractions[min(index, len(fractions) - 1)] if fractions else .5
                active = max(1, int(np.ceil(active * fraction)))
        return work
    total_work = outer_values.size * (sum(candidate_stage_work(len(block.candidate_ids)) for block in blocks) + 1)
    completed_work = 0
    last_reported_percent = -1

    def report_work() -> None:
        nonlocal completed_work, last_reported_percent
        completed_work += 1
        percent = min(100, int(100 * completed_work / max(total_work, 1)))
        if progress_callback is not None and percent > last_reported_percent:
            last_reported_percent = percent
            progress_callback(percent, 100)

    for outer_index, held_out in enumerate(outer_values):
        outer_test = fold_groups == held_out
        outer_train = ~outer_test
        selected_indices = []
        selected_ids = []
        candidate_accuracy = []
        candidate_diagnostics = []
        selected_blocks = {}
        for block in blocks:
            selected, inner_accuracy, diagnostic = _select_candidate(
                block, truth, fold_groups, outer_train,
                selection_l2=candidate_selection_l2,
                selection_rule=candidate_selection_rule,
                reference=(candidate_reference_by_block or {}).get(block.name),
                candidate_progress=report_work,
                fidelity=candidate_fidelity,
                fidelity_seed=int((candidate_fidelity or {}).get("seed", 0)) + outer_index,
            )
            selected_indices.append(selected)
            selected_ids.append(block.candidate_ids[selected])
            candidate_accuracy.append(inner_accuracy)
            candidate_diagnostics.append(diagnostic)
            selected_blocks[block.name] = np.asarray(block.values[selected]).reshape(truth.size, -1)
        # Candidate blocks are already validated and flattened in declared order.
        # Keep the caller's semantic feature names instead of regenerating generic
        # ``block:index`` labels, because channel and spike-stream identity is a
        # required part of the fusion result.
        combined = np.concatenate(tuple(selected_blocks.values()), axis=1)
        model, l2_accuracy = select_l2_grouped(
            combined[outer_train], truth[outer_train], fold_groups[outer_train],
            l2_grid=l2_grid, feature_names=combined_names,
            selection_rule=l2_selection_rule,
        )
        report_work()
        evaluated_test = combined[outer_test]
        if outer_test_transform is not None:
            evaluated_test = np.asarray(
                outer_test_transform(evaluated_test.copy(), combined[outer_train], combined_names, held_out),
                dtype=float,
            )
            if evaluated_test.shape != combined[outer_test].shape:
                raise ValueError("outer_test_transform changed the feature matrix shape")
        fold_scores = model.decision_scores(evaluated_test)
        fold_prediction = model.predict(evaluated_test)
        predictions[outer_test] = fold_prediction
        scores[outer_test] = fold_scores
        out_of_fold[outer_test] = evaluated_test
        for name, transform in perturbations.items():
            perturbed = np.asarray(
                transform(combined[outer_test].copy(), combined[outer_train], combined_names, held_out),
                dtype=float,
            )
            if perturbed.shape != combined[outer_test].shape:
                raise ValueError(f"outer_test_transforms[{name!r}] changed the feature matrix shape")
            perturbed_predictions[name][outer_test] = model.predict(perturbed)
        outer_accuracy = float(np.mean(fold_prediction == truth[outer_test]))
        accuracy_by_group[outer_index] = outer_accuracy
        fold_results.append(OuterFoldSelection(
            held_out_group=held_out,
            selected_candidate_indices=tuple(selected_indices),
            selected_candidate_ids=tuple(selected_ids),
            selected_l2=model.l2,
            inner_candidate_accuracy=tuple(candidate_accuracy),
            inner_candidate_diagnostics=tuple(candidate_diagnostics),
            inner_l2_accuracy=l2_accuracy,
            outer_accuracy=outer_accuracy,
        ))
    return NestedFusionResult(
        predictions=predictions,
        decision_scores=scores,
        out_of_fold_features=out_of_fold,
        feature_names=combined_names,
        folds=tuple(fold_results),
        accuracy=float(np.mean(predictions == truth)),
        accuracy_by_group=accuracy_by_group,
        perturbed_predictions=perturbed_predictions,
        perturbed_accuracy={name: float(np.mean(value == truth)) for name, value in perturbed_predictions.items()},
    )
