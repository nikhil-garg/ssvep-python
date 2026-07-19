"""Complementarity-aware nested optimization for multi-encoder features.

The implementation deliberately keeps every selection operation inside the
outer training fold.  It supports confidence-bound candidate racing, joint
encoder/ridge beam search, continuous tie-break objectives, and retrained
encoder ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Sequence

from ssvep_toolkit.algorithms.linear_fusion import fit_linear_fusion
from .nested_fusion import CandidateFeatureBlock


@dataclass(frozen=True)
class FoldSafeCandidateFeatureBlock:
    """Lazily generate candidates from the current outer-training mask.

    Use this when preprocessing (for example training-derived gain) must be
    fitted separately inside every outer fold.
    """

    name: str
    candidate_ids: tuple[Any, ...]
    feature_names: tuple[str, ...]
    builder: Callable[[Any], CandidateFeatureBlock]

    def materialize(self, training_mask: Any, trials: int) -> CandidateFeatureBlock:
        block = self.builder(training_mask)
        if block.name != self.name or block.candidate_ids != self.candidate_ids or block.feature_names != self.feature_names:
            raise ValueError("fold-safe builder changed its declared candidate schema")
        block.validate(trials)
        return block


@dataclass(frozen=True)
class ObjectiveMetrics:
    accuracy: float
    standard_error: float
    margin: float
    log_loss: float
    spike_cost: float = 0.0


@dataclass(frozen=True)
class RacingConfig:
    enabled: bool = True
    stages: tuple[int, ...] = (4, 8)
    confidence_z: float = 1.0
    minimum_survivors: int = 8
    seed: int = 20260719


@dataclass(frozen=True)
class JointSearchConfig:
    screened_per_encoder: int = 8
    beam_width: int = 12
    selection_rule: str = "one_standard_error"
    margin_tolerance: float = 1e-12
    spike_cost_weight: float = 0.0
    pooling_strength: float = 0.0


@dataclass(frozen=True)
class AdvancedOuterFold:
    held_out_group: Any
    selected_candidate_indices: tuple[int, ...]
    selected_candidate_ids: tuple[Any, ...]
    selected_l2: float
    inner_metrics: ObjectiveMetrics
    screening_diagnostics: tuple[dict[str, Any], ...]
    joint_diagnostics: dict[str, Any]
    outer_accuracy: float


@dataclass(frozen=True)
class AdvancedNestedFusionResult:
    predictions: Any
    decision_scores: Any
    folds: tuple[AdvancedOuterFold, ...]
    accuracy: float
    retrained_ablation_predictions: dict[str, Any]
    retrained_ablation_accuracy: dict[str, float]
    stability: dict[str, Any]


def _features(blocks: Sequence[CandidateFeatureBlock], indices: Sequence[int]) -> tuple[Any, tuple[str, ...]]:
    import numpy as np

    matrices = [np.asarray(block.values[index], dtype=float).reshape(np.asarray(block.values).shape[1], -1)
                for block, index in zip(blocks, indices)]
    return np.concatenate(matrices, axis=1), tuple(name for block in blocks for name in block.feature_names)


def _softmax_log_loss(scores: Any, labels: Any, classes: Any) -> float:
    import numpy as np

    values = np.asarray(scores, dtype=float)
    values = values - values.max(axis=1, keepdims=True)
    log_denominator = np.log(np.exp(values).sum(axis=1))
    lookup = {value: index for index, value in enumerate(classes.tolist())}
    encoded = np.asarray([lookup[value] for value in np.asarray(labels).tolist()], dtype=int)
    return float(np.mean(log_denominator - values[np.arange(encoded.size), encoded]))


def _classification_margin(scores: Any, labels: Any, classes: Any) -> float:
    import numpy as np

    values = np.asarray(scores, dtype=float)
    lookup = {value: index for index, value in enumerate(classes.tolist())}
    encoded = np.asarray([lookup[value] for value in np.asarray(labels).tolist()], dtype=int)
    correct = values[np.arange(encoded.size), encoded]
    masked = values.copy(); masked[np.arange(encoded.size), encoded] = -np.inf
    return float(np.mean(correct - masked.max(axis=1)))


def _combo_cost(blocks: Sequence[CandidateFeatureBlock], indices: Sequence[int], training_mask: Any) -> float:
    import numpy as np

    mask = np.asarray(training_mask, dtype=bool); total = 0.0
    for block, index in zip(blocks, indices):
        if block.candidate_costs is None:
            continue
        cost = np.asarray(block.candidate_costs, dtype=float)[index]
        total += float(cost if np.ndim(cost) == 0 else np.mean(cost[mask]))
    return total


def _cross_validated_combo(
    blocks: Sequence[CandidateFeatureBlock], indices: Sequence[int], labels: Any, groups: Any,
    training_mask: Any, l2_grid: Sequence[float],
) -> tuple[float, ObjectiveMetrics, Any]:
    """Evaluate a candidate combination and ridge grid on grouped inner folds."""
    import numpy as np

    truth = np.asarray(labels); fold_groups = np.asarray(groups); mask = np.asarray(training_mask, bool)
    features, names = _features(blocks, indices)
    classes = np.unique(truth)
    inner = np.unique(fold_groups[mask])
    fold_accuracy = np.empty((len(l2_grid), inner.size), dtype=float)
    all_scores = [[None] * inner.size for _ in l2_grid]
    all_truth = []
    for fold_index, held_out in enumerate(inner):
        validation = mask & (fold_groups == held_out); train = mask & (fold_groups != held_out)
        all_truth.append(truth[validation])
        for l2_index, l2 in enumerate(l2_grid):
            model = fit_linear_fusion(features[train], truth[train], l2=float(l2), feature_names=names)
            scores = model.decision_scores(features[validation]); all_scores[l2_index][fold_index] = scores
            fold_accuracy[l2_index, fold_index] = np.mean(model.predict(features[validation]) == truth[validation])
    means = fold_accuracy.mean(axis=1)
    best = int(np.flatnonzero(means == means.max())[-1])
    se = fold_accuracy.std(axis=1, ddof=1) / np.sqrt(max(inner.size, 1)) if inner.size > 1 else np.zeros(len(l2_grid))
    eligible = np.flatnonzero(means >= means[best] - se[best])
    chosen = int(eligible[np.argmax(np.asarray(l2_grid, dtype=float)[eligible])])
    stacked_scores = np.concatenate(all_scores[chosen], axis=0); stacked_truth = np.concatenate(all_truth)
    metrics = ObjectiveMetrics(
        accuracy=float(means[chosen]), standard_error=float(se[chosen]),
        margin=_classification_margin(stacked_scores, stacked_truth, classes),
        log_loss=_softmax_log_loss(stacked_scores, stacked_truth, classes),
        spike_cost=_combo_cost(blocks, indices, mask),
    )
    return float(l2_grid[chosen]), metrics, fold_accuracy


def _reference_distance(candidate_id: Any, reference: dict[str, float] | None) -> float:
    import numpy as np

    if not reference or not isinstance(candidate_id, dict):
        return 0.0
    terms = []
    for name, target in reference.items():
        if name not in candidate_id or not isinstance(candidate_id[name], (int, float)):
            continue
        value = float(candidate_id[name]); target = float(target)
        terms.append((np.log(value / target) if value > 0 and target > 0 else value - target) ** 2)
    return float(np.sqrt(sum(terms))) if terms else 0.0


def pooled_reference(
    individual: dict[str, float], population: dict[str, float], reliability: float,
) -> dict[str, float]:
    """Geometrically shrink positive individual parameters toward a population prior."""
    import numpy as np

    weight = float(np.clip(reliability, 0.0, 1.0)); result = dict(population)
    for name in set(individual) | set(population):
        if name not in individual:
            result[name] = float(population[name]); continue
        if name not in population:
            result[name] = float(individual[name]); continue
        left = float(individual[name]); right = float(population[name])
        result[name] = float(np.exp(weight * np.log(left) + (1 - weight) * np.log(right))) \
            if left > 0 and right > 0 else weight * left + (1 - weight) * right
    return result


def _screen_block(
    block: CandidateFeatureBlock, labels: Any, groups: Any, training_mask: Any, *,
    selection_l2: float, keep: int, racing: RacingConfig,
    reference: dict[str, float] | None, pooling_strength: float = 0.0,
) -> tuple[list[int], dict[str, Any]]:
    """Confidence-bound racing; never drops candidates whose intervals overlap."""
    import numpy as np

    truth = np.asarray(labels); fold_groups = np.asarray(groups); outer_train = np.asarray(training_mask, bool)
    values = np.asarray(block.values, dtype=float).reshape(len(block.candidate_ids), truth.size, -1)
    inner = np.unique(fold_groups[outer_train]); rng = np.random.default_rng(racing.seed)
    inner = inner[rng.permutation(inner.size)]
    stages = sorted({min(inner.size, int(stage)) for stage in racing.stages if int(stage) > 0} | {inner.size}) \
        if racing.enabled else [inner.size]
    accuracy = np.full((values.shape[0], inner.size), np.nan); margins = np.full_like(accuracy, np.nan)
    active = np.arange(values.shape[0]); previous = 0; history = []
    classes = np.unique(truth)
    for limit in stages:
        evaluated = active.copy()
        for candidate in evaluated:
            for fold in range(previous, limit):
                validation = outer_train & (fold_groups == inner[fold]); train = outer_train & ~validation
                model = fit_linear_fusion(values[candidate, train], truth[train], l2=selection_l2,
                                          feature_names=block.feature_names)
                scores = model.decision_scores(values[candidate, validation])
                accuracy[candidate, fold] = np.mean(model.predict(values[candidate, validation]) == truth[validation])
                margins[candidate, fold] = _classification_margin(scores, truth[validation], classes)
        retained = evaluated
        if limit < inner.size and evaluated.size > racing.minimum_survivors:
            means = np.nanmean(accuracy[evaluated, :limit], axis=1)
            if limit > 1:
                se = np.nanstd(accuracy[evaluated, :limit], axis=1, ddof=1) / np.sqrt(limit)
            else:
                se = np.ones(evaluated.size)
            lower = means - racing.confidence_z * se; upper = means + racing.confidence_z * se
            threshold = float(np.max(lower)); retained = evaluated[upper >= threshold]
            if retained.size < racing.minimum_survivors:
                order = np.lexsort((evaluated, -means)); retained = evaluated[order[:racing.minimum_survivors]]
        history.append({"folds": int(limit), "evaluated": evaluated.tolist(), "retained": retained.tolist()})
        active = retained; previous = limit
    means = np.nanmean(accuracy[active], axis=1); ses = np.nanstd(accuracy[active], axis=1, ddof=1) / np.sqrt(inner.size)
    mean_margin = np.nanmean(margins[active], axis=1)
    best_position = int(np.argmax(means)); cutoff = means[best_position] - ses[best_position]
    eligible_mask = means >= cutoff
    eligible = active[eligible_mask]
    best_index = int(active[int(np.argmax(means))])
    best_id = block.candidate_ids[best_index]
    individual = ({name: float(value) for name, value in best_id.items()
                   if isinstance(value, (int, float))} if isinstance(best_id, dict) else {})
    effective_reference = pooled_reference(
        individual, reference, 1.0 - float(np.clip(pooling_strength, 0, 1)),
    ) if reference and individual else reference
    ranking = sorted(active.tolist(), key=lambda index: (
        -(np.nanmean(accuracy[index])), -np.nanmean(margins[index]),
        _reference_distance(block.candidate_ids[index], effective_reference), index,
    ))
    # Always include statistically eligible candidates first, then fill the beam.
    eligible_order = [index for index in ranking if index in set(eligible.tolist())]
    pool = (eligible_order + [index for index in ranking if index not in set(eligible_order)])[:max(1, keep)]
    return pool, {
        "history": history, "survivors": active.tolist(), "eligible": eligible.tolist(),
        "selected_pool": pool, "mean_accuracy": {str(i): float(np.nanmean(accuracy[i])) for i in active},
        "mean_margin": {str(i): float(np.nanmean(margins[i])) for i in active},
        "effective_pooled_reference": effective_reference,
        "model_fits": int(np.isfinite(accuracy).sum()),
    }


def _choose_joint(
    blocks: Sequence[CandidateFeatureBlock], pools: Sequence[Sequence[int]], labels: Any, groups: Any,
    training_mask: Any, l2_grid: Sequence[float], config: JointSearchConfig,
) -> tuple[tuple[int, ...], float, ObjectiveMetrics, dict[str, Any]]:
    import numpy as np

    beam: list[tuple[int, ...]] = [()]; stages = []
    cache: dict[tuple[int, ...], tuple[float, ObjectiveMetrics, Any]] = {}
    for depth, pool in enumerate(pools, start=1):
        candidates = [prefix + (index,) for prefix, index in product(beam, pool)]
        active_blocks = blocks[:depth]
        for combo in candidates:
            cache[combo] = _cross_validated_combo(active_blocks, combo, labels, groups, training_mask, l2_grid)
        best_accuracy = max(cache[combo][1].accuracy for combo in candidates)
        best_combo = max(candidates, key=lambda combo: cache[combo][1].accuracy)
        cutoff = best_accuracy - cache[best_combo][1].standard_error
        eligible = [combo for combo in candidates if cache[combo][1].accuracy >= cutoff]
        ranked = sorted(candidates, key=lambda combo: (
            -cache[combo][1].accuracy, -cache[combo][1].margin,
            cache[combo][1].log_loss,
            config.spike_cost_weight * cache[combo][1].spike_cost, combo,
        ))
        protected = [combo for combo in ranked if combo in set(eligible)]
        beam = (protected + [combo for combo in ranked if combo not in set(protected)])[:config.beam_width]
        stages.append({"depth": depth, "evaluated": len(candidates), "eligible": len(eligible), "beam": beam})
    best_combo = max(beam, key=lambda combo: cache[combo][1].accuracy)
    cutoff = cache[best_combo][1].accuracy - cache[best_combo][1].standard_error
    eligible = [combo for combo in beam if config.selection_rule != "one_standard_error"
                or cache[combo][1].accuracy >= cutoff]
    selected = min(eligible, key=lambda combo: (
        -cache[combo][1].margin, cache[combo][1].log_loss,
        config.spike_cost_weight * cache[combo][1].spike_cost, combo,
    ))
    l2, metrics, _ = cache[selected]
    return selected, l2, metrics, {"stages": stages, "eligible_final": eligible, "selected": selected}


def parameter_stability(folds: Sequence[AdvancedOuterFold], block_names: Sequence[str]) -> dict[str, Any]:
    """Summarize fold-selection concentration without treating folds as independent subjects."""
    from collections import Counter

    result: dict[str, Any] = {}
    for block_index, name in enumerate(block_names):
        serialized = [repr(fold.selected_candidate_ids[block_index]) for fold in folds]
        counts = Counter(serialized); mode, frequency = counts.most_common(1)[0]
        result[name] = {
            "unique_candidates": len(counts), "mode_candidate": mode,
            "mode_fraction": frequency / len(folds), "selection_counts": dict(counts),
        }
    return result


def advanced_nested_grouped_fusion(
    blocks: Sequence[CandidateFeatureBlock | FoldSafeCandidateFeatureBlock], labels: Any, groups: Any, *,
    l2_grid: Sequence[float], candidate_selection_l2: float = 1.0,
    reference_by_block: dict[str, dict[str, float]] | None = None,
    racing: RacingConfig = RacingConfig(), joint: JointSearchConfig = JointSearchConfig(),
    retrained_ablation_blocks: Sequence[str] = (),
    retrained_feature_ablations: dict[str, Callable[[str], bool]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AdvancedNestedFusionResult:
    """Run joint encoder/ridge selection with untouched outer-block evaluation."""
    import numpy as np

    truth = np.asarray(labels); fold_groups = np.asarray(groups); outer_values = np.unique(fold_groups)
    if not blocks or truth.ndim != 1 or fold_groups.shape != truth.shape:
        raise ValueError("blocks, labels, and groups must be aligned")
    for block in blocks:
        if isinstance(block, CandidateFeatureBlock):
            block.validate(truth.size)
    classes = np.unique(truth); predictions = np.empty_like(truth)
    scores = np.empty((truth.size, classes.size), dtype=float); folds = []
    feature_ablations = dict(retrained_feature_ablations or {})
    ablation_names = tuple(retrained_ablation_blocks) + tuple(feature_ablations)
    if len(set(ablation_names)) != len(ablation_names):
        raise ValueError("retrained ablation names must be unique")
    ablation_predictions = {name: np.empty_like(truth) for name in ablation_names}
    total = len(outer_values) * (1 + len(ablation_names)); completed = 0

    def select_and_fit(active_blocks: Sequence[CandidateFeatureBlock], outer_train: Any, seed_offset: int = 0):
        diagnostics = []; pools = []
        for block_index, block in enumerate(active_blocks):
            local_racing = RacingConfig(racing.enabled, racing.stages, racing.confidence_z,
                                        racing.minimum_survivors, racing.seed + seed_offset + block_index)
            pool, diagnostic = _screen_block(
                block, truth, fold_groups, outer_train, selection_l2=candidate_selection_l2,
                keep=joint.screened_per_encoder, racing=local_racing,
                reference=(reference_by_block or {}).get(block.name),
                pooling_strength=joint.pooling_strength,
            )
            pools.append(pool); diagnostics.append(diagnostic)
        selected, l2, metrics, joint_diagnostic = _choose_joint(
            active_blocks, pools, truth, fold_groups, outer_train, l2_grid, joint,
        )
        matrix, names = _features(active_blocks, selected)
        model = fit_linear_fusion(matrix[outer_train], truth[outer_train], l2=l2, feature_names=names)
        return selected, model, matrix, metrics, tuple(diagnostics), joint_diagnostic

    for outer_index, held_out in enumerate(outer_values):
        outer_test = fold_groups == held_out; outer_train = ~outer_test
        materialized = tuple(
            block.materialize(outer_train, truth.size) if isinstance(block, FoldSafeCandidateFeatureBlock) else block
            for block in blocks
        )
        selected, model, matrix, metrics, diagnostics, joint_diagnostic = select_and_fit(
            materialized, outer_train, outer_index * 1000,
        )
        predictions[outer_test] = model.predict(matrix[outer_test]); scores[outer_test] = model.decision_scores(matrix[outer_test])
        outer_accuracy = float(np.mean(predictions[outer_test] == truth[outer_test]))
        folds.append(AdvancedOuterFold(
            held_out, selected, tuple(block.candidate_ids[index] for block, index in zip(materialized, selected)),
            model.l2, metrics, diagnostics, joint_diagnostic, outer_accuracy,
        ))
        completed += 1
        if progress_callback: progress_callback(completed, total)
        for ablation_name in ablation_names:
            if ablation_name in feature_ablations:
                predicate = feature_ablations[ablation_name]; reduced_list = []
                for block in materialized:
                    keep = np.asarray([not predicate(name) for name in block.feature_names], dtype=bool)
                    if not np.any(keep):
                        continue
                    values = np.asarray(block.values).reshape(len(block.candidate_ids), truth.size, -1)[..., keep]
                    reduced_list.append(CandidateFeatureBlock(
                        block.name, values, block.candidate_ids,
                        tuple(name for name, retained in zip(block.feature_names, keep) if retained),
                        block.candidate_costs,
                    ))
                reduced = tuple(reduced_list)
            else:
                reduced = tuple(block for block in materialized if block.name != ablation_name)
            if not reduced:
                raise ValueError("a retrained ablation cannot remove every feature block")
            _, ablation_model, ablation_matrix, _, _, _ = select_and_fit(
                reduced, outer_train, outer_index * 1000 + 100,
            )
            ablation_predictions[ablation_name][outer_test] = ablation_model.predict(ablation_matrix[outer_test])
            completed += 1
            if progress_callback: progress_callback(completed, total)
    return AdvancedNestedFusionResult(
        predictions, scores, tuple(folds), float(np.mean(predictions == truth)), ablation_predictions,
        {name: float(np.mean(values == truth)) for name, values in ablation_predictions.items()},
        parameter_stability(folds, [block.name for block in blocks]),
    )
