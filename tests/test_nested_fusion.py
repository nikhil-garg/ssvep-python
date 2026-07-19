import numpy as np

from ssvep_toolkit.evaluation import CandidateFeatureBlock, nested_grouped_linear_fusion


def test_nested_fusion_selects_candidates_without_outer_block_access() -> None:
    rng = np.random.default_rng(12)
    classes = 3
    blocks = 8
    labels = np.tile(np.arange(classes), blocks)
    groups = np.repeat(np.arange(blocks), classes)
    one_hot = np.eye(classes)[labels]
    rf = np.stack((
        one_hot + rng.normal(0, 0.08, one_hot.shape),
        rng.normal(size=one_hot.shape),
    ))
    delta = np.stack((
        one_hot[:, :2] + rng.normal(0, 0.12, (labels.size, 2)),
        rng.normal(size=(labels.size, 2)),
    ))
    result = nested_grouped_linear_fusion(
        (
            CandidateFeatureBlock("rf", rf, ("informative", "noise"), ("rf:0", "rf:1", "rf:2")),
            CandidateFeatureBlock("delta", delta, ("informative", "noise"), ("delta:0", "delta:1")),
        ),
        labels,
        groups,
        l2_grid=(0.01, 1.0, 100.0),
    )
    assert result.accuracy > 0.9
    assert result.predictions.shape == labels.shape
    assert result.out_of_fold_features.shape == (labels.size, 5)
    assert len(result.folds) == blocks
    assert all(fold.selected_candidate_indices == (0, 0) for fold in result.folds)


def test_outer_test_transform_does_not_modify_training_selection() -> None:
    labels = np.tile(np.arange(2), 6)
    groups = np.repeat(np.arange(6), 2)
    features = np.eye(2)[labels][None, ...]
    seen = []
    def erase(test, train, names, held_out):
        seen.append((held_out, train.shape, names))
        return np.zeros_like(test)
    result = nested_grouped_linear_fusion(
        (CandidateFeatureBlock("rf", features, ("only",), ("rf:a", "rf:b")),),
        labels, groups, outer_test_transform=erase,
    )
    assert len(seen) == 6
    assert result.accuracy == .5
    assert np.all(result.out_of_fold_features == 0)


def test_nested_fusion_reports_completed_outer_fold_progress() -> None:
    labels = np.tile(np.arange(2), 6)
    groups = np.repeat(np.arange(6), 2)
    features = np.eye(2)[labels][None, ...]
    progress = []
    nested_grouped_linear_fusion(
        (CandidateFeatureBlock("rf", features, ("only",), ("rf:a", "rf:b")),),
        labels, groups, progress_callback=lambda current, total: progress.append((current, total)),
    )
    assert progress[-1] == (100, 100)
    assert all(left[0] < right[0] for left, right in zip(progress, progress[1:]))


def test_multi_fidelity_prunes_candidates_only_with_inner_folds() -> None:
    labels = np.tile(np.arange(2), 6); groups = np.repeat(np.arange(6), 2)
    perfect = np.eye(2)[labels]
    values = np.stack([perfect for _ in range(8)])
    candidates = tuple({"threshold": float(index + 1)} for index in range(8))
    result = nested_grouped_linear_fusion(
        (CandidateFeatureBlock("rf", values, candidates, ("rf:a", "rf:b")),),
        labels, groups, candidate_selection_rule="one_standard_error",
        candidate_reference_by_block={"rf": {"threshold": 1.0}},
        candidate_fidelity={"enabled": True, "folds": [2, 4], "retain_fractions": [.5, .5], "seed": 7},
    )
    for fold in result.folds:
        diagnostic = fold.inner_candidate_diagnostics[0]
        assert diagnostic["multi_fidelity_enabled"]
        assert diagnostic["model_fits"] < 8 * 5
        assert len(diagnostic["finalist_indices"]) == 2


def test_multiple_test_time_ablations_reuse_the_same_outer_models() -> None:
    labels = np.tile(np.arange(2), 6); groups = np.repeat(np.arange(6), 2)
    features = np.eye(2)[labels][None, ...]
    result = nested_grouped_linear_fusion(
        (CandidateFeatureBlock("rf", features, ("only",), ("rf:a", "rf:b")),), labels, groups,
        outer_test_transforms={"erase": lambda test, train, names, held: np.zeros_like(test)},
    )
    assert result.accuracy == 1
    assert result.perturbed_accuracy["erase"] == .5


def test_one_standard_error_uses_declared_reference_and_reports_boundaries() -> None:
    labels = np.tile(np.arange(2), 6)
    groups = np.repeat(np.arange(6), 2)
    perfect = np.eye(2)[labels]
    values = np.stack((perfect, perfect, perfect))
    candidates = (
        {"threshold": 0.01}, {"threshold": 0.1}, {"threshold": 1.0},
    )
    result = nested_grouped_linear_fusion(
        (CandidateFeatureBlock("rf", values, candidates, ("rf:a", "rf:b")),),
        labels, groups,
        candidate_selection_rule="one_standard_error",
        candidate_reference_by_block={"rf": {"threshold": 0.1}},
    )
    for fold in result.folds:
        assert fold.selected_candidate_indices == (1,)
        diagnostic = fold.inner_candidate_diagnostics[0]
        assert diagnostic["selection_rule"] == "one_standard_error"
        assert diagnostic["boundary_hits"] == {}
        assert diagnostic["searched_parameter_count"] == 1

    boundary_result = nested_grouped_linear_fusion(
        (CandidateFeatureBlock("rf", values, candidates, ("rf:a", "rf:b")),),
        labels, groups,
        candidate_selection_rule="one_standard_error",
        candidate_reference_by_block={"rf": {"threshold": 0.01}},
    )
    assert all(
        fold.inner_candidate_diagnostics[0]["boundary_hits"] == {"threshold": "lower"}
        for fold in boundary_result.folds
    )
