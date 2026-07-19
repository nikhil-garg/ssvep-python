import numpy as np

from ssvep_toolkit.evaluation import (
    CandidateFeatureBlock,
    FoldSafeCandidateFeatureBlock,
    JointSearchConfig,
    RacingConfig,
    advanced_nested_grouped_fusion,
    endpoint_results,
    pareto_endpoints,
    pooled_reference,
)


def test_joint_search_finds_complementary_encoders_and_retrains_ablation() -> None:
    rng = np.random.default_rng(9)
    labels = np.tile(np.arange(4), 8); groups = np.repeat(np.arange(8), 4)
    # Each informative encoder resolves a different bit of the class label.
    first = (labels // 2)[:, None] + rng.normal(0, .02, (labels.size, 1))
    second = (labels % 2)[:, None] + rng.normal(0, .02, (labels.size, 1))
    noise = rng.normal(size=(labels.size, 1))
    rf = CandidateFeatureBlock("rf", np.stack((first, noise)), ("bit-a", "noise"), ("rf:a",))
    delta = CandidateFeatureBlock("delta", np.stack((second, noise)), ("bit-b", "noise"), ("delta:b",))
    result = advanced_nested_grouped_fusion(
        (rf, delta), labels, groups, l2_grid=(.01, 1, 100),
        racing=RacingConfig(enabled=False),
        joint=JointSearchConfig(screened_per_encoder=2, beam_width=4),
        retrained_ablation_blocks=("rf",),
        retrained_feature_ablations={"remove-rf-features": lambda name: name.startswith("rf:")},
    )
    assert result.accuracy == 1
    assert result.retrained_ablation_accuracy["rf"] < result.accuracy
    assert result.retrained_ablation_accuracy["remove-rf-features"] == result.retrained_ablation_accuracy["rf"]
    assert all(fold.selected_candidate_indices == (0, 0) for fold in result.folds)


def test_fold_safe_builder_receives_outer_training_mask_only() -> None:
    labels = np.tile(np.arange(2), 6); groups = np.repeat(np.arange(6), 2)
    seen = []
    values = np.eye(2)[labels][None, ...]
    def builder(mask):
        seen.append(np.asarray(mask).copy())
        return CandidateFeatureBlock("rf", values, ("only",), ("rf:0", "rf:1"))
    lazy = FoldSafeCandidateFeatureBlock("rf", ("only",), ("rf:0", "rf:1"), builder)
    result = advanced_nested_grouped_fusion(
        (lazy,), labels, groups, l2_grid=(1,), racing=RacingConfig(enabled=False),
        joint=JointSearchConfig(screened_per_encoder=1, beam_width=1),
    )
    assert result.accuracy == 1
    assert len(seen) == 6
    assert all(mask.sum() == 10 for mask in seen)


def test_pooling_and_endpoint_pareto_helpers() -> None:
    pooled = pooled_reference({"threshold": 4.0, "fixed_gain": 3.0}, {"threshold": 1.0, "tau": .02}, .5)
    assert np.isclose(pooled["threshold"], 2.0)
    assert pooled["fixed_gain"] == 3.0 and pooled["tau"] == .02
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([[0, 0, 0, 1], [0, 1, 0, 1]])
    summaries = endpoint_results(labels, predictions, (.2, .5), np.array([[1, 1, 1, 1], [5, 5, 5, 5]]), classes=2)
    assert summaries[1].accuracy == 1
    assert set(pareto_endpoints(summaries)) == set(summaries)


def test_spike_cost_tie_break_excludes_outer_test_trials() -> None:
    labels = np.tile(np.arange(2), 6); groups = np.repeat(np.arange(6), 2)
    values = np.eye(2)[labels][None, ...]
    costs = np.ones((1, labels.size)); costs[:, groups == 0] = 1000
    block = CandidateFeatureBlock("rf", values, ("only",), ("rf:0", "rf:1"), costs)
    result = advanced_nested_grouped_fusion(
        (block,), labels, groups, l2_grid=(1,), racing=RacingConfig(enabled=False),
        joint=JointSearchConfig(screened_per_encoder=1, beam_width=1, spike_cost_weight=1),
    )
    fold_zero = next(fold for fold in result.folds if fold.held_out_group == 0)
    assert fold_zero.inner_metrics.spike_cost == 1
