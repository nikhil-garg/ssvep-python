import numpy as np

from ssvep_toolkit.algorithms import (
    concatenate_feature_blocks,
    fit_linear_fusion,
    select_l2_grouped,
)


def test_linear_fusion_combines_named_blocks() -> None:
    labels = np.repeat(np.arange(3), 8)
    rng = np.random.default_rng(4)
    informative = np.eye(3)[labels] + rng.normal(0, 0.05, (labels.size, 3))
    nuisance = rng.normal(0, 1, (labels.size, 2))
    features, names = concatenate_feature_blocks({"rf": informative, "delta": nuisance})
    model = fit_linear_fusion(features, labels, l2=0.1, feature_names=names)
    assert np.mean(model.predict(features) == labels) == 1.0
    assert model.feature_names[:3] == ("rf:0", "rf:1", "rf:2")


def test_grouped_selection_fits_scaling_inside_each_fold() -> None:
    labels = np.tile(np.arange(2), 12)
    groups = np.repeat(np.arange(12), 2)
    features = np.column_stack((2 * labels - 1, np.linspace(-1, 1, labels.size)))
    model, fold_accuracy = select_l2_grouped(features, labels, groups, (0.01, 1.0, 100.0))
    assert fold_accuracy.shape == (3, 12)
    assert model.l2 in (0.01, 1.0, 100.0)
    assert np.mean(model.predict(features) == labels) == 1.0


def test_one_standard_error_ridge_prefers_stronger_equivalent_model() -> None:
    labels = np.tile(np.arange(2), 6)
    groups = np.repeat(np.arange(6), 2)
    features = np.eye(2)[labels]
    model, accuracy = select_l2_grouped(
        features, labels, groups, (0.01, 0.1, 1.0),
        selection_rule="one_standard_error",
    )
    assert np.all(accuracy == 1)
    assert model.l2 == 1.0
