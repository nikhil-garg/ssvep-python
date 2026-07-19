import numpy as np

from ssvep_toolkit.visualization.feature_space import compute_feature_embedding, standardized_features


def test_pca_embedding_is_deterministic_and_auditable() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(24, 7))
    standardized, center, scale = standardized_features(features)
    embedding, metadata = compute_feature_embedding(features, "pca")
    assert standardized.shape == features.shape
    assert center.shape == scale.shape == (7,)
    assert embedding.shape == (24, 2)
    assert metadata["method"] == "pca"
    assert 0 < np.sum(metadata["explained_variance_ratio"]) <= 1
