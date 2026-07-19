import numpy as np

from ssvep_toolkit.registry import ExperimentRegistry, import_npz_checkpoints


def test_import_npz_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez_compressed(
        checkpoints / "delta_subject_01_08_classes.npz", encoder="delta",
        subject_id=1, class_count=8, accuracy=.625, selected_threshold_uV=.4,
        evaluation_design="same_subject_same_segments",
    )
    registry = ExperimentRegistry(tmp_path / "runs.sqlite3").initialize()
    result = import_npz_checkpoints(registry, checkpoints)
    assert result["imported"] == 1
    run = registry.list_runs()[0]
    assert (run.subject_id, run.class_count, run.encoder) == (1, 8, "delta")
    assert registry.summary()["metrics"] == 1
    import_npz_checkpoints(registry, checkpoints)
    assert registry.summary()["metrics"] == 1
