import pytest

from ssvep_toolkit.registry import ExperimentRegistry


def test_registry_tracks_reproducible_run_lifecycle(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.sqlite").initialize()
    study = registry.create_study("nested-fusion", {"outer": "leave-one-block-out"})
    run = registry.create_run(
        study, "s01-c08-fold00", {"alpha": 0.01}, subject_id=1,
        class_count=8, outer_fold=0, encoder="fusion", status="running",
    )
    registry.log_parameters(run, {"alpha": 0.01, "channels": ["O1", "Oz", "O2"]})
    registry.log_metric(run, "accuracy", 0.75, split="outer_test")
    registry.add_artifact(run, "checkpoint", tmp_path / "result.npz")
    registry.event("fold finished", run_id=run)
    registry.set_status(run, "complete")
    record = registry.list_runs(study)[0]
    assert record.status == "complete"
    assert record.subject_id == 1
    assert record.finished_utc is not None
    assert registry.summary() == {"studies": 1, "runs_by_status": {"complete": 1}, "metrics": 1}


def test_registry_rejects_name_reuse_with_changed_configuration(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.sqlite").initialize()
    study = registry.create_study("study", {"duration": .5})
    registry.create_run(study, "cell", {"threshold": .1})
    with pytest.raises(ValueError, match="different configuration"):
        registry.create_study("study", {"duration": 1.0})
    with pytest.raises(ValueError, match="different configuration"):
        registry.create_run(study, "cell", {"threshold": .2})
