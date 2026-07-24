from pathlib import Path

from ssvep_toolkit.experiments import StudyRunner


def test_runner_writes_reproducibility_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "study.yaml"
    config.write_text("""schema_version: 1
study: {name: unit, validation_level: outer_test}
dataset: {subjects: [1, 2]}
task: {class_count: 4, n_blocks: 3}
optimization: {n_candidates: 5}
output: {root: runs}
""", encoding="utf-8")
    runner = StudyRunner()
    study = runner.load(config)
    result = runner.run(runner.plan(study), study)
    assert result.plan.n_cells == 6
    assert (result.run_dir / "config_requested.yaml").exists()
    assert (result.run_dir / "provenance.json").exists()
    assert runner.resume(result.run_dir).status == "planned"


def test_runner_rejects_causal_zero_phase_metadata(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("""schema_version: 1
study: {name: invalid, validation_level: outer_test}
dataset: {subjects: [1]}
preprocessing: {filter_mode: causal, zero_phase: true}
output: {root: runs}
""", encoding="utf-8")
    runner = StudyRunner()
    definition = runner.load(config)
    import pytest
    with pytest.raises(ValueError, match="zero-phase"):
        runner.run(runner.plan(definition), definition)
