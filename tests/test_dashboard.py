from ssvep_toolkit.dashboard import render_dashboard
from ssvep_toolkit.registry import ExperimentRegistry
from ssvep_toolkit.progress import ProgressJournal
import json
import numpy as np


def test_dashboard_contains_registry_run(tmp_path):
    database = tmp_path / "registry.sqlite3"
    registry = ExperimentRegistry(database).initialize()
    study = registry.create_study("nested-pilot", {"outer": "block"})
    run = registry.create_run(study, "subject-01-fold-09", {}, subject_id=1,
                              class_count=8, outer_fold=9, encoder="fusion")
    registry.log_metric(run, "accuracy", 0.75, split="outer_test")
    registry.set_status(run, "complete")
    output = render_dashboard(database, tmp_path / "dashboard.html")
    text = output.read_text(encoding="utf-8")
    assert "SSVEP Experiment Dashboard" in text
    assert "subject-01-fold-09" in text
    assert "outer_test" in text


def test_gui_module_imports_without_optional_qt():
    from ssvep_toolkit.gui import launch_gui
    assert callable(launch_gui)


def test_dashboard_indexes_neuron_example_metadata(tmp_path):
    database = tmp_path / "registry.sqlite3"
    ExperimentRegistry(database).initialize()
    examples = tmp_path / "examples"; examples.mkdir()
    image = examples / "example.png"; image.write_bytes(b"png")
    (examples / "example.json").write_text(json.dumps({
        "image": image.name, "subject": 1, "frequency_hz": 16,
        "electrode": "O1", "block": 4, "duration_ms": 1000,
    }), encoding="utf-8")
    output = render_dashboard(database, tmp_path / "dashboard.html", example_directory=examples)
    text = output.read_text(encoding="utf-8")
    assert "Signal, internal state, and spikes" in text
    assert "example.png" in text


def test_dashboard_embeds_unit_aware_trace_preview_and_validation_warning(tmp_path):
    database = tmp_path / "registry.sqlite3"
    ExperimentRegistry(database).initialize()
    examples = tmp_path / "examples"; examples.mkdir()
    image = examples / "trace.png"; image.write_bytes(b"png")
    (examples / "trace.json").write_text(json.dumps({
        "image": image.name, "subject": 2, "frequency_hz": 17, "electrode": "Oz",
        "block": 1, "duration_ms": 4, "sampling_rate_hz": 1000,
        "filter_half_width_hz": 1.0, "filter_order": 5,
    }), encoding="utf-8")
    np.savez(examples / "trace.npz", time_ms=np.arange(4), raw_uV=np.array((1., 2., 1., 0.)),
             rf_spikes=np.array((1, 3)))
    output = render_dashboard(database, tmp_path / "dashboard" / "index.html", example_directory=examples)
    text = output.read_text(encoding="utf-8")
    assert "Raw EEG" in text and "µV" in text
    assert "Nested outer test" in text
    assert "apparent_same_data" in text
    assert "rf_spikes_ms" in text
    assert "../examples/trace.png" in text


def test_dashboard_contains_all_metric_overview_and_paginated_runs(tmp_path):
    database = tmp_path / "registry.sqlite3"
    registry = ExperimentRegistry(database).initialize()
    study = registry.create_study("outer-study", {})
    run = registry.create_run(study, "subject_01_04_classes_causal.npz", {}, subject_id=1,
                              class_count=4, encoder="fusion", status="complete")
    registry.log_metric(run, "accuracy", 0.75, split="outer_test")
    registry.log_metric(run, "practical_itr_bits_per_minute", 20.0, split="outer_test")
    registry.log_metric(run, "boundary_rate:resonate_fire.alpha", 0.25, split="inner_validation")
    output = render_dashboard(database, tmp_path / "dashboard.html")
    text = output.read_text(encoding="utf-8")
    assert "All selected metrics at a glance" in text
    assert "Practical ITR" in text
    assert "Inner validation" in text
    assert "Boundary selection rate · resonate fire · alpha" in text
    assert "Page ${page+1}/${pages}" in text


def test_dashboard_surfaces_structured_phase_and_eta(tmp_path):
    database = tmp_path / "registry.sqlite3"; ExperimentRegistry(database).initialize()
    experiments = tmp_path / "experiments"; study = experiments / "transparent-study"; study.mkdir(parents=True)
    journal = ProgressJournal(study / "progress.jsonl", config={"x": 1}, study="transparent-study")
    journal.write("inner_fusion", 2, 10, "Selecting candidates")
    output = render_dashboard(database, tmp_path / "dashboard.html", experiment_directory=experiments)
    text = output.read_text(encoding="utf-8")
    assert "Current phase" in text
    assert "inner_fusion" in text
    assert "Selecting candidates" in text


def test_dashboard_renders_cell_and_study_progress_bars(tmp_path):
    database = tmp_path / "registry.sqlite3"; ExperimentRegistry(database).initialize()
    experiments = tmp_path / "experiments"; study = experiments / "advanced"; study.mkdir(parents=True)
    (study / "progress.jsonl").write_text(json.dumps({
        "status": "running", "phase": "advanced_nested", "current": 0.25, "total": 30,
        "cell_current": 27, "cell_total": 108, "message": "Selecting candidates",
    }) + "\n", encoding="utf-8")
    output = render_dashboard(database, tmp_path / "dashboard.html", experiment_directory=experiments)
    text = output.read_text(encoding="utf-8")
    assert "Live and pending experiments" in text
    assert "current cell ${x.cell_current}/${x.cell_total}" in text
    assert "progressbar" in text
