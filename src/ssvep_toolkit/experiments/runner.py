"""Generic planning, artifact creation, and resumption for YAML studies."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import yaml

from ssvep_toolkit.progress import ProgressJournal, latest_progress

from .models import RunPlan, StudyDefinition, StudyResult
from .provenance import collect_provenance


class StudyRunner:
    """Create reproducible run directories before study-specific execution."""

    def load(self, config_path: str | Path) -> StudyDefinition:
        source_path = Path(config_path).resolve()
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != 1:
            raise ValueError("schema_version: 1 is required")
        study = raw.get("study", {})
        level = study.get("validation_level")
        if level not in {"outer_test", "inner_validation", "apparent_same_data"}:
            raise ValueError("study.validation_level must be outer_test, inner_validation, or apparent_same_data")
        output = raw.get("output", {})
        root = Path(output.get("root", "outputs/experiments"))
        project_root = source_path.parents[2] if source_path.parent.name == "studies" else source_path.parent
        output_dir = root if root.is_absolute() else (project_root / root).resolve()
        return StudyDefinition(1, str(study["name"]), level, raw, source_path, output_dir)

    def plan(self, study: StudyDefinition) -> RunPlan:
        raw = study.raw
        task = raw.get("task", {})
        subjects = raw.get("dataset", {}).get("subjects", raw.get("study", {}).get("subjects", [1]))
        n_classes = int(task.get("class_count", raw.get("study", {}).get("class_count", 1)))
        n_blocks = int(task.get("n_blocks", 12))
        candidate_count = int(raw.get("optimization", {}).get("n_candidates", 1))
        warnings = []
        if study.validation_level == "apparent_same_data":
            warnings.append("Parameter selection and scoring use the same data.")
        if raw.get("preprocessing", {}).get("filter_mode") == "offline":
            warnings.append("Offline filtering uses future samples.")
        return RunPlan(study.output_dir / study.name, len(subjects), n_classes, n_blocks, candidate_count,
                       study.validation_level, tuple(warnings))

    def run(self, plan: RunPlan, study: StudyDefinition) -> StudyResult:
        run_dir = plan.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        requested_path = run_dir / "config_requested.yaml"
        if not requested_path.exists():
            requested_path.write_text(study.source_path.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(study.raw, sort_keys=False), encoding="utf-8")
        (run_dir / "run_plan.json").write_text(json.dumps(asdict(plan), default=str, indent=2), encoding="utf-8")
        preprocessing = study.raw.get("preprocessing", {})
        filter_mode = preprocessing.get("filter_mode")
        provenance = collect_provenance(
            study.raw, validation_level=study.validation_level, filter_mode=filter_mode,
            dataset_root=study.raw.get("dataset", {}).get("root"),
        )
        (run_dir / "provenance.json").write_text(json.dumps(provenance.to_dict(), indent=2), encoding="utf-8")
        journal_path = run_dir / "progress.jsonl"
        if not journal_path.exists():
            ProgressJournal(journal_path, config=study.raw, study=study.name).write(
                "planned", 0, plan.n_cells, "Run directory created", status="planned"
            )
        return StudyResult(run_dir, "planned", plan)

    def resume(self, run_dir: str | Path) -> StudyResult:
        target = Path(run_dir)
        plan = RunPlan(**json.loads((target / "run_plan.json").read_text(encoding="utf-8")))
        event = latest_progress(target / "progress.jsonl")
        return StudyResult(target, str(event.get("status", "unknown")) if event else "unknown", plan)
