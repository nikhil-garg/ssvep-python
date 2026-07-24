"""Configuration and result models for reproducible studies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ValidationLevel = Literal["outer_test", "inner_validation", "apparent_same_data"]


@dataclass(frozen=True)
class StudyDefinition:
    """Resolved study configuration independent of a GUI or CLI."""

    schema_version: int
    name: str
    validation_level: ValidationLevel
    raw: dict[str, Any]
    source_path: Path
    output_dir: Path


@dataclass(frozen=True)
class RunPlan:
    run_dir: Path
    n_subjects: int
    n_classes: int
    n_blocks: int
    n_candidate_settings: int
    validation_level: ValidationLevel
    warnings: tuple[str, ...]

    @property
    def n_cells(self) -> int:
        return self.n_subjects * self.n_blocks


@dataclass(frozen=True)
class StudyResult:
    run_dir: Path
    status: str
    plan: RunPlan
