"""Protocols shared by SSVEP dataset loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .models import DatasetMetadata, EpochBatch


class SSVEPDataset(Protocol):
    """Subject-oriented SSVEP dataset interface."""

    @property
    def metadata(self) -> DatasetMetadata: ...

    def subjects(self) -> Sequence[str]: ...

    def load_subject(self, subject_id: str) -> EpochBatch: ...


def subject_path(root: Path, subject_id: int) -> Path:
    return root / f"data_s{subject_id}_64.mat"
