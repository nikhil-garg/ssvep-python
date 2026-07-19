from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .matlab import Matlab73Dataset


SUBJECT_RE = re.compile(r"data_s(\d+)_64\.mat$", re.IGNORECASE)


@dataclass(frozen=True)
class SubjectFile:
    subject: int
    path: Path
    bytes: int
    logical_shape: tuple[int, ...] | None = None
    dtype: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DatasetInventory:
    root: Path
    subjects: tuple[SubjectFile, ...]
    metadata_files: tuple[Path, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.subjects)


def inspect_dataset(root: str | Path, inspect_hdf5: bool = True) -> DatasetInventory:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    subjects: list[SubjectFile] = []
    for path in root.glob("data_s*_64.mat"):
        match = SUBJECT_RE.match(path.name)
        if not match:
            continue
        subject = int(match.group(1))
        shape = None
        dtype = None
        error = None
        if inspect_hdf5:
            try:
                with Matlab73Dataset(path) as source:
                    shape, dtype = source.logical_shape, source.dtype
            except Exception as exc:  # inspection must report, not abort
                error = f"{type(exc).__name__}: {exc}"
        subjects.append(SubjectFile(subject, path, path.stat().st_size, shape, dtype, error))
    subjects.sort(key=lambda item: item.subject)
    metadata_names = (
        "Participants_information.csv",
        "Electrode_channels_information.csv",
        "Sub_score.csv",
        "Sub_score.mat",
    )
    metadata = tuple(root / name for name in metadata_names if (root / name).exists())
    return DatasetInventory(root, tuple(subjects), metadata)

