"""Loader for the 1–60 Hz MATLAB 7.3 SSVEP dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ssvep_toolkit.data.matlab import EXPECTED_LOGICAL_SHAPE, Matlab73Dataset

from .base import subject_path
from .models import DatasetMetadata, EpochBatch


POSTERIOR_CHANNELS = ("O1", "Oz", "O2")
POSTERIOR_INDICES = (60, 61, 62)


class OneToSixtyHzDataset:
    """Current dataset conventions, including MATLAB storage axes and onset."""

    def __init__(self, root: str | Path, *, condition: int = 2, onset_sample: int = 140) -> None:
        self.root = Path(root)
        self.condition = condition
        self.onset_sample = onset_sample

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata("one_to_sixty_hz", 1000.0, POSTERIOR_CHANNELS, "uV assumed", 2, 12)

    def subjects(self) -> tuple[str, ...]:
        return tuple(str(index) for index in range(1, 31) if subject_path(self.root, index).exists())

    def load_subject(self, subject_id: str) -> EpochBatch:
        numeric_id = int(subject_id)
        with Matlab73Dataset(subject_path(self.root, numeric_id)) as source:
            if source.logical_shape != EXPECTED_LOGICAL_SHAPE:
                raise ValueError(f"unexpected logical shape: {source.logical_shape}")
            values = source.read_channel_chunk(POSTERIOR_INDICES[0], POSTERIOR_INDICES[-1] + 1)
        selected = values[self.condition - 1]
        selected = selected[:, :, :, :]
        # channel, sample, frequency, block -> trial, channel, sample
        data = selected.transpose(2, 3, 0, 1).reshape(60 * 12, 3, selected.shape[1]).astype(np.float64)
        labels = np.repeat(np.arange(60, dtype=int), 12)
        groups = np.tile(np.arange(12, dtype=int), 60)
        return EpochBatch(data, labels, groups, 1000.0, POSTERIOR_CHANNELS, tuple(range(1, 61)), self.onset_sample)
