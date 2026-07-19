from __future__ import annotations

from pathlib import Path
from typing import Any


MATLAB_LOGICAL_AXES = ("condition", "channel", "sample", "frequency", "block")
H5PY_STORAGE_AXES = tuple(reversed(MATLAB_LOGICAL_AXES))
EXPECTED_LOGICAL_SHAPE = (2, 64, 5140, 60, 12)


class Matlab73Dataset:
    """Axis-safe, lazy reader for one subject's MATLAB 7.3 EEG file."""

    def __init__(self, path: str | Path, variable: str = "datas") -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required to read MATLAB 7.3 files") from exc
        self.path = Path(path)
        self._file: Any = h5py.File(self.path, "r")
        if variable not in self._file:
            self._file.close()
            raise KeyError(f"{self.path.name} has no variable {variable!r}")
        self._dataset = self._file[variable]
        self.storage_shape = tuple(int(x) for x in self._dataset.shape)
        self.logical_shape = tuple(reversed(self.storage_shape))
        if len(self.logical_shape) != 5:
            self.close()
            raise ValueError(f"expected a 5-D EEG array, found {self.logical_shape}")

    @property
    def dtype(self) -> str:
        return str(self._dataset.dtype)

    @property
    def matches_expected_shape(self) -> bool:
        return self.logical_shape == EXPECTED_LOGICAL_SHAPE

    def read_trial(self, condition: int, frequency: int, block: int) -> Any:
        """Return all channels for one trial as `(channel, sample)`.

        Public selectors are one-based to match the paper and MATLAB scripts.
        """
        import numpy as np

        if not 1 <= condition <= self.logical_shape[0]:
            raise IndexError("condition is outside the dataset")
        if not 1 <= frequency <= self.logical_shape[3]:
            raise IndexError("frequency is outside the dataset")
        if not 1 <= block <= self.logical_shape[4]:
            raise IndexError("block is outside the dataset")
        stored = self._dataset[block - 1, frequency - 1, :, :, condition - 1]
        # Stored scalar selection leaves `(sample, channel)`.
        return np.asarray(stored).T

    @property
    def storage_chunks(self) -> tuple[int, ...] | None:
        chunks = self._dataset.chunks
        return tuple(int(x) for x in chunks) if chunks else None

    def read_channel_chunk(self, start: int, stop: int) -> Any:
        """Read zero-based logical channels as all other axes.

        Returns logical `(condition, channel, sample, frequency, block)` order.
        This aligns reads with the unusual source HDF5 chunks and is much faster
        than repeatedly reading individual trials.
        """
        import numpy as np

        if not 0 <= start < stop <= self.logical_shape[1]:
            raise IndexError("invalid channel slice")
        stored = np.asarray(self._dataset[:, :, :, start:stop, :])
        return stored.transpose(4, 3, 2, 1, 0)

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "Matlab73Dataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
