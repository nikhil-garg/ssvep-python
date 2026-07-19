from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ssvep_toolkit.config import AppConfig
from ssvep_toolkit.data.matlab import Matlab73Dataset
from ssvep_toolkit.preprocessing.downsampling import downsample, output_sample_count


@dataclass(frozen=True)
class PreprocessingPlan:
    subjects: int
    trials: int
    channels: int
    input_samples: int
    output_samples: int
    estimated_output_bytes: int


def describe_preprocessing(config: AppConfig) -> PreprocessingPlan:
    channels = len(config.channel_indices())
    trials = (
        len(config.dataset.subjects)
        * len(config.dataset.conditions)
        * len(config.dataset.frequencies)
        * 12
    )
    samples = 5140
    ds = config.preprocessing.downsampling
    output_samples = (
        output_sample_count(samples, ds.original_rate_hz, ds.target_rate_hz, ds.method)
        if ds.enabled
        else samples
    )
    effective_rate = ds.target_rate_hz if ds.enabled else ds.original_rate_hz
    if config.preprocessing.latency.enabled:
        output_samples -= round(config.preprocessing.latency.seconds * effective_rate)
    return PreprocessingPlan(
        subjects=len(config.dataset.subjects),
        trials=trials,
        channels=channels,
        input_samples=samples,
        output_samples=output_samples,
        estimated_output_bytes=trials * channels * output_samples * 4,
    )


def _open_output(path: Path, config: AppConfig, samples: int):
    import h5py
    import numpy as np

    mode = "w" if config.execution.overwrite else "a"
    handle = h5py.File(path, mode)
    shape = (
        len(config.dataset.conditions),
        len(config.channel_indices()),
        samples,
        len(config.dataset.frequencies),
        12,
    )
    if "data" not in handle:
        handle.create_dataset(
            "data", shape=shape, dtype="float32", chunks=(1, len(config.channel_indices()), samples, 1, 1)
        )
        handle.create_dataset("completed", shape=(shape[0], shape[3], shape[4]), dtype="bool")
    elif tuple(handle["data"].shape) != shape:
        handle.close()
        raise ValueError(f"existing output has incompatible shape: {path}")
    handle.attrs["logical_axes"] = "condition,channel,sample,frequency,block"
    handle.attrs["conditions"] = np.asarray(config.dataset.conditions)
    handle.attrs["frequencies_hz"] = np.asarray(config.dataset.frequencies)
    handle.attrs["channels_matlab"] = np.asarray([x + 1 for x in config.channel_indices()])
    handle.attrs["configuration_json"] = json.dumps(config.to_dict(), sort_keys=True)
    return handle


def run_preprocessing(config: AppConfig, progress: Callable[[str], None] = print) -> list[Path]:
    import numpy as np

    config.output.root.mkdir(parents=True, exist_ok=True)
    ds = config.preprocessing.downsampling
    samples = (
        output_sample_count(5140, ds.original_rate_hz, ds.target_rate_hz, ds.method)
        if ds.enabled
        else 5140
    )
    effective_rate = ds.target_rate_hz if ds.enabled else ds.original_rate_hz
    latency_samples = (
        round(config.preprocessing.latency.seconds * effective_rate)
        if config.preprocessing.latency.enabled
        else 0
    )
    samples -= latency_samples
    if samples <= 0:
        raise ValueError("latency correction removes all samples")
    outputs: list[Path] = []
    for subject in config.dataset.subjects:
        source_path = config.dataset.root / config.dataset.subject_pattern.format(subject=subject)
        if not source_path.exists():
            raise FileNotFoundError(f"missing subject file: {source_path}")
        output_path = config.output.root / f"subject_{subject:02d}_preprocessed.h5"
        progress(f"Subject {subject}: {source_path.name}")
        with Matlab73Dataset(source_path) as source, _open_output(output_path, config, samples) as target:
            target.attrs["source_path"] = str(source_path.resolve())
            target.attrs["source_bytes"] = source_path.stat().st_size
            target.attrs["source_logical_shape"] = source.logical_shape
            target.attrs["sampling_rate_hz"] = ds.target_rate_hz if ds.enabled else ds.original_rate_hz
            optimized = ds.enabled and ds.method == "matlab_compatible" and source.storage_chunks is not None
            if optimized and not bool(target["completed"][...].all()):
                _preprocess_channel_chunks(source, target, config, latency_samples, progress)
                target["completed"][...] = True
                target.flush()
                progress("  all selected channel chunks complete")
                outputs.append(output_path)
                continue
            for ci, condition in enumerate(config.dataset.conditions):
                for fi, frequency in enumerate(config.dataset.frequencies):
                    for bi in range(12):
                        if config.execution.resume and bool(target["completed"][ci, fi, bi]):
                            continue
                        trial = source.read_trial(condition, frequency, bi + 1)
                        trial = trial[config.channel_indices(), :]
                        if ds.enabled:
                            trial = downsample(trial, ds.original_rate_hz, ds.target_rate_hz, ds.method)
                        if latency_samples:
                            trial = trial[..., latency_samples:]
                        if trial.shape[-1] != samples:
                            raise ValueError(f"unexpected output sample count {trial.shape[-1]}")
                        target["data"][ci, :, :, fi, bi] = np.asarray(trial, dtype=np.float32)
                        target["completed"][ci, fi, bi] = True
                        target.flush()
                    progress(f"  condition={condition}, frequency={frequency} Hz complete")
        outputs.append(output_path)
    return outputs


def _preprocess_channel_chunks(source, target, config: AppConfig, latency_samples: int, progress) -> None:
    """Process each physical HDF5 channel chunk only once."""
    import numpy as np

    requested = config.channel_indices()
    chunk_width = source.storage_chunks[3]
    condition_indices = np.asarray(config.dataset.conditions, dtype=int) - 1
    frequency_indices = np.asarray(config.dataset.frequencies, dtype=int) - 1
    factor = config.preprocessing.downsampling.original_rate_hz // config.preprocessing.downsampling.target_rate_hz
    groups: dict[int, list[tuple[int, int]]] = {}
    for output_index, logical_channel in enumerate(requested):
        start = (logical_channel // chunk_width) * chunk_width
        groups.setdefault(start, []).append((output_index, logical_channel - start))
    for start, mappings in sorted(groups.items()):
        stop = min(start + chunk_width, source.logical_shape[1])
        progress(f"  reading source channels {start + 1}-{stop}")
        raw = source.read_channel_chunk(start, stop)
        raw = np.take(raw, condition_indices, axis=0)
        raw = np.take(raw, frequency_indices, axis=3)
        local = [item[1] for item in mappings]
        output_positions = [item[0] for item in mappings]
        processed = raw[:, local, ::factor, :, :]
        if latency_samples:
            processed = processed[:, :, latency_samples:, :, :]
        target["data"][:, output_positions, :, :, :] = np.asarray(processed, dtype=np.float32)
        target.flush()
