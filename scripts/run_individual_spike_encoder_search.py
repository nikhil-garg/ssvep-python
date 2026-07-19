"""Resumable subject-wise delta and LIF parameter searches."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import os
import time

import numpy as np
import yaml

from ssvep_toolkit.data.matlab import Matlab73Dataset
from ssvep_toolkit.evaluation.spike_encoder_experiment import (
    apparent_template_result,
    delta_count_features,
    lif_count_features,
)
from ssvep_toolkit.preprocessing import BandpassParameters, target_frequency_filter_bank


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent
CONFIG_PATH = Path(os.environ.get("SPIKE_ENCODER_CONFIG", ROOT / "configs/individual_spike_encoders.yaml"))
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output"]["root"]
CP = OUT / "checkpoints"
CP.mkdir(parents=True, exist_ok=True)
ALL_COUNTS = np.array((2, 4, 8, 16, 32))
CLASS_SETS = [np.rint(np.linspace(8, 39, count)).astype(int) for count in ALL_COUNTS]
SUBJECTS = [int(x) for x in os.environ.get("SPIKE_ENCODER_SUBJECTS", ",".join(map(str, CONFIG["experiment"]["subjects"]))).split(",")]
COUNTS = [int(x) for x in os.environ.get("SPIKE_ENCODER_CLASS_COUNTS", ",".join(map(str, CONFIG["experiment"]["class_counts"]))).split(",")]
ENCODERS = os.environ.get("SPIKE_ENCODERS", "delta,lif").split(",")
BP = BandpassParameters(**CONFIG["bandpass"])
CROP_START = 140
DECISION_SAMPLES = round(float(CONFIG["experiment"]["duration_seconds"]) * 1000)
SOURCE_EPOCH_SAMPLES = 5140


def load_subject(subject_id: int) -> np.ndarray:
    with Matlab73Dataset(DATA / f"data_s{subject_id}_64.mat") as source:
        chunk = source.read_channel_chunk(60, 64)
    # Retain the full 5.14 s source epoch. Narrow-band filtering a pre-cropped
    # 1 s window causes edge transients; crop only after filtering.
    return chunk[1, :3, :, :, :].transpose(2, 3, 0, 1).astype(np.float32)


def atomic_save(path: Path, **payload: object) -> None:
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def run_delta(subject_id: int, count: int, frequencies: np.ndarray, bands: np.ndarray, labels: np.ndarray) -> None:
    path = CP / f"delta_subject_{subject_id:02d}_{count:02d}_classes.npz"
    if path.exists():
        print(f"delta S{subject_id:02d} C{count:02d} resumed", flush=True)
        return
    started = time.perf_counter()
    derivative = np.diff(bands.astype(float), axis=-1)
    reference = float(np.median(np.sqrt(np.mean(derivative ** 2, axis=-1))))
    grid = np.asarray(list(product(CONFIG["delta"]["threshold_scale"], CONFIG["delta"]["asymmetry"])), float)
    accuracy = np.empty(len(grid), np.float32)
    for index, (scale, asymmetry) in enumerate(grid):
        features = delta_count_features(bands, reference * scale, asymmetry)
        accuracy[index] = apparent_template_result(features, labels)[0]
    best = int(np.argmax(accuracy)); scale, asymmetry = grid[best]
    threshold = reference * scale
    features = delta_count_features(bands, threshold, asymmetry)
    score, prediction, scores = apparent_template_result(features, labels)
    atomic_save(path, encoder="delta", subject_id=subject_id, class_count=count, frequencies_hz=frequencies,
                grid=grid, grid_accuracy=accuracy, selected_index=best, selected_threshold_scale=scale,
                selected_threshold_uV=threshold, selected_asymmetry=asymmetry, reference_derivative_rms_uV=reference,
                features=features, scores=scores, prediction=prediction, accuracy=score,
                elapsed_seconds=time.perf_counter()-started, filter_then_crop=True,
                source_epoch_samples=SOURCE_EPOCH_SAMPLES, decision_samples=DECISION_SAMPLES,
                evaluation_design=CONFIG["experiment"]["evaluation"])
    print(f"delta S{subject_id:02d} C{count:02d} {100*score:.1f}% threshold={threshold:.4g}uV asym={asymmetry:g}", flush=True)


def run_lif(subject_id: int, count: int, frequencies: np.ndarray, bands: np.ndarray, labels: np.ndarray) -> None:
    path = CP / f"lif_subject_{subject_id:02d}_{count:02d}_classes.npz"
    if path.exists():
        print(f"lif S{subject_id:02d} C{count:02d} resumed", flush=True)
        return
    started = time.perf_counter()
    reference = float(np.median(np.sqrt(np.mean(bands.astype(float) ** 2, axis=-1))))
    grid = np.asarray(list(product(CONFIG["lif"]["threshold_scale"], CONFIG["lif"]["tau_seconds"])), float)
    gain = float(CONFIG["lif"]["input_gain"]); accuracy = np.empty(len(grid), np.float32)
    for index, (scale, tau) in enumerate(grid):
        features = lif_count_features(bands, 1000, reference * scale, tau, input_gain=gain)
        accuracy[index] = apparent_template_result(features, labels)[0]
    best = int(np.argmax(accuracy)); scale, tau = grid[best]; threshold = reference * scale
    features = lif_count_features(bands, 1000, threshold, tau, input_gain=gain)
    score, prediction, scores = apparent_template_result(features, labels)
    atomic_save(path, encoder="lif", subject_id=subject_id, class_count=count, frequencies_hz=frequencies,
                grid=grid, grid_accuracy=accuracy, selected_index=best, selected_threshold_scale=scale,
                selected_threshold_uV=threshold, selected_tau_seconds=tau, input_gain=gain,
                reference_band_rms_uV=reference, features=features, scores=scores, prediction=prediction,
                accuracy=score, elapsed_seconds=time.perf_counter()-started,
                filter_then_crop=True, source_epoch_samples=SOURCE_EPOCH_SAMPLES, decision_samples=DECISION_SAMPLES,
                evaluation_design=CONFIG["experiment"]["evaluation"])
    print(f"lif S{subject_id:02d} C{count:02d} {100*score:.1f}% threshold={threshold:.4g}uV tau={tau:g}s", flush=True)


for subject_id in SUBJECTS:
    print(f"Loading subject {subject_id}/30", flush=True)
    subject = load_subject(subject_id)
    for count in COUNTS:
        frequencies = CLASS_SETS[np.flatnonzero(ALL_COUNTS == count)[0]]
        raw = subject[frequencies - 1].reshape(-1, 3, subject.shape[-1])
        labels = np.repeat(np.arange(count), 12)
        full_bands = target_frequency_filter_bank(raw, frequencies, 1000, BP)
        bands = full_bands[..., CROP_START:CROP_START + DECISION_SAMPLES].astype(np.float32)
        if "delta" in ENCODERS:
            run_delta(subject_id, count, frequencies, bands, labels)
        if "lif" in ENCODERS:
            run_lif(subject_id, count, frequencies, bands, labels)
print("INDIVIDUAL SPIKE ENCODER SEARCH COMPLETE", flush=True)
