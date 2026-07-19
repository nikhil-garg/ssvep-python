"""Subject-wise fusion of monopolar and Oz-referenced R&F spike patterns.

The experiment deliberately reports apparent accuracy: parameter selection,
template calibration, fusion-weight selection, and scoring use the same twelve
segments per class. Every subject/class-count cell is resumable.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import os
import time

import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import (
    OscillatorBankClassifier,
    ResonateAndFireParameters,
)
from ssvep_toolkit.data.matlab import Matlab73Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent
BASE = ROOT / "outputs/experiments/resonate_and_fire_deep_gain_search"
OUT = ROOT / "outputs/experiments/resonate_and_fire_fused_reference_search"
CP = OUT / "checkpoints"
CP.mkdir(parents=True, exist_ok=True)

ALL_COUNTS = np.array((2, 4, 8, 16, 32))
CLASS_SETS = [np.rint(np.linspace(8, 39, count)).astype(int) for count in ALL_COUNTS]
SUBJECTS = np.asarray([int(x) for x in os.environ.get("RF_SUBJECTS", ",".join(map(str, range(1, 31)))).split(",")])
COUNTS = np.asarray([int(x) for x in os.environ.get("RF_CLASS_COUNTS", "2,4,8,16,32").split(",")])

# Broader on damping than the earlier search because spatial subtraction changes
# both the RMS distribution and the effective narrow-band signal-to-noise ratio.
ALPHAS = np.array((0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4))
THRESHOLDS = np.array((0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
OPERATING_RMS = np.array((0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0))
BIPOLAR_GRID = np.asarray(list(product(ALPHAS, THRESHOLDS, OPERATING_RMS)), dtype=float)
CHANNEL_NAMES = np.array(("O1", "Oz", "O2", "O1-Oz", "O2-Oz"))


def load_subject(subject_id: int) -> np.ndarray:
    with Matlab73Dataset(DATA / f"data_s{subject_id}_64.mat") as source:
        chunk = source.read_channel_chunk(60, 64)
    return chunk[1, :3, 140:1140, :, :].transpose(2, 3, 0, 1).astype(np.float32)


def adapt(raw: np.ndarray, operating_rms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = raw - raw.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered.astype(float) ** 2, axis=-1))
    gain = operating_rms / np.maximum(rms, 1e-6)
    return centered * gain[..., None], rms, gain


def parameters(alpha: float, threshold: float) -> ResonateAndFireParameters:
    return ResonateAndFireParameters(
        damping_alpha=float(alpha), threshold=float(threshold), input_gain=0.8,
        integration_substeps=4, refractory_cycles=0.5, solver="exact",
        reset_mode="zero", spike_detection="upward_crossing",
    )


def channel_template_scores(
    signal: np.ndarray,
    labels: np.ndarray,
    frequencies: np.ndarray,
    alpha: float,
    threshold: float,
    harmonics: tuple[int, ...],
    spread: tuple[float, ...],
) -> np.ndarray:
    model = OscillatorBankClassifier(
        frequencies, 1000, parameters(alpha, threshold), harmonics=harmonics,
        harmonic_weights=tuple(1 / harmonic for harmonic in harmonics), spread_hz=spread,
    )
    model.channel_scale_ = np.ones(1)
    values = signal.reshape(-1, 1, 1000)
    model.fit_calibration(values, labels, (1000,))
    return model.decision_scores(values, (1000,))[0]


def bipolar_candidate_accuracy(
    raw_bipolar: np.ndarray,
    labels: np.ndarray,
    frequencies: np.ndarray,
    alpha: float,
    threshold: float,
    operating_rms: float,
    harmonics: tuple[int, ...],
    spread: tuple[float, ...],
) -> float:
    adapted, _, _ = adapt(raw_bipolar, operating_rms)
    channel_scores = np.stack([
        channel_template_scores(adapted[:, :, channel], labels, frequencies, alpha, threshold, harmonics, spread)
        for channel in range(2)
    ])
    prediction = np.argmax(channel_scores.mean(axis=0), axis=1)
    return float(np.mean(prediction == labels))


def weight_candidates(seed: int) -> np.ndarray:
    fixed = np.vstack((
        np.eye(5),
        np.array((1, 1, 1, 0, 0), float) / 3,
        np.array((0, 0, 0, 1, 1), float) / 2,
        np.ones(5) / 5,
        np.array((1, 1, 1, 0.5, 0.5), float) / 4,
    ))
    random = np.random.default_rng(seed).dirichlet(np.ones(5), size=1024)
    return np.vstack((fixed, random))


def optimize_weights(scores: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    candidates = weight_candidates(seed)
    fused = np.einsum("wc,ctk->wtk", candidates, scores, optimize=True)
    predictions = np.argmax(fused, axis=2)
    accuracy = np.mean(predictions == labels[None, :], axis=1)
    best_accuracy = accuracy.max()
    tied = np.flatnonzero(accuracy == best_accuracy)
    # Among accuracy ties, prefer the least concentrated solution to avoid an
    # arbitrary one-hot channel selection caused by discrete accuracy.
    entropy = -np.sum(candidates[tied] * np.log(np.maximum(candidates[tied], 1e-12)), axis=1)
    best = tied[np.argmax(entropy)]
    return candidates[best], float(accuracy[best]), fused[best], accuracy


def atomic_save(path: Path, **payload: object) -> None:
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def run_cell(subject_id: int, class_count: int, subject: np.ndarray) -> None:
    path = CP / f"subject_{subject_id:02d}_{class_count:02d}_classes.npz"
    if path.exists():
        print(f"S{subject_id:02d} C{class_count:02d} resumed", flush=True)
        return
    frequencies = CLASS_SETS[np.flatnonzero(ALL_COUNTS == class_count)[0]]
    raw_mono = subject[frequencies - 1]
    raw_bipolar = np.stack((raw_mono[:, :, 0] - raw_mono[:, :, 1], raw_mono[:, :, 2] - raw_mono[:, :, 1]), axis=2)
    labels = np.repeat(np.arange(class_count), raw_mono.shape[1])
    started = time.perf_counter()

    with np.load(BASE / "checkpoints" / f"subject_{subject_id:02d}_{class_count:02d}_classes.npz") as base:
        mono_alpha = float(base["selected_alpha"])
        mono_threshold = float(base["selected_threshold"])
        mono_operating = float(base["selected_operating_rms"])
        harmonics = tuple(map(int, base["selected_harmonics"]))
        spread = tuple(map(float, base["selected_spread_hz"]))
        baseline_accuracy = float(base["accuracy"])

    grid_accuracy = np.empty(len(BIPOLAR_GRID), dtype=np.float32)
    for index, (alpha, threshold, operating) in enumerate(BIPOLAR_GRID):
        grid_accuracy[index] = bipolar_candidate_accuracy(
            raw_bipolar, labels, frequencies, alpha, threshold, operating, harmonics, spread,
        )
        if (index + 1) % 72 == 0:
            print(f"S{subject_id:02d} C{class_count:02d} bipolar {index + 1}/{len(BIPOLAR_GRID)}", flush=True)
    best_index = int(np.argmax(grid_accuracy))
    bipolar_alpha, bipolar_threshold, bipolar_operating = BIPOLAR_GRID[best_index]

    adapted_mono, mono_rms, mono_gain = adapt(raw_mono, mono_operating)
    adapted_bipolar, bipolar_rms, bipolar_gain = adapt(raw_bipolar, bipolar_operating)
    channel_scores = []
    for channel in range(3):
        channel_scores.append(channel_template_scores(
            adapted_mono[:, :, channel], labels, frequencies, mono_alpha, mono_threshold, harmonics, spread,
        ))
    for channel in range(2):
        channel_scores.append(channel_template_scores(
            adapted_bipolar[:, :, channel], labels, frequencies, bipolar_alpha, bipolar_threshold, harmonics, spread,
        ))
    channel_scores = np.asarray(channel_scores)

    weights, fused_accuracy, fused_scores, weight_accuracy = optimize_weights(
        channel_scores, labels, seed=1000 * subject_id + class_count,
    )
    prediction = np.argmax(fused_scores, axis=1)
    mono_equal_prediction = np.argmax(channel_scores[:3].mean(axis=0), axis=1)
    bipolar_equal_prediction = np.argmax(channel_scores[3:].mean(axis=0), axis=1)

    atomic_save(
        path,
        subject_id=subject_id, class_count=class_count, frequencies_hz=frequencies,
        channel_names=CHANNEL_NAMES, channel_template_scores=channel_scores,
        selected_channel_weights=weights, fused_scores=fused_scores, fused_prediction=prediction,
        fused_accuracy=fused_accuracy, baseline_three_channel_accuracy=baseline_accuracy,
        recalibrated_equal_monopolar_accuracy=np.mean(mono_equal_prediction == labels),
        optimized_bipolar_accuracy=np.mean(bipolar_equal_prediction == labels),
        bipolar_grid=BIPOLAR_GRID, bipolar_grid_accuracy=grid_accuracy,
        selected_bipolar_index=best_index, selected_bipolar_alpha=bipolar_alpha,
        selected_bipolar_threshold=bipolar_threshold, selected_bipolar_operating_rms=bipolar_operating,
        selected_monopolar_alpha=mono_alpha, selected_monopolar_threshold=mono_threshold,
        selected_monopolar_operating_rms=mono_operating, selected_harmonics=np.asarray(harmonics),
        selected_spread_hz=np.asarray(spread), raw_monopolar_rms_uV=mono_rms.reshape(-1, 3),
        raw_bipolar_rms_uV=bipolar_rms.reshape(-1, 2), monopolar_gain_per_uV=mono_gain.reshape(-1, 3),
        bipolar_gain_per_uV=bipolar_gain.reshape(-1, 2), weight_candidates=weight_candidates(1000 * subject_id + class_count),
        weight_candidate_accuracy=weight_accuracy, elapsed_seconds=time.perf_counter() - started,
        evaluation_design="same_subject_same_segments_bipolar_parameter_and_fusion_weight_optimization_1s_no_holdout",
    )
    print(
        f"S{subject_id:02d} C{class_count:02d} complete base={100*baseline_accuracy:.1f}% "
        f"bipolar={100*np.mean(bipolar_equal_prediction == labels):.1f}% fused={100*fused_accuracy:.1f}% "
        f"weights={np.round(weights, 3)} elapsed={time.perf_counter()-started:.1f}s",
        flush=True,
    )


for sid in SUBJECTS:
    print(f"Loading subject {sid}/30", flush=True)
    data = load_subject(int(sid))
    for count in COUNTS:
        run_cell(int(sid), int(count), data)
print("FUSED REFERENCE SEARCH COMPLETE", flush=True)
