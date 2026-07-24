"""8-class single-encoder R&F versus BPF/LIF comparison.

Parameters are selected separately for every held-out stimulus block, using
only the other eleven blocks from that subject.  Checkpoints preserve the full
candidate response surface and block-wise selected parameters for analysis.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import os
import time

import numpy as np
import yaml

from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier, ResonateAndFireParameters
from ssvep_toolkit.algorithms.encoding import EncoderConfig, encode_spike_features
from ssvep_toolkit.algorithms.spike_encoding import LIFEncoderParameters
from ssvep_toolkit.data.matlab import Matlab73Dataset
from ssvep_toolkit.preprocessing import BandpassParameters, target_frequency_filter_bank
from ssvep_toolkit.evaluation.spike_encoder_experiment import lif_count_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent


def axis(spec: dict[str, object]) -> np.ndarray:
    minimum, maximum, count = float(spec["min"]), float(spec["max"]), int(spec["count"])
    if minimum <= 0 or maximum < minimum or count < 1:
        raise ValueError(f"invalid bounded parameter: {spec}")
    return np.geomspace(minimum, maximum, count) if spec.get("spacing") == "log" else np.linspace(minimum, maximum, count)


def candidates(section: dict[str, object], names: tuple[str, ...], optimization: dict[str, object]) -> np.ndarray:
    grid = np.asarray(list(product(*(axis(section[name]) for name in names))), dtype=float)
    if optimization.get("algorithm", "grid") == "grid":
        return grid
    if optimization.get("algorithm") != "random":
        raise ValueError("optimization.algorithm must be grid or random")
    count = min(int(optimization.get("random_candidates", len(grid))), len(grid))
    rng = np.random.default_rng(int(optimization.get("seed", 0)))
    return grid[rng.choice(len(grid), size=count, replace=False)]


def rf_candidates(section: dict[str, object], optimization: dict[str, object]) -> np.ndarray:
    """R&F candidates including the local oscillator-bank geometry."""
    base = candidates(section, ("alpha", "threshold", "input_gain"), {"algorithm": "grid"})
    counts = np.asarray(section.get("bank_neurons", [1]), dtype=float)
    widths = np.asarray(section.get("spread_half_width_hz", [0.0]), dtype=float)
    grid = np.asarray([(*row, count, width) for row in base for count in counts for width in widths], dtype=float)
    if optimization.get("algorithm", "grid") == "grid":
        return grid
    count = min(int(optimization.get("random_candidates", len(grid))), len(grid))
    rng = np.random.default_rng(int(optimization.get("seed", 0)))
    return grid[rng.choice(len(grid), size=count, replace=False)]


def score_from_training(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    center = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0, ddof=1), 1e-6)
    normalized_train = (train_x - center) / scale
    normalized_test = (test_x - center) / scale
    templates = np.stack([normalized_train[train_y == label].mean(axis=0) for label in range(int(train_y.max()) + 1)])
    return -np.mean((normalized_test[:, None, :] - templates[None, :, :]) ** 2, axis=-1)


def apparent_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(score_from_training(features, labels, features), axis=1) == labels))


def evaluate_surface(candidate_features: np.ndarray, labels: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate OOF accuracy, selected indices, OOF predictions, and block accuracies."""
    n_candidates, n_trials = candidate_features.shape[:2]
    candidate_correct = np.zeros((n_candidates, n_trials), dtype=bool)
    selected = np.empty(len(np.unique(blocks)), dtype=int)
    prediction = np.empty(n_trials, dtype=int)
    block_accuracy = np.empty(len(selected), dtype=float)
    for fold, block in enumerate(np.unique(blocks)):
        test = blocks == block; train = ~test
        inner = np.array([apparent_accuracy(values[train], labels[train]) for values in candidate_features])
        best = int(np.argmax(inner)); selected[fold] = best
        for index, values in enumerate(candidate_features):
            candidate_correct[index, test] = np.argmax(score_from_training(values[train], labels[train], values[test]), axis=1) == labels[test]
        prediction[test] = np.argmax(score_from_training(candidate_features[best, train], labels[train], candidate_features[best, test]), axis=1)
        block_accuracy[fold] = np.mean(prediction[test] == labels[test])
    return candidate_correct.mean(axis=1), selected, prediction, block_accuracy


def subject_trials(subject_id: int, frequencies: np.ndarray, duration_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Matlab73Dataset(DATA / f"data_s{subject_id}_64.mat") as source:
        posterior = source.read_channel_chunk(60, 64)[1, :3]
    # MATLAB storage is channel x sample x frequency x block.  Preserve the
    # validated loader's class x block x channel x sample convention.
    picked = posterior[:, 140:140 + duration_samples, frequencies - 1, :].transpose(2, 3, 0, 1)
    values = picked.reshape(len(frequencies) * 12, 3, duration_samples).astype(np.float32)
    labels = np.repeat(np.arange(len(frequencies)), 12)
    blocks = np.tile(np.arange(12), len(frequencies))
    return values, labels, blocks


def rf_features(raw: np.ndarray, frequencies: np.ndarray, params: np.ndarray, section: dict[str, object]) -> np.ndarray:
    # Three fixed summaries per target retain bank selectivity while allowing
    # fair template comparison across candidates with 1, 2, 4, or 8 neurons.
    output = np.empty((len(params), raw.shape[0], len(frequencies) * 3), dtype=np.float32)
    for index, (alpha, threshold, gain, bank_neurons, half_width) in enumerate(params):
        offsets = (0.0,) if int(bank_neurons) == 1 else tuple(np.linspace(-half_width, half_width, int(bank_neurons)))
        parameters = ResonateAndFireParameters(
                damping_alpha=float(alpha), threshold=float(threshold), input_gain=float(gain),
                transient_seconds=float(section["transient_seconds"]), refractory_cycles=float(section["refractory_cycles"]),
                integration_substeps=int(section["integration_substeps"]), reset_mode="zero", solver="exact",
                spike_detection="upward_crossing", normalize_input_by_resonance=bool(section["normalize_input_by_resonance"]),
        )
        model = OscillatorBankClassifier(frequencies, 1000, parameters, harmonics=tuple(section["harmonics"]), spread_hz=offsets)
        model.channel_scale_ = np.ones(raw.shape[1])
        grouped = model.neuron_scores(raw, (raw.shape[-1],))[0].mean(axis=2)
        summary = np.stack((grouped.mean(axis=-1), grouped.max(axis=-1), grouped.std(axis=-1)), axis=-1)
        output[index] = summary.reshape(raw.shape[0], -1)
    return output


def lif_features(raw: np.ndarray, frequencies: np.ndarray, params: np.ndarray, section: dict[str, object]) -> np.ndarray:
    bands = target_frequency_filter_bank(raw, frequencies, 1000, BandpassParameters(
        enabled=True, order=int(section["bandpass_order"]), half_width_hz=float(section["bandpass_half_width_hz"]), zero_phase=True,
    ))
    output = np.empty((len(params), raw.shape[0], len(frequencies) * raw.shape[1]), dtype=np.float32)
    for index, (threshold, tau, gain) in enumerate(params):
        output[index] = lif_count_features(bands, 1000, float(threshold), float(tau), input_gain=float(gain), preserve_channels=True)
    return output


def save(path: Path, **payload: object) -> None:
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/single_encoder_8class.yaml")
    parser.add_argument("--subjects", type=str, help="Comma-separated subject IDs")
    parser.add_argument("--encoders", default="resonate_fire,lif", help="resonate_fire,lif")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    study, optimization = config["study"], config["optimization"]
    subjects = [int(item) for item in args.subjects.split(",")] if args.subjects else list(map(int, study["subjects"]))
    frequencies = int(study["class_start_hz"]) + int(study["class_spacing_hz"]) * np.arange(int(study["class_count"]))
    if np.any(frequencies > 60): raise ValueError("8-class frequency grid exceeds available data")
    root = ROOT / config["output"]["root"]; checkpoints = root / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    (root / "study_plan.json").write_text(json.dumps({"config": config, "frequencies_hz": frequencies.tolist()}, indent=2), encoding="utf-8")
    selected_encoders = set(args.encoders.split(","))
    for subject in subjects:
        raw, labels, blocks = subject_trials(subject, frequencies, round(1000 * float(study["duration_seconds"])))
        for encoder, names, extractor in (
            ("resonate_fire", ("alpha", "threshold", "input_gain", "bank_neurons", "spread_half_width_hz"), rf_features),
            ("lif", ("threshold_uV", "tau_seconds", "input_gain"), lif_features),
        ):
            if encoder not in selected_encoders or not config[encoder].get("enabled", True): continue
            path = checkpoints / f"{encoder}_subject_{subject:02d}_08_classes.npz"
            if path.exists(): print(f"resume {path.name}", flush=True); continue
            started = time.perf_counter(); parameter_grid = rf_candidates(config[encoder], optimization) if encoder == "resonate_fire" else candidates(config[encoder], names, optimization)
            print(f"{encoder} S{subject:02d}: {len(parameter_grid)} candidates", flush=True)
            features = extractor(raw, frequencies, parameter_grid, config[encoder])
            surface, selected, prediction, block_accuracy = evaluate_surface(features, labels, blocks)
            save(path, encoder=encoder, subject_id=subject, class_count=8, frequencies_hz=frequencies,
                 parameter_names=np.asarray(names), parameter_grid=parameter_grid, candidate_oof_accuracy=surface,
                 selected_candidate_per_block=selected, selected_parameters_per_block=parameter_grid[selected],
                 prediction=prediction, labels=labels, blocks=blocks, block_accuracy=block_accuracy,
                 accuracy=float(np.mean(prediction == labels)), elapsed_seconds=time.perf_counter()-started,
                 evaluation_design="leave_one_stimulus_block_out; parameter selection on remaining blocks only",
                 optimization_algorithm=optimization["algorithm"])
            print(f"{encoder} S{subject:02d}: {100*np.mean(prediction == labels):.1f}%", flush=True)


if __name__ == "__main__": main()
