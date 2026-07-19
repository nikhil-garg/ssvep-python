"""Resumable subject-wise nested outer-block evaluation of R&F, delta and LIF."""
from __future__ import annotations

from itertools import product
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for dependency_path in (ROOT / "src", ROOT / ".venv" / "Lib" / "site-packages"):
    if dependency_path.exists() and str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import numpy as np
import yaml

from ssvep_toolkit.algorithms.resonate_and_fire import (OscillatorBankClassifier, ResonateAndFireParameters,
                                                        damping_from_bandwidth, simulate_bank_event_features)
from ssvep_toolkit.data.matlab import Matlab73Dataset
from ssvep_toolkit.evaluation import (
    CandidateFeatureBlock, evaluation_report, harmonic_collisions,
    factorial_class_sets, nested_grouped_linear_fusion, select_class_frequencies,
)
from ssvep_toolkit.preprocessing import (butterworth_bandpass, butterworth_bandpass_stream,
                                         butterworth_sos, apply_branch_gain, causal_running_gain,
                                         fit_prestimulus_branch_gain, fit_training_branch_gain)
from ssvep_toolkit.progress import ProgressJournal, WorkEstimate
from ssvep_toolkit.registry import ExperimentRegistry


CONFIG_PATH = Path(os.environ.get("SSVEP_NESTED_CONFIG", ROOT / "configs/nested_multi_encoder.yaml"))
BRANCH_NAMES = ("O1", "Oz", "O2", "O1-Oz", "O2-Oz")
_JOURNAL: ProgressJournal | None = None


def progress(phase: str, current: int, total: int, message: str) -> None:
    """Emit a stable machine-readable progress line for GUI and log clients."""
    display = message
    if _JOURNAL is not None:
        event = _JOURNAL.write(phase, int(current), int(total), message)
        if event["eta_seconds"] is not None:
            eta = float(event["eta_seconds"])
            display += f" · ETA {eta:.0f} s" if eta < 60 else f" · ETA {eta/60:.1f} min"
    print(f"@@PROGRESS|{phase}|{int(current)}|{int(total)}|{display}", flush=True)


def busy(message: str) -> None:
    print(f"@@BUSY|{message}", flush=True)
    if _JOURNAL is not None:
        _JOURNAL.write("busy", None, None, message)


def load_subject(subject_id: int, condition: int) -> np.ndarray:
    with Matlab73Dataset(ROOT.parent / f"data_s{subject_id}_64.mat") as source:
        chunk = source.read_channel_chunk(60, 64)
    original = chunk[condition - 1, :3].transpose(2, 3, 0, 1).astype(np.float32)
    return np.concatenate((original, original[..., (0,), :] - original[..., (1,), :],
                           original[..., (2,), :] - original[..., (1,), :]), axis=2)


def rf_parameter_grid(config: dict, frequencies: np.ndarray) -> list[tuple[float, float, float]]:
    """Build an identifiable damping/threshold grid, optionally from bandwidth."""
    if config.get("bandwidth_hz"):
        center = float(np.median(frequencies))
        alphas = [float(damping_from_bandwidth(center, width)) for width in config["bandwidth_hz"]]
    else:
        alphas = [float(value) for value in config["damping_alpha"]]
    return list(product(alphas, config["threshold"], config["operating_rms"]))


def adapt_rf_signals(raw: np.ndarray, config: dict, *, training_mask: np.ndarray | None = None,
                     prestimulus: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the configured gain strategy with explicit provenance."""
    centered = raw - raw.mean(axis=-1, keepdims=True)
    gain = config.get("gain", {}); mode = gain.get("mode", "legacy_segment_rms")
    target = float(gain.get("target_rms", config.get("operating_rms", [0.75])[0]))
    if mode == "legacy_segment_rms":
        rms = np.sqrt(np.mean(centered.astype(float) ** 2, axis=-1, keepdims=True))
        return centered * (target / np.maximum(rms, 1e-6)), {"mode": mode, "target_rms": target}
    if mode == "training_branch":
        if training_mask is None:
            raise ValueError("training_branch gain requires an outer-training mask")
        calibration = fit_training_branch_gain(
            raw, training_mask, target_rms=target, method=gain.get("statistic", "median"),
        )
        return apply_branch_gain(raw, calibration), {
            "mode": mode, "target_rms": target,
            "reference_rms_uV": np.asarray(calibration.reference_rms).tolist(),
            "gain_per_uV": np.asarray(calibration.gain_per_unit).tolist(),
            "training_trials": calibration.training_trials,
        }
    if mode == "prestimulus":
        if prestimulus is None:
            raise ValueError("prestimulus gain requires pre-stimulus samples")
        per_trial = fit_prestimulus_branch_gain(prestimulus, target_rms=target)
        return centered * per_trial[..., None], {"mode": mode, "target_rms": target}
    if mode == "causal_running":
        initial = None if prestimulus is None else np.sqrt(np.mean(
            (prestimulus - prestimulus.mean(axis=-1, keepdims=True)) ** 2, axis=-1,
        ))
        initial_mean = None if prestimulus is None else prestimulus.mean(axis=-1)
        adapted, trace = causal_running_gain(
            raw, 1000, target_rms=target, tau_seconds=float(gain.get("tau_seconds", 0.5)),
            initial_rms=initial, initial_mean=initial_mean, maximum_gain=gain.get("maximum_gain"),
        )
        return adapted, {"mode": mode, "target_rms": target,
                         "gain_quantiles": np.quantile(trace, (0, .05, .5, .95, 1)).tolist()}
    raise ValueError(f"unknown R&F gain mode: {mode}")


def rf_candidates(raw: np.ndarray, frequencies: np.ndarray, duration: int, config: dict, *,
                  training_mask: np.ndarray | None = None,
                  prestimulus: np.ndarray | None = None) -> CandidateFeatureBlock:
    grid = rf_parameter_grid(config, frequencies)
    event_index = {"rate": 0, "ttfs": 1, "phase_cos": 2, "phase_sin": 3}
    event_features = tuple(config.get("event_features", ("rate",)))
    if not event_features or any(name not in event_index for name in event_features):
        raise ValueError(f"R&F event_features must be selected from {tuple(event_index)}")
    values = np.empty((len(grid), raw.shape[0], len(BRANCH_NAMES) * len(frequencies) * len(event_features)), np.float32)
    adapted, gain_provenance = adapt_rf_signals(
        raw, config, training_mask=training_mask, prestimulus=prestimulus,
    )
    for index, (alpha, threshold, operating) in enumerate(grid):
        print(f"  R&F feature candidate {index + 1}/{len(grid)}", flush=True)
        per_branch = []
        parameters = ResonateAndFireParameters(
            damping_alpha=float(alpha), threshold=float(threshold), input_gain=float(config["input_gain"]),
            normalize_input_by_resonance=bool(config.get("normalize_input_by_resonance", False)),
            integration_substeps=int(config["integration_substeps"]),
            refractory_cycles=float(config["refractory_cycles"]), solver="exact",
            reset_mode="zero", spike_detection="upward_crossing",
        )
        for branch in range(adapted.shape[1]):
            model = OscillatorBankClassifier(
                frequencies, 1000, parameters, harmonics=config["harmonics"],
                spread_hz=config["spread_hz"],
            )
            model.channel_scale_ = np.ones(1)
            events = simulate_bank_event_features(
                model.transform(adapted[:, branch:branch+1, :duration]), model.neuron_frequencies_hz,
                1000, parameters, (duration,),
            )[0]
            shaped = events.reshape(
                events.shape[0], len(frequencies), len(model.harmonics), len(model.spread_hz), 4,
            ).mean(axis=3)
            harmonic_weights = np.asarray(model.harmonic_weights, dtype=float)
            grouped = np.sum(shaped * harmonic_weights[None, None, :, None], axis=2) / harmonic_weights.sum()
            per_branch.append(grouped[..., [event_index[name] for name in event_features]].reshape(raw.shape[0], -1))
        values[index] = np.concatenate(per_branch, axis=1)
        progress("rf_candidates", index + 1, len(grid), "Generating R&F candidate features")
    names = tuple(
        f"rf:{branch}:target_{frequency:g}Hz:{feature}"
        for branch in BRANCH_NAMES for frequency in frequencies for feature in event_features
    )
    ids_list = []
    for a, t, o in grid:
        candidate_id = {
            "alpha": a, "threshold": t, "operating_rms": o,
            "drive_threshold_ratio": float(o) * float(config["input_gain"]) / float(t),
        }
        # Preserve the locked confirmatory checkpoint schema when its legacy
        # configuration has no explicit gain/bandwidth design.
        if config.get("gain") or config.get("bandwidth_hz"):
            candidate_id.update(
                gain_mode=gain_provenance["mode"],
                bandwidth_at_center_hz=float(a) * float(np.median(frequencies)) / np.pi,
            )
        ids_list.append(candidate_id)
    ids = tuple(ids_list)
    if "rate" in event_features:
        rate_offset = event_features.index("rate")
        available_seconds = max((duration - round(float(config.get("transient_seconds", .1)) * 1000)) / 1000, .001)
        costs = np.mean(np.maximum(values[..., rate_offset::len(event_features)], 0), axis=2) * available_seconds
    else:
        costs = None
    return CandidateFeatureBlock("resonate_fire", values, ids, names, costs)


def filtered_candidates(raw_full: np.ndarray, frequencies: np.ndarray, crop: slice,
                        mode: str, bandpass: dict, delta: dict, lif: dict) -> tuple[CandidateFeatureBlock, CandidateFeatureBlock]:
    delta_grid = list(product(delta["threshold_uV"], delta["asymmetry"]))
    lif_grid = list(product(lif["threshold_uV"], lif["tau_seconds"]))
    trials, branches = raw_full.shape[:2]; targets = len(frequencies)
    delta_values = np.zeros((len(delta_grid), trials, targets * branches * 2), np.float32)
    lif_values = np.zeros((len(lif_grid), trials, targets * branches), np.float32)
    zero_phase = mode == "offline"
    for target, frequency in enumerate(frequencies):
        print(f"  {mode} filtered target {target + 1}/{targets}: {frequency:g} Hz", flush=True)
        low = max(.1, float(frequency) - float(bandpass["half_width_hz"]))
        high = float(frequency) + float(bandpass["half_width_hz"])
        if zero_phase:
            filtered = butterworth_bandpass(
                raw_full, 1000, low, high, order=int(bandpass["order"]), zero_phase=True,
            )[..., crop]
        else:
            sos = butterworth_sos(1000, low, high, order=int(bandpass["order"]))
            _, state = butterworth_bandpass_stream(raw_full[..., :crop.start], sos)
            filtered, _ = butterworth_bandpass_stream(raw_full[..., crop], sos, state)
        differences = np.diff(filtered, axis=-1, prepend=filtered[..., :1])
        delta_thresholds = np.asarray([item[0] for item in delta_grid], dtype=float)
        down_thresholds = np.asarray([item[0] * item[1] for item in delta_grid], dtype=float)
        up_counts = np.sum(
            differences[None, ...] > delta_thresholds[:, None, None, None], axis=-1,
        )
        down_counts = np.sum(
            differences[None, ...] < -down_thresholds[:, None, None, None], axis=-1,
        )
        pairs = np.stack((up_counts, down_counts), axis=-1)
        start = target * branches * 2
        delta_values[:, :, start:start + branches * 2] = pairs.reshape(len(delta_grid), trials, -1)

        lif_thresholds = np.asarray([item[0] for item in lif_grid], dtype=float)
        lif_taus = np.asarray([item[1] for item in lif_grid], dtype=float)
        decay = np.exp(-1 / (1000 * lif_taus))[:, None, None]
        membrane = np.zeros((len(lif_grid), trials, branches), dtype=float)
        counts = np.zeros_like(membrane)
        for sample in range(filtered.shape[-1]):
            previous = membrane.copy()
            membrane = decay * membrane + (1 - decay) * float(lif["input_gain"]) * filtered[None, ..., sample]
            spike = (previous < lif_thresholds[:, None, None]) & (
                membrane >= lif_thresholds[:, None, None]
            )
            counts += spike
            membrane[spike] = 0
        start = target * branches
        lif_values[:, :, start:start + branches] = counts.astype(np.float32)
        progress(f"{mode}_filter_bank", target + 1, targets,
                 f"Encoding {mode} delta/LIF target filters")
    delta_names = tuple(f"delta:{branch}:{stream}:target_{frequency:g}Hz" for frequency in frequencies
                        for branch in BRANCH_NAMES for stream in ("UP", "DN"))
    lif_names = tuple(f"lif:{branch}:target_{frequency:g}Hz" for frequency in frequencies for branch in BRANCH_NAMES)
    delta_ids = tuple({"threshold_uV": t, "asymmetry": a, "filter": mode} for t, a in delta_grid)
    lif_ids = tuple({"threshold_uV": t, "tau_seconds": tau, "filter": mode} for t, tau in lif_grid)
    delta_costs = delta_values.sum(axis=2)
    lif_costs = lif_values.sum(axis=2)
    return (CandidateFeatureBlock("delta", delta_values, delta_ids, delta_names, delta_costs),
            CandidateFeatureBlock("lif", lif_values, lif_ids, lif_names, lif_costs))


def atomic_save(path: Path, **payload: object) -> None:
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def save_feature_cache(path: Path, *blocks: CandidateFeatureBlock) -> None:
    """Write fast temporary feature caches so interrupted runs resume cheaply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    payload: dict[str, object] = {"block_count": len(blocks)}
    for index, block in enumerate(blocks):
        payload[f"name_{index}"] = block.name
        payload[f"values_{index}"] = block.values
        payload[f"candidate_ids_{index}"] = json.dumps(block.candidate_ids)
        payload[f"feature_names_{index}"] = np.asarray(block.feature_names)
        if block.candidate_costs is not None:
            payload[f"candidate_costs_{index}"] = np.asarray(block.candidate_costs)
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def load_feature_cache(path: Path) -> tuple[CandidateFeatureBlock, ...]:
    with np.load(path, allow_pickle=False) as cache:
        count = int(cache["block_count"])
        return tuple(CandidateFeatureBlock(
            str(cache[f"name_{index}"]), np.asarray(cache[f"values_{index}"]),
            tuple(json.loads(str(cache[f"candidate_ids_{index}"]))),
            tuple(str(value) for value in cache[f"feature_names_{index}"]),
            np.asarray(cache[f"candidate_costs_{index}"]) if f"candidate_costs_{index}" in cache else None,
        ) for index in range(count))


def branch_dropout(test: np.ndarray, train: np.ndarray, names: tuple[str, ...], held_out: int) -> np.ndarray:
    branch = BRANCH_NAMES[int(held_out) % len(BRANCH_NAMES)]
    mask = np.asarray([f":{branch}:" in name for name in names])
    test[:, mask] = 0
    return test


def count_noise(fraction: float):
    def transform(test: np.ndarray, train: np.ndarray, names: tuple[str, ...], held_out: int) -> np.ndarray:
        scale = np.maximum(train.std(axis=0, ddof=1), 1e-6)
        return test + np.random.default_rng(1000 + int(held_out)).normal(0, fraction * scale, test.shape)
    return transform


def zero_named_features(label: str, predicate):
    """Create a transparent test-time ablation without refitting the model."""
    def transform(test: np.ndarray, train: np.ndarray, names: tuple[str, ...], held_out: int) -> np.ndarray:
        mask = np.asarray([bool(predicate(name)) for name in names])
        test[:, mask] = 0
        return test
    transform.__name__ = f"zero_{label}"
    return transform


def main() -> None:
    global _JOURNAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--class-counts", nargs="+", type=int, choices=(2, 4, 8, 16, 32))
    parser.add_argument("--class-selection-strategy", choices=(
        "fixed_spacing_harmonic_aware", "compact_harmonic_aware", "low_contiguous", "legacy_spread",
    ))
    parser.add_argument("--log-file", type=Path, help="append stdout and stderr to a persistent log")
    args = parser.parse_args()
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")); study_cfg = config["study"]
    output = ROOT / study_cfg["output"]; checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(ROOT / study_cfg["registry"]).initialize()
    study = registry.create_study(study_cfg["name"], config, "Outer held-out block, inner encoder and ridge selection.")
    subjects = args.subjects or [int(x) for x in os.environ.get(
        "SSVEP_NESTED_SUBJECTS", ",".join(map(str, study_cfg["subjects"]))).split(",")]
    counts = args.class_counts or [int(x) for x in os.environ.get(
        "SSVEP_NESTED_CLASS_COUNTS", ",".join(map(str, study_cfg["class_counts"]))).split(",")]
    estimate = WorkEstimate(
        subjects=len(subjects), class_tasks=len(counts), filter_modes=len(study_cfg["filter_modes"]),
        rf_candidates=(len(config["resonate_fire"]["damping_alpha"]) *
                       len(config["resonate_fire"]["threshold"]) *
                       len(config["resonate_fire"]["operating_rms"])),
        delta_candidates=len(config["delta"]["threshold_uV"]) * len(config["delta"]["asymmetry"]),
        lif_candidates=len(config["lif"]["threshold_uV"]) * len(config["lif"]["tau_seconds"]),
        ridge_candidates=len(config["fusion"]["l2_grid"]),
    )
    _JOURNAL = ProgressJournal(output / "progress.jsonl", config=config, study=study_cfg["name"], estimate=estimate)
    dataset_manifest = []
    for subject in subjects:
        source = ROOT.parent / f"data_s{subject}_64.mat"
        dataset_manifest.append({
            "subject": subject, "path": source.name, "bytes": source.stat().st_size,
            "modified_ns": source.stat().st_mtime_ns,
        })
    selection_plan = study_cfg.get("class_selection", {})
    factorial_cfg = selection_plan.get("factorial_audit", {})
    factorial_designs = factorial_class_sets(
        counts, factorial_cfg.get("spacings_hz", (2, 4)),
        available_hz=selection_plan.get("available_hz", (8, 60)),
        starts_hz=factorial_cfg.get("starts_hz"),
        interference_harmonics=selection_plan.get("interference_harmonics", (2, 3)),
        maximum_collisions=factorial_cfg.get("maximum_collisions", 0),
    ) if factorial_cfg else ()
    (output / "run_plan.json").write_text(json.dumps({
        "study": study_cfg["name"], "config_sha256": _JOURNAL.config_hash,
        "work_estimate": estimate.to_dict(), "subjects": subjects, "class_counts": counts,
        "filter_modes": study_cfg["filter_modes"], "dataset_manifest": dataset_manifest,
        "factorial_class_designs": [design.to_dict() for design in factorial_designs],
    }, indent=2), encoding="utf-8")
    print(f"PLAN {json.dumps(estimate.to_dict(), sort_keys=True)}", flush=True)
    duration = round(float(study_cfg["decision_seconds"]) * 1000); crop = slice(140, 140 + duration)
    feature_cache = output / "feature_cache"; feature_cache.mkdir(parents=True, exist_ok=True)
    storage = config.get("storage", {})
    optimization = config.get("optimization", {})
    cache_tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    for subject in subjects:
        if all((checkpoints / f"subject_{subject:02d}_{count:02d}_classes_{mode}.npz").exists()
               for count in counts for mode in study_cfg["filter_modes"]):
            print(f"resume subject {subject}: every requested checkpoint exists", flush=True); continue
        busy(f"Loading subject {subject} EEG")
        raw_all = load_subject(subject, int(study_cfg["condition"]))
        for count in counts:
            paths_by_mode = {mode: checkpoints / f"subject_{subject:02d}_{count:02d}_classes_{mode}.npz"
                             for mode in study_cfg["filter_modes"]}
            pending_modes = [mode for mode, path in paths_by_mode.items() if not path.exists()]
            if not pending_modes:
                print(f"resume subject {subject}, {count} classes: complete", flush=True); continue
            selection = study_cfg.get("class_selection", {})
            selection_strategy = args.class_selection_strategy or selection.get("strategy", "legacy_spread")
            frequencies = np.asarray(select_class_frequencies(
                count,
                available_hz=selection.get("available_hz", study_cfg.get("class_range_hz", (8, 39))),
                strategy=selection_strategy,
                interference_harmonics=selection.get("interference_harmonics", (2, 3)),
                spacing_hz=selection.get("spacing_hz_by_class_count", {}).get(count),
                start_hz=selection.get("start_hz"),
            ), dtype=int)
            collisions = harmonic_collisions(
                frequencies, selection.get("interference_harmonics", (2, 3)),
            )
            print(
                f"{count} classes: {frequencies.tolist()} · "
                f"harmonic collisions={[(x.source_hz, x.harmonic, x.target_hz) for x in collisions]}",
                flush=True,
            )
            selected = raw_all[frequencies - 1]
            raw_full = selected.reshape(-1, len(BRANCH_NAMES), selected.shape[-1])
            raw = raw_full[..., crop]; labels = np.repeat(np.arange(count), 12); groups = np.tile(np.arange(12), count)
            rf_cache = feature_cache / f"subject_{subject:02d}_{count:02d}_{cache_tag}_rf.npz"
            if rf_cache.exists():
                busy("Loading cached R&F features")
                print(f"loading {rf_cache.name}", flush=True); rf = load_feature_cache(rf_cache)[0]
            else:
                rf = rf_candidates(raw, frequencies, duration, config["resonate_fire"])
                save_feature_cache(rf_cache, rf); print(f"saved {rf_cache.name}", flush=True)
            for mode in pending_modes:
                path = paths_by_mode[mode]
                mode_cache = feature_cache / f"subject_{subject:02d}_{count:02d}_{cache_tag}_{mode}_delta_lif.npz"
                started = time.perf_counter()
                if mode_cache.exists():
                    busy(f"Loading cached {mode} delta/LIF features")
                    print(f"loading {mode_cache.name}", flush=True); delta, lif = load_feature_cache(mode_cache)
                else:
                    delta, lif = filtered_candidates(
                        raw_full, frequencies, crop, mode, config["bandpass"], config["delta"], config["lif"],
                    )
                    save_feature_cache(mode_cache, delta, lif); print(f"saved {mode_cache.name}", flush=True)
                transforms = {"feature_count_noise": count_noise(
                    float(config["robustness"]["feature_count_noise_std_fraction"])
                )}
                if config["robustness"]["rotating_branch_dropout"]:
                    transforms["rotating_branch_dropout"] = branch_dropout
                ablations = config.get("ablations", {})
                if ablations.get("enabled", False):
                    for branch in ablations.get("branches", BRANCH_NAMES):
                        transforms[f"drop_branch:{branch}"] = zero_named_features(
                            f"branch_{branch}", lambda name, branch=branch: f":{branch}:" in name,
                        )
                    for encoder in ablations.get("encoders", ("rf", "delta", "lif")):
                        transforms[f"drop_encoder:{encoder}"] = zero_named_features(
                            f"encoder_{encoder}", lambda name, encoder=encoder: name.startswith(f"{encoder}:"),
                        )
                result = nested_grouped_linear_fusion(
                    (rf, delta, lif), labels, groups, l2_grid=config["fusion"]["l2_grid"],
                    candidate_selection_l2=float(config["fusion"]["candidate_selection_l2"]),
                    candidate_selection_rule=optimization.get("candidate_selection_rule", "max_mean"),
                    candidate_reference_by_block=optimization.get("reference_parameters", {}),
                    candidate_fidelity=optimization.get("multi_fidelity", {}),
                    l2_selection_rule=optimization.get("l2_selection_rule", "max_mean"),
                    progress_callback=lambda current, total: progress(
                        "primary_fusion", current, total, "Nested primary outer folds"
                    ),
                    outer_test_transforms=transforms,
                )
                report = evaluation_report(
                    labels, result.predictions, classes=count, decision_seconds=duration / 1000,
                    onset_latency_seconds=float(study_cfg["onset_latency_seconds"]),
                    practical_overhead_seconds=float(study_cfg["practical_overhead_seconds"]),
                    spike_counts=result.out_of_fold_features,
                    perturbed_predictions=result.perturbed_predictions.get("rotating_branch_dropout"),
                )
                selections = json.dumps([{
                    "held_out_block": int(fold.held_out_group), "candidate_ids": fold.selected_candidate_ids,
                    "candidate_diagnostics": fold.inner_candidate_diagnostics,
                    "l2": fold.selected_l2,
                    "l2_boundary": (
                        "lower" if np.isclose(fold.selected_l2, min(config["fusion"]["l2_grid"])) else
                        "upper" if np.isclose(fold.selected_l2, max(config["fusion"]["l2_grid"])) else None
                    ),
                    "outer_accuracy": fold.outer_accuracy,
                } for fold in result.folds])
                boundary_hits = sum(
                    len(diagnostic["boundary_hits"])
                    for fold in result.folds for diagnostic in fold.inner_candidate_diagnostics
                )
                boundary_opportunities = sum(
                    diagnostic["searched_parameter_count"]
                    for fold in result.folds for diagnostic in fold.inner_candidate_diagnostics
                )
                boundary_hit_fraction = boundary_hits / max(boundary_opportunities, 1)
                boundary_tallies: dict[str, list[int]] = {}
                for fold in result.folds:
                    for block_name, diagnostic in zip(("resonate_fire", "delta", "lif"),
                                                      fold.inner_candidate_diagnostics):
                        for parameter in diagnostic["searched_parameters"]:
                            key = f"{block_name}.{parameter}"
                            boundary_tallies.setdefault(key, [0, 0])[1] += 1
                        for parameter in diagnostic["boundary_hits"]:
                            boundary_tallies[f"{block_name}.{parameter}"][0] += 1
                boundary_rates = {
                    key: hits / opportunities for key, (hits, opportunities) in boundary_tallies.items()
                }
                l2_boundary_hit_fraction = np.mean([
                    np.isclose(fold.selected_l2, min(config["fusion"]["l2_grid"])) or
                    np.isclose(fold.selected_l2, max(config["fusion"]["l2_grid"]))
                    for fold in result.folds
                ])
                centered = raw - raw.mean(axis=-1, keepdims=True)
                segment_rms_uV = np.sqrt(np.mean(centered.astype(float) ** 2, axis=-1))
                oof_gain_per_uV = np.empty_like(segment_rms_uV)
                for fold in result.folds:
                    test = groups == fold.held_out_group
                    operating = float(fold.selected_candidate_ids[0]["operating_rms"])
                    oof_gain_per_uV[test] = operating / np.maximum(segment_rms_uV[test], 1e-6)
                quantile_levels = np.asarray((0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
                checkpoint_payload = dict(
                    subject_id=subject, class_count=count, filter_mode=mode,
                    frequencies_hz=frequencies, labels=labels, groups=groups, predictions=result.predictions,
                    decision_scores=result.decision_scores,
                    feature_names=np.asarray(result.feature_names), accuracy=result.accuracy,
                    accuracy_by_block=result.accuracy_by_group, fold_selections_json=selections,
                    candidate_pool_json=json.dumps({
                        block.name: block.candidate_ids for block in (rf, delta, lif)
                    }),
                    inner_candidate_accuracy_resonate_fire=np.stack([
                        fold.inner_candidate_accuracy[0] for fold in result.folds
                    ]).astype(np.float32),
                    inner_candidate_accuracy_delta=np.stack([
                        fold.inner_candidate_accuracy[1] for fold in result.folds
                    ]).astype(np.float32),
                    inner_candidate_accuracy_lif=np.stack([
                        fold.inner_candidate_accuracy[2] for fold in result.folds
                    ]).astype(np.float32),
                    inner_l2_accuracy=np.stack([
                        fold.inner_l2_accuracy for fold in result.folds
                    ]).astype(np.float32),
                    distribution_quantile_levels=quantile_levels,
                    segment_branch_rms_uV_quantiles=np.quantile(segment_rms_uV, quantile_levels, axis=0),
                    oof_selected_gain_per_uV_quantiles=np.quantile(oof_gain_per_uV, quantile_levels, axis=0),
                    branch_names=np.asarray(BRANCH_NAMES),
                    class_selection_strategy=selection_strategy,
                    class_design_json=json.dumps({
                        "class_count": count, "frequencies_hz": frequencies.tolist(),
                        "start_hz": int(frequencies[0]),
                        "spacing_hz": int(np.min(np.diff(frequencies))),
                        "span_hz": int(frequencies[-1] - frequencies[0]),
                        "minimum_spacing_hz": int(np.min(np.diff(frequencies))),
                        "harmonic_collision_count": len(collisions),
                    }),
                    harmonic_collisions_json=json.dumps([
                        {"source_hz": x.source_hz, "harmonic": x.harmonic, "target_hz": x.target_hz}
                        for x in collisions
                    ]),
                    rotating_branch_dropout_accuracy=result.perturbed_accuracy.get("rotating_branch_dropout", np.nan),
                    feature_count_noise_accuracy=result.perturbed_accuracy["feature_count_noise"],
                    perturbation_accuracy_json=json.dumps(result.perturbed_accuracy),
                    perturbation_predictions_json=json.dumps({
                        name: prediction.astype(int).tolist()
                        for name, prediction in result.perturbed_predictions.items()
                    }),
                    report_json=json.dumps(report.to_dict()), elapsed_seconds=time.perf_counter() - started,
                    optimization_procedure_json=json.dumps({
                        "candidate_selection_rule": optimization.get("candidate_selection_rule", "max_mean"),
                        "l2_selection_rule": optimization.get("l2_selection_rule", "max_mean"),
                        "reference_parameters": optimization.get("reference_parameters", {}),
                        "boundary_action": optimization.get("boundary_action", "report"),
                        "boundary_review": optimization.get("boundary_review", {}),
                        "multi_fidelity": optimization.get("multi_fidelity", {}),
                    }),
                    optimization_boundary_hit_count=boundary_hits,
                    optimization_boundary_hit_fraction=boundary_hit_fraction,
                    optimization_boundary_rates_json=json.dumps(boundary_rates),
                    optimization_l2_boundary_hit_fraction=l2_boundary_hit_fraction,
                    evaluation_design="within_subject_outer_held_out_block_inner_candidate_and_ridge_selection",
                    causal_filter_state_carried=(mode == "causal"),
                    causal_context_samples=(crop.start if mode == "causal" else 0),
                    offline_uses_future_samples=(mode == "offline"),
                    filter_provenance_json=json.dumps({
                        "mode": mode, "order": config["bandpass"]["order"],
                        "half_width_hz": config["bandpass"]["half_width_hz"],
                        "causal_state_initialized_at_source_start": mode == "causal",
                        "causal_context_samples_before_decision": crop.start if mode == "causal" else 0,
                        "zero_phase_uses_samples_after_decision": mode == "offline",
                    }),
                )
                if bool(storage.get("save_trial_features", True)):
                    checkpoint_payload.update(
                        out_of_fold_features=result.out_of_fold_features,
                        segment_branch_rms_uV=segment_rms_uV,
                        oof_selected_gain_per_uV=oof_gain_per_uV,
                    )
                busy("Compressing and saving checkpoint")
                atomic_save(path, **checkpoint_payload)
                run = registry.create_run(study, path.name, config, subject_id=subject, class_count=count,
                                          encoder="grouped_linear_fusion", status="running")
                for name, value in report.to_dict().items():
                    if isinstance(value, (int, float)) and value is not None:
                        registry.replace_metric(run, name, value, split="outer_test")
                if "rotating_branch_dropout" in result.perturbed_accuracy:
                    registry.replace_metric(run, "rotating_branch_dropout_accuracy",
                                            result.perturbed_accuracy["rotating_branch_dropout"],
                                            split="outer_test_perturbed")
                registry.replace_metric(run, "feature_count_noise_accuracy",
                                        result.perturbed_accuracy["feature_count_noise"],
                                        split="outer_test_perturbed")
                for name, value in result.perturbed_accuracy.items():
                    if name not in {"rotating_branch_dropout", "feature_count_noise"}:
                        registry.replace_metric(run, f"ablation_accuracy:{name}", value,
                                                split="outer_test_perturbed")
                registry.replace_metric(run, "optimization_boundary_hit_fraction", boundary_hit_fraction,
                                        split="inner_validation")
                registry.replace_metric(run, "optimization_l2_boundary_hit_fraction",
                                        l2_boundary_hit_fraction, split="inner_validation")
                for parameter, rate in boundary_rates.items():
                    registry.replace_metric(run, f"boundary_rate:{parameter}", rate,
                                            split="inner_validation")
                registry.add_artifact(run, "checkpoint", path); registry.set_status(run, "complete")
                print(f"{path.name}: {100*result.accuracy:.2f}%", flush=True)
                progress("cell_complete", 1, 1, f"Completed {path.name}")
            if all(path.exists() for path in paths_by_mode.values()) and not bool(storage.get("keep_feature_cache", False)):
                for cache_path in feature_cache.glob(f"subject_{subject:02d}_{count:02d}_*.npz"):
                    cache_path.unlink()
                print(f"removed completed temporary feature cache for subject {subject}, {count} classes", flush=True)
    _JOURNAL.write("study_complete", 1, 1, "All requested cells completed", status="complete")


if __name__ == "__main__":
    main()
