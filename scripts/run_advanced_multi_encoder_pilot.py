"""Run the joint, gain-safe, endpoint-aware multi-encoder pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts", ROOT / ".venv" / "Lib" / "site-packages"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import yaml

from run_nested_multi_encoder import (BRANCH_NAMES, atomic_save, filtered_candidates, load_subject,
                                      rf_candidates, rf_parameter_grid)
from ssvep_toolkit.evaluation import (FoldSafeCandidateFeatureBlock, JointSearchConfig,
                                      RacingConfig, advanced_nested_grouped_fusion,
                                      endpoint_results, evaluate_fbcca, evaluate_trca,
                                      harmonic_collisions, pareto_endpoints, select_class_frequencies)


def rf_schema(config: dict, frequencies: np.ndarray) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    event_features = tuple(config.get("event_features", ("rate",)))
    mode = config.get("gain", {}).get("mode", "legacy_segment_rms")
    ids = tuple({
        "alpha": alpha, "threshold": threshold, "operating_rms": operating,
        "drive_threshold_ratio": float(operating) * float(config["input_gain"]) / float(threshold),
        "gain_mode": mode,
        "bandwidth_at_center_hz": float(alpha) * float(np.median(frequencies)) / np.pi,
    } for alpha, threshold, operating in rf_parameter_grid(config, frequencies))
    names = tuple(f"rf:{branch}:target_{frequency:g}Hz:{feature}"
                  for branch in BRANCH_NAMES for frequency in frequencies for feature in event_features)
    return ids, names


def serializable_folds(result) -> list[dict[str, object]]:
    return [{
        "held_out_block": int(fold.held_out_group),
        "candidate_ids": fold.selected_candidate_ids,
        "candidate_indices": fold.selected_candidate_indices,
        "l2": fold.selected_l2,
        "inner_metrics": fold.inner_metrics.__dict__,
        "screening": fold.screening_diagnostics,
        "joint_search": fold.joint_diagnostics,
        "outer_accuracy": fold.outer_accuracy,
    } for fold in result.folds]


def historical_matlab_reference() -> dict[str, object]:
    """Summarize the completed paper-compatible Figure 10 reproduction."""
    path = ROOT / "outputs/reference_study/results/figure_10_fbcca.npz"
    if not path.exists():
        return {"available": False, "path": str(path)}
    with np.load(path) as result:
        durations = result["x"]; accuracy = result["accuracy"]
        indices = [int(np.flatnonzero(np.isclose(durations, value))[0]) for value in (.2, .3, .4, .5)]
        # Condition 2, middle paper band (12, 13, 14, 15 Hz).
        selected = np.take(accuracy[:, 1, 1, :], indices, axis=-1)
    return {
        "available": True, "path": str(path.relative_to(ROOT)), "condition": 2,
        "classes_hz": [12, 13, 14, 15], "endpoints_seconds": [.2, .3, .4, .5],
        "cohort_mean_accuracy": selected.mean(axis=0).tolist(),
        "pilot_subject_accuracy_500ms": {
            str(subject): float(selected[subject - 1, -1]) for subject in (1, 2, 4)
        },
        "scope_warning": "Historical paper band; not the matched 17 Hz-start class sets.",
    }


def matched_matlab_baselines(subject: int, condition: int, frequencies: np.ndarray,
                             endpoint_seconds: float) -> dict[str, object]:
    """Evaluate conventional baselines on MATLAB-compatible 250 Hz epochs."""
    import h5py

    path = ROOT / f"outputs/reference_study/preprocessed/subject_{subject:02d}_preprocessed.h5"
    with h5py.File(path, "r") as source:
        fs = float(source.attrs["sampling_rate_hz"]); samples = round(endpoint_seconds * fs)
        # stored axes: condition, channel, sample, frequency, block
        complete = np.asarray(source["data"][condition - 1])
    data = complete[:, :samples, frequencies - 1, :].transpose(2, 3, 0, 1)
    # Transfer the paper's condition-2 middle-band FBCCA parameters to the
    # exact requested class set. This is explicit because no paper parameter
    # table exists for an arbitrary 16-frequency class set.
    fbcca_prediction, fbcca_accuracy = evaluate_fbcca(
        data, frequencies, fs, first_low_hz=8.8, harmonics=10,
        subbands=3, weight_a=0.0, weight_b=0.0,
    )
    trca_prediction, trca_accuracy = evaluate_trca(data)
    return {
        "sampling_rate_hz": fs, "channels": 9, "matlab_compatible_downsampling": True,
        "fbcca_transferred_middle_band_accuracy": fbcca_accuracy,
        "fbcca_transferred_middle_band_predictions": fbcca_prediction.reshape(-1).tolist(),
        "trca_leave_block_out_accuracy": trca_accuracy,
        "trca_leave_block_out_predictions": trca_prediction.reshape(-1).tolist(),
        "warning": "FBCCA uses paper condition-2 middle-band parameters transferred to the matched class set.",
    }


def append_progress(path: Path, **event: object) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), **event,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nested_multi_encoder_joint_pilot.yaml")
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--class-counts", nargs="+", type=int)
    parser.add_argument("--gain-mode", choices=("training_branch", "prestimulus", "causal_running"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")); study = config["study"]
    subjects = args.subjects or study["subjects"]; counts = args.class_counts or study["class_counts"]
    if args.gain_mode:
        config["resonate_fire"]["gain"]["mode"] = args.gain_mode
    gain_mode = config["resonate_fire"]["gain"]["mode"]
    output = ROOT / study["output"]
    if args.gain_mode:
        output = output / f"gain_{gain_mode}"
    checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    default_endpoints = tuple(float(value) for value in study.get("decision_endpoints_seconds", (.5,)))
    endpoint_plan = {
        mode: tuple(float(value) for value in study.get(
            "decision_endpoints_seconds_by_filter_mode", {},
        ).get(mode, default_endpoints))
        for mode in study["filter_modes"]
    }
    (output / "run_plan.json").write_text(json.dumps({
        "study": study["name"], "subjects": subjects, "class_counts": counts,
        "filter_modes": study["filter_modes"],
        "endpoints_seconds_by_filter_mode": endpoint_plan,
        "gain_mode": gain_mode,
        "evaluation": "outer-block nested joint encoder/ridge selection",
        "historical_matlab_reference": historical_matlab_reference(),
    }, indent=2), encoding="utf-8")
    total_cells = len(subjects) * len(counts) * sum(len(endpoint_plan[mode]) for mode in study["filter_modes"])
    completed_cells = sum(1 for subject in subjects for count in counts for mode in study["filter_modes"]
                          for endpoint in endpoint_plan[mode] if (
        checkpoints / f"subject_{subject:02d}_{count:02d}_classes_{mode}_{round(endpoint * 1000):04d}ms.npz"
    ).exists())
    append_progress(progress_path, status="running", phase="study_start", current=completed_cells,
                    total=total_cells, message="Advanced joint pilot started or resumed")
    advanced = config["advanced_optimization"]
    racing = RacingConfig(
        enabled=advanced["racing"]["enabled"], stages=tuple(advanced["racing"]["stages"]),
        confidence_z=float(advanced["racing"]["confidence_z"]),
        minimum_survivors=int(advanced["racing"]["minimum_survivors"]),
        seed=int(advanced["racing"]["seed"]),
    )
    joint = JointSearchConfig(
        screened_per_encoder=int(advanced["screened_per_encoder"]),
        beam_width=int(advanced["beam_width"]), selection_rule="one_standard_error",
        spike_cost_weight=float(advanced["objective"]["spike_cost_weight"]),
        pooling_strength=float(advanced["subject_pooling"]["strength"]),
    )
    for subject in subjects:
        raw_all = load_subject(int(subject), int(study["condition"]))
        baseline_cache = {}
        for count in counts:
            selection = study["class_selection"]
            frequencies = np.asarray(select_class_frequencies(
                int(count), available_hz=selection["available_hz"], strategy=selection["strategy"],
                interference_harmonics=selection["interference_harmonics"],
                spacing_hz=selection["spacing_hz_by_class_count"].get(int(count)),
                start_hz=selection["start_hz"],
            ), dtype=int)
            selected = raw_all[frequencies - 1]
            raw_full = selected.reshape(-1, len(BRANCH_NAMES), selected.shape[-1])
            labels = np.repeat(np.arange(count), 12); groups = np.tile(np.arange(12), count)
            for mode in study["filter_modes"]:
                endpoints = endpoint_plan[mode]
                endpoint_predictions = []; endpoint_spikes = []; endpoint_paths = []
                for endpoint in endpoints:
                    duration = round(endpoint * 1000); crop = slice(140, 140 + duration)
                    path = checkpoints / f"subject_{subject:02d}_{count:02d}_classes_{mode}_{duration:04d}ms.npz"
                    endpoint_paths.append(path)
                    if path.exists():
                        with np.load(path, allow_pickle=False) as saved:
                            endpoint_predictions.append(np.asarray(saved["predictions"])); endpoint_spikes.append(float(saved["mean_selected_spike_cost"]))
                        continue
                    raw = raw_full[..., crop]
                    prestimulus = raw_full[..., :crop.start]
                    baseline_key = (int(count), float(endpoint))
                    if baseline_key not in baseline_cache:
                        baseline_cache[baseline_key] = matched_matlab_baselines(
                            int(subject), int(study["condition"]), frequencies, float(endpoint),
                        )
                    conventional_baselines = baseline_cache[baseline_key]
                    delta, lif = filtered_candidates(
                        raw_full, frequencies, crop, mode, config["bandpass"], config["delta"], config["lif"],
                    )
                    ids, names = rf_schema(config["resonate_fire"], frequencies); fold_cache = {}
                    def build_rf(training_mask, *, raw=raw, prestimulus=prestimulus, ids=ids, names=names):
                        key = np.packbits(np.asarray(training_mask, dtype=np.uint8)).tobytes()
                        if key not in fold_cache:
                            fold_cache[key] = rf_candidates(
                                raw, frequencies, duration, config["resonate_fire"], training_mask=training_mask,
                                prestimulus=prestimulus,
                            )
                        return fold_cache[key]
                    rf = FoldSafeCandidateFeatureBlock("resonate_fire", ids, names, build_rf)
                    started = time.perf_counter()
                    result = advanced_nested_grouped_fusion(
                        (rf, delta, lif), labels, groups, l2_grid=config["fusion"]["l2_grid"],
                        candidate_selection_l2=float(config["fusion"]["candidate_selection_l2"]),
                        reference_by_block=advanced["reference_parameters"], racing=racing, joint=joint,
                        retrained_ablation_blocks=tuple(advanced["retrained_ablation_encoders"]),
                        retrained_feature_ablations={
                            f"branch:{branch}": (lambda name, branch=branch: f":{branch}:" in name)
                            for branch in advanced.get("retrained_ablation_branches", ())
                        },
                        progress_callback=lambda current, total: (
                            print(f"@@PROGRESS|advanced_nested|{current}|{total}|S{subject} {count}c {mode} {duration}ms", flush=True),
                            append_progress(
                                progress_path, status="running", phase="advanced_nested",
                                # ``current`` remains a completed-cell count.  The
                                # in-cell fraction is recorded separately, allowing
                                # dashboards to render both bars without ambiguity.
                                current=completed_cells, total=total_cells,
                                cell_current=current, cell_total=total,
                                subject=subject, class_count=count, filter_mode=mode,
                                endpoint_ms=duration,
                                message="Joint selection and retrained ablations",
                            ),
                        ),
                    )
                    mean_cost = float(np.mean([fold.inner_metrics.spike_cost for fold in result.folds]))
                    atomic_save(
                        path, subject_id=subject, class_count=count, filter_mode=mode,
                        endpoint_seconds=endpoint, frequencies_hz=frequencies, labels=labels, groups=groups,
                        predictions=result.predictions, decision_scores=result.decision_scores,
                        accuracy=result.accuracy, mean_selected_spike_cost=mean_cost,
                        fold_selections_json=json.dumps(serializable_folds(result)),
                        stability_json=json.dumps(result.stability),
                        retrained_ablation_accuracy_json=json.dumps(result.retrained_ablation_accuracy),
                        retrained_ablation_predictions_json=json.dumps({
                            name: values.tolist() for name, values in result.retrained_ablation_predictions.items()
                        }),
                        conventional_baselines_json=json.dumps(conventional_baselines),
                        gain_mode=config["resonate_fire"]["gain"]["mode"],
                        harmonic_collisions_json=json.dumps([item.__dict__ for item in harmonic_collisions(frequencies)]),
                        elapsed_seconds=time.perf_counter() - started,
                        evaluation_design="within_subject_outer_block_joint_encoder_ridge_selection_training_only_gain",
                    )
                    endpoint_predictions.append(result.predictions); endpoint_spikes.append(mean_cost)
                    completed_cells += 1
                    append_progress(
                        progress_path, status="running", phase="cell_complete",
                        current=completed_cells, total=total_cells, subject=subject,
                        class_count=count, filter_mode=mode, endpoint_ms=duration,
                        accuracy=result.accuracy, elapsed_seconds=float(time.perf_counter() - started),
                        message=path.name,
                    )
                endpoint_predictions_array = np.stack(endpoint_predictions)
                spike_matrix = np.broadcast_to(np.asarray(endpoint_spikes)[:, None], endpoint_predictions_array.shape)
                summaries = endpoint_results(
                    labels, endpoint_predictions_array, endpoints, spike_matrix, classes=int(count),
                    overhead_seconds=float(study["practical_overhead_seconds"]),
                    spike_penalty=float(advanced["objective"]["spike_cost_weight"]),
                    latency_penalty=float(advanced["objective"]["latency_weight"]),
                )
                summary_path = output / f"subject_{subject:02d}_{count:02d}_classes_{mode}_endpoints.json"
                summary_path.write_text(json.dumps({
                    "endpoints": [item.__dict__ for item in summaries],
                    "pareto_frontier": [item.__dict__ for item in pareto_endpoints(summaries)],
                    "checkpoints": [str(path.relative_to(output)) for path in endpoint_paths],
                }, indent=2), encoding="utf-8")
                print(f"complete S{subject} {count}c {mode}: {summary_path.name}", flush=True)
    append_progress(progress_path, status="complete", phase="study_complete", current=total_cells,
                    total=total_cells, message="All advanced pilot cells completed")


if __name__ == "__main__":
    main()
