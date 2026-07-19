from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ssvep_toolkit.config import ConfigError, load_config
from ssvep_toolkit.data import inspect_dataset
from ssvep_toolkit.runners import describe_preprocessing, run_preprocessing


def _size(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ssvep", description="SSVEP dataset processing toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_cmd = sub.add_parser("inspect", help="inspect dataset files and HDF5 dimensions")
    inspect_cmd.add_argument("--data-dir", required=True, type=Path)
    inspect_cmd.add_argument("--fast", action="store_true", help="do not open each HDF5 file")
    inspect_cmd.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    pre = sub.add_parser("preprocess", help="run chunked preprocessing")
    pre.add_argument("--config", required=True, type=Path)
    pre.add_argument("--subjects", nargs="+", type=int)
    pre.add_argument("--frequencies", nargs="+", type=int)
    pre.add_argument("--conditions", nargs="+", type=int)
    pre.add_argument("--dry-run", action="store_true")
    plot = sub.add_parser("plot", help="create plots from preprocessed HDF5 data")
    plot_sub = plot.add_subparsers(dest="plot_command", required=True)
    spectrum = plot_sub.add_parser("spectrum", help="plot amplitude spectrum and SNR")
    spectrum.add_argument("--input", required=True, type=Path)
    spectrum.add_argument("--condition-index", type=int, default=1)
    spectrum.add_argument("--frequency-index", type=int, default=1)
    spectrum.add_argument("--maximum-hz", type=float, default=100.0)
    spectrum.add_argument("--output", required=True, type=Path)
    analyze = sub.add_parser("analyze", help="run SSVEP classification")
    analyze.add_argument("algorithm", choices=("fbcca", "trca", "fbtrca"))
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--duration", type=float, default=1.0)
    analyze.add_argument("--first-low-hz", type=float, default=6.0)
    analyze.add_argument("--subbands", type=int, default=5)
    analyze.add_argument("--weight-a", type=float, default=1.25)
    analyze.add_argument("--weight-b", type=float, default=0.25)
    analyze.add_argument("--rest-seconds", type=float, default=1.0)
    encode = sub.add_parser("encode-spikes", help="create target-frequency delta or LIF spike streams")
    encode.add_argument("encoder", choices=("resonate_fire", "delta", "lif"))
    encode.add_argument("--input", required=True, type=Path, help=".npy array or .npz containing 'data'; time is last")
    encode.add_argument("--output", required=True, type=Path)
    encode.add_argument("--frequencies", nargs="+", required=True, type=float)
    encode.add_argument("--sampling-rate", type=float, default=1000.0)
    encode.add_argument("--threshold", type=float, required=True)
    encode.add_argument("--asymmetry", type=float, default=1.0,
                        help="delta DN threshold multiplier; UP threshold is --threshold")
    encode.add_argument("--tau", type=float, default=0.02, help="LIF membrane time constant in seconds")
    encode.add_argument("--input-gain", type=float, default=1.0)
    encode.add_argument("--bandpass", action=argparse.BooleanOptionalAction, default=None,
                        help="target band bank; defaults off for R&F and on for delta/LIF")
    encode.add_argument("--bandpass-order", type=int, default=5)
    encode.add_argument("--bandpass-half-width", type=float, default=1.0)
    encode.add_argument("--causal-filter", action="store_true",
                        help="use causal filtering instead of zero-phase offline filtering")
    encode.add_argument("--damping-alpha", type=float, default=0.3)
    encode.add_argument("--normalize-input-by-resonance", action=argparse.BooleanOptionalAction, default=False,
                        help="optionally divide normalized-time R&F input drive by resonance frequency")
    encode.add_argument("--harmonics", nargs="+", type=int, default=(1,))
    encode.add_argument("--spread-hz", nargs="+", type=float, default=(0.0,))
    encode.add_argument("--integration-substeps", type=int, default=4)
    encode.add_argument("--refractory-cycles", type=float, default=0.5)
    figure = sub.add_parser("figure", help="render a paper figure from an NPZ result")
    figure.add_argument("number", type=int, choices=range(4, 14))
    figure.add_argument("--input", required=True, type=Path)
    figure.add_argument("--output", required=True, type=Path)
    experiment = sub.add_parser("experiment", help="run experimental classifiers")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    rf = experiment_sub.add_parser("resonate-fire", help="run the subject-grouped oscillator-bank experiment")
    rf.add_argument("--raw-dir", required=True, type=Path)
    rf.add_argument("--output-dir", required=True, type=Path)
    rf.add_argument("--frequencies", nargs="+", type=int, default=tuple(range(1, 61)))
    rf.add_argument("--condition", type=int, choices=(1, 2), default=2)
    rf.add_argument("--spread-hz", nargs="+", type=float, default=(-0.5, 0.0, 0.5))
    rf.add_argument("--harmonics", nargs="+", type=int, default=(1, 2, 3))
    rf.add_argument("--integration-substeps", type=int, default=4,
                    help="internal Euler steps per 1 ms EEG sample")
    rf.add_argument("--refractory-cycles", type=float, default=0.5,
                    help="minimum reset interval measured in each neuron's cycles")
    registry = sub.add_parser("registry", help="inspect the shared experiment-run registry")
    registry.add_argument("--database", type=Path, default=Path("outputs/registry/experiments.sqlite3"))
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    registry_sub.add_parser("init", help="initialize an empty registry")
    registry_sub.add_parser("summary", help="summarize studies, runs, and metrics")
    registry_list = registry_sub.add_parser("list", help="list registered runs")
    registry_list.add_argument("--study-id", type=int)
    registry_list.add_argument("--json", action="store_true")
    registry_import = registry_sub.add_parser("import-checkpoints", help="index legacy NPZ checkpoint files")
    registry_import.add_argument("root", type=Path)
    registry_import.add_argument("--study-name")
    dashboard = sub.add_parser("dashboard", help="build a portable HTML dashboard from the registry")
    dashboard.add_argument("--database", type=Path, default=Path("outputs/registry/experiments.sqlite3"))
    dashboard.add_argument("--output", type=Path, default=Path("outputs/dashboard/index.html"))
    dashboard.add_argument("--examples", type=Path, default=Path("outputs/examples/neuron_behavior"))
    example = sub.add_parser("example-neuron", help="plot real EEG, encoder states, and output spikes")
    example.add_argument("--data-dir", type=Path, required=True)
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--subject", type=int, default=1)
    example.add_argument("--condition", type=int, choices=(1, 2), default=2)
    example.add_argument("--frequency", type=int, choices=range(1, 61), default=8)
    example.add_argument("--block", type=int, choices=range(1, 13), default=1)
    example.add_argument("--electrode", choices=("O1", "Oz", "O2", "O1-Oz", "O2-Oz"), default="Oz")
    example.add_argument("--start-ms", type=int, default=140)
    example.add_argument("--duration-ms", type=int, default=1000)
    example.add_argument("--encoders", nargs="+", choices=("resonate_fire", "delta", "lif"),
                         default=("resonate_fire", "delta", "lif"))
    sub.add_parser("gui", help="launch the optional PySide desktop workbench")
    return parser


def _inspect(args: argparse.Namespace) -> int:
    inventory = inspect_dataset(args.data_dir, inspect_hdf5=not args.fast)
    if args.json:
        print(json.dumps({
            "root": str(inventory.root),
            "total_bytes": inventory.total_bytes,
            "subjects": [
                {
                    "subject": item.subject,
                    "path": str(item.path),
                    "bytes": item.bytes,
                    "logical_shape": item.logical_shape,
                    "dtype": item.dtype,
                    "error": item.error,
                }
                for item in inventory.subjects
            ],
            "metadata_files": [str(x) for x in inventory.metadata_files],
        }, indent=2))
        return 0
    print(f"Dataset: {inventory.root}")
    print(f"Subjects: {len(inventory.subjects)}; EEG size: {_size(inventory.total_bytes)}")
    for item in inventory.subjects:
        detail = f"shape={item.logical_shape}, dtype={item.dtype}" if item.logical_shape else "not opened"
        if item.error:
            detail = item.error
        print(f"  S{item.subject:02d}  {_size(item.bytes):>10}  {detail}")
    print("Metadata:")
    for path in inventory.metadata_files:
        kind = "Excel workbook with .csv extension" if path.read_bytes()[:4] == b"PK\x03\x04" else path.suffix
        print(f"  {path.name}: {kind}")
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.subjects:
        config.dataset.subjects = args.subjects
    if args.frequencies:
        config.dataset.frequencies = args.frequencies
    if args.conditions:
        config.dataset.conditions = args.conditions
    config.validate()
    plan = describe_preprocessing(config)
    print(f"Subjects: {plan.subjects}; trials: {plan.trials}; channels: {plan.channels}")
    print(f"Samples per trial: {plan.input_samples} -> {plan.output_samples}")
    print(f"Estimated output: {_size(plan.estimated_output_bytes)}")
    print(f"Output directory: {config.output.root}")
    if args.dry_run:
        print("Dry run complete; no output was written.")
        return 0
    outputs = run_preprocessing(config)
    for path in outputs:
        print(f"Created: {path}")
    return 0


def _plot(args: argparse.Namespace) -> int:
    if args.plot_command != "spectrum":
        raise ValueError(f"unsupported plot command: {args.plot_command}")
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("plotting requires the project dependencies") from exc
    from ssvep_toolkit.features import amplitude_spectrum, signal_to_noise_ratio
    from ssvep_toolkit.visualization import plot_spectrum

    with h5py.File(args.input, "r") as source:
        ci = args.condition_index - 1
        fi = args.frequency_index - 1
        if not 0 <= ci < source["data"].shape[0] or not 0 <= fi < source["data"].shape[3]:
            raise ValueError("condition-index or frequency-index is outside the file")
        # (channel, sample, block) -> (block, channel, sample)
        data = np.asarray(source["data"][ci, :, :, fi, :]).transpose(2, 0, 1)
        sampling_rate = float(source.attrs["sampling_rate_hz"])
        stimulus = float(source.attrs["frequencies_hz"][fi])
    frequencies, amplitude = amplitude_spectrum(data, sampling_rate)
    snr = signal_to_noise_ratio(amplitude)
    plot_spectrum(
        frequencies,
        amplitude,
        snr_db=snr,
        stimulus_hz=stimulus,
        maximum_hz=args.maximum_hz,
        title=f"{stimulus:g} Hz stimulation",
        output=args.output,
    )
    print(f"Created: {args.output.resolve()}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from ssvep_toolkit.runners import run_classification

    output = run_classification(
        args.input,
        args.output,
        args.algorithm,
        duration_seconds=args.duration,
        first_low_hz=args.first_low_hz,
        subbands=args.subbands,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
        rest_seconds=args.rest_seconds,
    )
    print(f"Created: {output.resolve()}")
    return 0


def _figure(args: argparse.Namespace) -> int:
    from ssvep_toolkit.visualization import load_reference_study_figure_data, render_reference_study_figure

    render_reference_study_figure(args.number, load_reference_study_figure_data(args.input), args.output)
    print(f"Created: {args.output.resolve()}")
    return 0


def _encode_spikes(args: argparse.Namespace) -> int:
    import numpy as np
    from ssvep_toolkit.algorithms import (
        DeltaEncoderParameters, EncoderConfig, LIFEncoderParameters,
        encode_spike_features, encode_target_frequency_bank,
    )
    from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters
    from ssvep_toolkit.preprocessing import BandpassParameters

    if args.input.suffix.lower() == ".npy":
        data = np.load(args.input)
    elif args.input.suffix.lower() == ".npz":
        with np.load(args.input) as source:
            if "data" not in source:
                raise KeyError("NPZ input must contain an array named 'data'")
            data = np.asarray(source["data"])
    else:
        raise ValueError("encode-spikes input must be .npy or .npz")
    if data.ndim == 1:
        data = data[None, None, :]
    elif data.ndim == 2:
        data = data[:, None, :]
    elif data.ndim != 3:
        raise ValueError("encoder input must be sample, trial-by-sample, or trial-by-channel-by-sample")
    bandpass_enabled = args.encoder != "resonate_fire" if args.bandpass is None else args.bandpass
    bandpass = BandpassParameters(
        enabled=bandpass_enabled, order=args.bandpass_order,
        half_width_hz=args.bandpass_half_width, zero_phase=not args.causal_filter,
    )
    if args.encoder == "resonate_fire":
        result = encode_spike_features(
            data, args.frequencies, args.sampling_rate,
            EncoderConfig(
                kind="resonate_fire", bandpass=bandpass,
                resonate_fire=ResonateAndFireParameters(
                    damping_alpha=args.damping_alpha, threshold=args.threshold,
                    input_gain=args.input_gain, integration_substeps=args.integration_substeps,
                    normalize_input_by_resonance=args.normalize_input_by_resonance,
                    refractory_cycles=args.refractory_cycles, solver="exact",
                    reset_mode="zero", spike_detection="upward_crossing",
                ),
                harmonics=tuple(args.harmonics), spread_hz=tuple(args.spread_hz),
            ),
        )
        spikes = None
        counts = np.asarray(result.counts)
        stream_names = np.array(result.stream_names)
    else:
        delta = DeltaEncoderParameters(args.threshold, args.asymmetry) if args.encoder == "delta" else None
        lif = LIFEncoderParameters(args.threshold, args.tau, args.input_gain) if args.encoder == "lif" else None
        spikes = encode_target_frequency_bank(
            data, args.frequencies, args.sampling_rate, encoder=args.encoder,
            delta_parameters=delta, lif_parameters=lif, bandpass=bandpass,
        )
        stream_counts = spikes.sum(axis=(-3, -1)).transpose(1, 0, 2)
        counts = stream_counts.reshape(stream_counts.shape[0], -1)
        stream_names = np.array(("UP", "DN") if args.encoder == "delta" else ("LIF",))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, frequencies_hz=np.asarray(args.frequencies), counts=counts,
        sampling_rate_hz=args.sampling_rate, encoder=args.encoder, stream_names=stream_names,
        threshold=args.threshold, asymmetry=args.asymmetry, tau_seconds=args.tau,
        input_gain=args.input_gain, bandpass_enabled=bandpass_enabled,
        bandpass_order=args.bandpass_order, bandpass_half_width_hz=args.bandpass_half_width,
        zero_phase=not args.causal_filter, damping_alpha=args.damping_alpha,
        normalize_input_by_resonance=args.normalize_input_by_resonance,
        harmonics=np.asarray(args.harmonics), spread_hz=np.asarray(args.spread_hz),
        **({"spikes": spikes} if spikes is not None else {}),
    )
    shape = spikes.shape if spikes is not None else counts.shape
    print(f"Created: {args.output.resolve()} shape={shape}")
    return 0


def _experiment(args: argparse.Namespace) -> int:
    if args.experiment_command != "resonate-fire":
        raise ValueError("unsupported experiment")
    import numpy as np
    from ssvep_toolkit.evaluation import load_raw_resonate_and_fire_data, run_grouped_resonate_and_fire_experiment
    from ssvep_toolkit.visualization import render_resonate_and_fire_suite

    inputs = sorted(args.raw_dir.glob("data_s*_64.mat"), key=lambda p: int(p.stem.split("s")[1].split("_")[0]))
    if not inputs:
        raise FileNotFoundError("no data_s*_64.mat files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frequency_tag = f"{min(args.frequencies)}-{max(args.frequencies)}hz_{len(args.frequencies)}classes"
    cache = args.output_dir / f"cache/raw_o1_oz_o2_1000hz_{frequency_tag}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache) as source:
            data, sampling_rate = source["data"], float(source["sampling_rate_hz"])
    else:
        data, sampling_rate = load_raw_resonate_and_fire_data(inputs, args.frequencies, args.condition)
        np.savez_compressed(cache, data=np.asarray(data, dtype=np.float32), sampling_rate_hz=sampling_rate,
                            frequencies_hz=np.asarray(args.frequencies), condition=args.condition)
    result = run_grouped_resonate_and_fire_experiment(
        data, sampling_rate, args.frequencies, args.output_dir / "results/grouped_5fold_nested_parameters.npz",
        spread_hz=args.spread_hz, harmonics=args.harmonics, integration_substeps=args.integration_substeps,
        refractory_cycles=args.refractory_cycles,
    )
    render_resonate_and_fire_suite(result, args.output_dir / "figures", raw_data=data)
    print(f"Created: {result.resolve()}")
    return 0


def _registry(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from ssvep_toolkit.registry import ExperimentRegistry

    registry = ExperimentRegistry(args.database).initialize()
    if args.registry_command == "init":
        print(f"Initialized: {args.database.resolve()}")
        return 0
    if args.registry_command == "summary":
        print(json.dumps(registry.summary(), indent=2))
        return 0
    if args.registry_command == "import-checkpoints":
        from ssvep_toolkit.registry import import_npz_checkpoints
        print(json.dumps(import_npz_checkpoints(registry, args.root, study_name=args.study_name), indent=2))
        return 0
    runs = registry.list_runs(args.study_id)
    if args.json:
        print(json.dumps([asdict(run) for run in runs], indent=2))
        return 0
    if not runs:
        print("No runs registered.")
        return 0
    print("ID  STUDY  STATUS     SUBJECT  CLASSES  FOLD  ENCODER          KEY")
    for run in runs:
        subject = "-" if run.subject_id is None else str(run.subject_id)
        classes = "-" if run.class_count is None else str(run.class_count)
        fold = "-" if run.outer_fold is None else str(run.outer_fold)
        encoder = run.encoder or "-"
        print(f"{run.id:<3} {run.study_id:<6} {run.status:<10} {subject:<8} {classes:<8} "
              f"{fold:<5} {encoder:<16} {run.run_key}")
    return 0


def _dashboard(args: argparse.Namespace) -> int:
    from ssvep_toolkit.dashboard import render_dashboard

    output = render_dashboard(args.database, args.output, example_directory=args.examples)
    print(f"Created: {output.resolve()}")
    return 0


def _example_neuron(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from ssvep_toolkit.visualization import NeuronExampleConfig, render_neuron_example

    output = render_neuron_example(
        args.data_dir,
        NeuronExampleConfig(
            subject=args.subject, condition=args.condition, frequency_hz=args.frequency,
            block=args.block, electrode=args.electrode, start_ms=args.start_ms,
            duration_ms=args.duration_ms,
        ),
        args.output, encoders=args.encoders,
    )
    print(f"Created: {output.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "preprocess":
            return _preprocess(args)
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "figure":
            return _figure(args)
        if args.command == "encode-spikes":
            return _encode_spikes(args)
        if args.command == "experiment":
            return _experiment(args)
        if args.command == "registry":
            return _registry(args)
        if args.command == "dashboard":
            return _dashboard(args)
        if args.command == "example-neuron":
            return _example_neuron(args)
        if args.command == "gui":
            from ssvep_toolkit.gui import launch_gui
            return launch_gui()
        return _plot(args)
    except (ConfigError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
