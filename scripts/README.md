# Experiment scripts

Reusable algorithms, preprocessing, configuration, evaluation, and plotting
belong under `src/ssvep_toolkit`. Files in this directory are resumable research
entry points: they select a dataset slice and parameter grid, call package
functions, and write immutable checkpoints under `outputs/experiments`.

Naming conventions:

- `run_<study>.py`: long or resumable experiment
- `evaluate_<study>.py`: fixed-parameter comparison
- `plot_<study>.py`: plots existing checkpoints without recomputation
- `analyze_<study>.py`: secondary numerical analysis of existing results
- `profile_<study>.py`: performance or data-distribution audit
- `launch_*.py` / `wait_*.ps1`: process orchestration only

The current top-level names are retained while long-running jobs and documents
refer to them. New shared behavior must not be copied into another script;
promote it into the package first. A later mechanical migration can group thin
wrappers under `scripts/experiments`, `scripts/plots`, and `scripts/maintenance`
without changing output-folder names.

`run_advanced_multi_encoder_pilot.py` is the thin entry point for the joint,
fold-safe pilot. Selection, racing, latency, gain calibration, retraining, and
stability logic lives under `src/ssvep_toolkit`; this script loads data,
generates candidates, writes endpoint checkpoints, and reports progress.
