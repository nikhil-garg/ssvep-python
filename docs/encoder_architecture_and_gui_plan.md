# Encoder architecture and GUI plan

## Stable pipeline

All encoders follow the same conceptual stages:

1. Load epochs as `(trial, EEG channel, sample)`.
2. Apply encoder-specific preprocessing.
3. Encode each EEG channel independently.
4. Aggregate or retain channel/stream spike features.
5. Calibrate class templates and classify.
6. Optionally fuse independently optimized encoders.

R&F, delta, and LIF are selected through `EncoderConfig`. R&F leaves band-pass
filtering off by default. Delta and LIF create one order-5 `[f-1, f+1] Hz` band
per target by default.

For 32 classes with O1/Oz/O2:

- LIF raster: `(32 targets, trials, 3 EEG channels, 1 stream, samples)`;
  96 neuron instances. The initial classifier sums EEG-channel counts into 32
  features.
- Delta raster: `(32 targets, trials, 3 EEG channels, 2 streams, samples)`;
  192 UP/DN comparator streams. The initial classifier sums EEG channels but
  retains polarity, producing 64 features.
- R&F counts: `(trials, target/resonator features)` after its oscillator bank
  combines independently simulated EEG-channel responses.

Channel aggregation must remain a named policy rather than hidden behavior:
`sum`, `retain`, or `learned_weights`. Individual encoder searches currently
use `sum`; learned channel and encoder weights belong to the later fusion stage.

## Code ownership

- `algorithms/encoding.py`: uniform public encoder façade and feature result
- `algorithms/spike_encoding.py`: low-level delta and LIF raster dynamics
- `algorithms/resonate_and_fire/`: low-level R&F dynamics and classifier
- `preprocessing/bandpass.py`: general target-frequency filtering
- `evaluation/spike_encoder_experiment.py`: reusable count features and scoring
- `scripts/run_*.py`: only dataset selection, grids, checkpoints, and progress
- `configs/*.yaml`: editable experiment definitions

Existing top-level scripts remain in place while active runs and documentation
refer to them. They should be moved only as a mechanical migration after the
long-running experiments finish.

## GUI structure

The first GUI should edit and launch configurations, not duplicate algorithms.
Suggested panels:

1. Dataset: subjects, class counts/frequencies, channels, condition, duration.
2. Encoder: R&F / delta / LIF / later fusion.
3. Preprocessing: band-pass toggle, order, half-width, zero-phase/causal.
4. Encoder parameters:
   - R&F: damping, threshold, gain, harmonics, spread, solver.
   - Delta: threshold and UP/DN asymmetry.
   - LIF: threshold, tau/leakage, input gain.
5. Feature aggregation: sum, retain, or learned channel weights.
6. Evaluation: apparent, held-out trials, subject-wise, or cross-subject.
7. Execution: dry run, estimated workload, resume, progress, logs, cancel.
8. Results: accuracy, confusion, parameter distributions, spike rasters, and
   internal-state examples.

The GUI should serialize exactly the same YAML consumed by command-line runs.
This keeps runs reproducible and allows GUI and batch execution to coexist.
