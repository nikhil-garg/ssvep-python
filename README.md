# SSVEP Spike Encoding

Python tools for the 1–60 Hz SSVEP dataset, with conventional SSVEP baselines
and three interchangeable spike encoders: resonate-and-fire (R&F), delta, and
leaky integrate-and-fire (LIF). The scientific target is subject-wise,
latency-aware classification with validation-safe multi-encoder fusion.

The repository name should be `ssvep-spike-encoding`. Raw EEG and generated
results deliberately remain outside version control.

## Install

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,metadata,analysis]"
```

Install the optional desktop GUI with:

```powershell
python -m pip install -e ".[gui]"
```

## Main interfaces

```powershell
# Inspect and preprocess MATLAB/HDF5 data
ssvep inspect --data-dir "C:\path\to\SSVEP dataset 1-60Hz"
ssvep preprocess --config configs\default.yaml --dry-run

# Encode target-frequency spike streams
ssvep encode-spikes resonate_fire --input eeg.npy --output rf.npz `
  --frequencies 8 9 10 11 --sampling-rate 1000 --threshold 0.01
ssvep encode-spikes delta --input eeg.npy --output delta.npz `
  --frequencies 8 9 10 11 --sampling-rate 1000 --threshold 0.2
ssvep encode-spikes lif --input eeg.npy --output lif.npz `
  --frequencies 8 9 10 11 --sampling-rate 1000 --threshold 0.5 --tau 0.02

# Index experiment checkpoints and build the local dashboard
ssvep registry import-checkpoints outputs\experiments\individual_spike_encoders
ssvep dashboard --output outputs\dashboard\index.html

# Render raw/filtered EEG, internal states, thresholds, and spikes
ssvep example-neuron --data-dir .. --subject 1 --frequency 16 --block 4 `
  --electrode O1 --output outputs\examples\neuron_behavior\s01_16hz_b04_o1.png

# Optional desktop workbench
ssvep gui
```

The generated dashboard keeps nested outer-test evidence, perturbed robustness
tests, and apparent same-data exploration in separate views with explicit
validation warnings. It summarizes every available metric with units, provides
an interactive primary chart and metric table, and displays accuracy, ITR,
latency, robustness, spike cost, runtime, and experiment progress. Signal views
combine scalable downsampled traces with exact spike-event times and links to
the full-resolution PNG evidence. Continuous traces are capped for browser
performance, while the run table is filtered and paginated rather than inserted
into the page all at once.

The experiment tab exposes stimulus start/spacing, subject, 4/16-class task,
decision duration, condition, causal/offline filtering, band-pass settings, R&F
damping/threshold/gain/harmonic/spread grids, delta thresholds/asymmetry, LIF
threshold/tau/gain, and the ridge-fusion grid. Every GUI launch writes a unique
YAML configuration and output directory before processing, preventing
incompatible parameter runs from sharing checkpoints. Neuron figures are scaled
to the available preview area while retaining their saved resolution and render
in a separate process so the interface remains responsive. The experiment tab
also shows the three most recently updated experiment directories.

Long operations do not run on the GUI thread. Neuron-figure rendering,
experiments, dashboard generation, recursive experiment scans, and confirmed
cleanup each have their own progress indicator. Candidate generation, target
filtering, and nested fusion report determinate phase progress; data loading,
compression, and storage scans use an indeterminate busy indicator when a safe
total is not known. Buttons are disabled only for the conflicting operation,
and process failures restore the controls instead of leaving a permanent busy
state.

Offline filtering is a zero-phase analysis benchmark: it uses samples on both
sides of a time point and therefore cannot be deployed causally. Causal filtering
uses only present and past samples and is the real-time result. Both should be
reported, but latency/ITR claims must use the causal result.

Expensive encoder features are cached atomically with a configuration fingerprint
and reused after interruption. Completed compact checkpoints retain predictions,
scores, fold selections, metrics and distribution quantiles; temporary feature
caches are removed after all requested modes finish. GUI cleanup is conservative:
it can remove confirmed GUI runs older than 30 days while retaining a chosen
number of recent runs, plus stale partial files. Legacy scientific experiments
are never removed automatically.

R&F encoding is unfiltered by default and retains 1000 Hz EEG sampling. Time is
normalized by resonance frequency, while dividing the input drive by resonance
frequency is now a separate opt-in compatibility switch. Raw spike counts are
never divided by resonance frequency. Delta
and LIF use one independently filtered channel per requested stimulus frequency;
their default fifth-order band is `f ± 1 Hz`. Delta produces distinct UP and DN
streams. For a 32-class task, each EEG branch therefore drives 32 target filters
and 32 delta or LIF encoders. Filtering must be applied to the full source epoch
before cropping the decision window.

## Validation-safe multi-encoder study

The main study configuration is
[`configs/nested_multi_encoder.yaml`](configs/nested_multi_encoder.yaml). Run a
small cell first:

```powershell
python scripts\run_nested_multi_encoder.py --subjects 1 --class-counts 4
```

Remove those overrides for all 30 subjects. The primary study targets four
classes at 4 Hz spacing and sixteen classes at 2 Hz spacing. The starting
frequency minimizes exact 2x/3x class collisions; selected frequencies and any
remaining collisions are saved in every checkpoint. The current sets are
17/21/25/29 Hz and 17/19/.../47 Hz, respectively, so the 4-class set is a
controlled subset of the 16-class set.

R&F uses fundamental neurons by default. Responses from every target neuron
remain separate fusion features, so an 8 Hz trial exciting a 16 Hz neuron is a
learnable cross-target pattern rather than an automatic vote for the 16 Hz
class. Each subject uses block-wise nested validation:

1. One block is held out as the untouched outer test set.
2. R&F, delta, and LIF parameter candidates are selected using only inner folds.
3. Candidate features retain encoder, target, channel/branch, and spike-stream identity.
4. Grouped ridge strength is selected inside the outer training set.
5. The fitted fusion model predicts the held-out block exactly once.
6. Offline zero-phase and deployable causal filters are evaluated separately.

Outputs report accuracy, neural-window and practical ITR, decision latency,
spikes per trial, spikes per correct selection, block variability, parameter
selections, and the fold-selected per-segment/per-branch gain distribution.
Nonlinear fusion is intentionally deferred until the linear outer-block result
is established.

### Parameter optimization and bounds

Encoder candidates are chosen independently for every subject and outer fold
using only the remaining inner blocks. The search uses a predeclared broad,
log-spaced pool and the one-standard-error rule: candidates within one standard
error of the best inner mean are treated as statistically equivalent, then the
one nearest a declared physiological reference point is selected. Ridge fusion
uses the same rule and prefers stronger regularization among equivalent values.
This is less sensitive to a one-trial validation fluctuation than maximizing a
large grid directly.

Redundant scale axes are removed before searching. R&F operating RMS is fixed
while the identifiable drive-to-threshold ratio is varied; LIF input gain is
fixed while threshold is varied. Delta thresholds remain absolute µV values so
real segment and channel amplitude differences are retained. Current bounds,
their rationale, and reference points are declared in
[`configs/nested_multi_encoder.yaml`](configs/nested_multi_encoder.yaml).

Every checkpoint records the evaluated inner-fold accuracy landscape (with
explicit missing entries for candidates pruned by multi-fidelity search), selected
candidate, standard error, eligible one-SE set, per-parameter lower/upper-bound
hits, pruning stages, and ridge-boundary status. Boundary rates are also registered as
`inner_validation` metrics. Bounds are never changed during a confirmatory run:
an edge is proposed for fourfold log expansion in the next locked pilot only if
it is selected in at least 25% of inner selections and adjacent validation
values do not already decline toward that edge.

## Joint, gain-safe endpoint pilot

The additive next-pilot configuration is
[`configs/nested_multi_encoder_joint_pilot.yaml`](configs/nested_multi_encoder_joint_pilot.yaml).
It does not alter the locked 30-subject confirmatory study. Launch a single
smoke cell with:

```powershell
python scripts\run_advanced_multi_encoder_pilot.py --subjects 1 --class-counts 4
```

The advanced evaluator screens candidates with confidence-bound racing, retains
statistically overlapping candidates, and uses a beam search to select R&F,
delta, LIF, and ridge parameters jointly. Outer-test blocks remain untouched.
Accuracy is the primary inner objective; one-SE-equivalent solutions are ranked
by classification margin, softmax log loss, spike cost, and the pooled reference.

R&F gain is fitted from outer-training trials only, once per subject branch,
which preserves held-out trial amplitudes. `--gain-mode prestimulus` and
`--gain-mode causal_running` create isolated output subdirectories for the two
deployable alternatives. The running estimator applies every sample using its
previous mean/RMS state. The pilot evaluates 200/300/400/500 ms endpoints and
saves accuracy, practical ITR, spike cost, scalar utility, and the accuracy /
latency / spike-cost Pareto frontier.

Every outer fold also repeats the complete selection and training procedure
after removing each encoder and each O1/Oz/O2/bipolar branch. These retrained
ablations answer a different question from test-time feature zeroing and are
substantially more expensive. Parameter stability, racing histories, joint
beams, selected ridge values, and the effective subject/population references
are retained in the checkpoints.

Normalized R&F damping can be expressed as approximate bandwidth using
`bandwidth_hz`, `damping_from_bandwidth`, and `quality_factor`. Intrinsic neuron
bandwidth remains distinct from the bank's `spread_hz` offsets.

## Transparent next-generation study path

The revised runner implements five additional improvements:

1. **Structured processing provenance.** Every run creates `run_plan.json` and
   an append-only `progress.jsonl` containing the configuration SHA-256,
   dataset file manifest, candidate/cell counts, phase timing, completion
   fraction, ETA, process identity, and final status. The GUI shows the planned
   workload before launch; the dashboard shows the latest phase and ETA.
2. **Explicit causal state.** Causal SOS filters process the 140 ms onset prefix
   first, carry their final state into the decision window, and save this fact
   in each checkpoint. Chunked and one-shot causal filtering are tested for
   numerical equality. Offline filtering is separately marked as using future
   samples.
3. **Validation-safe multi-fidelity pruning.** Within each outer fold, all
   encoder candidates start on four inner blocks, half continue through eight,
   and half of those continue through all eleven. The deterministic fold order,
   candidates evaluated/retained at each stage, and model-fit count are saved.
   No outer-block observation participates in pruning.
4. **Factorial class design.** Class count, spacing, starting frequency,
   frequency span, and harmonic-collision count are explicit factors. The run
   plan enumerates valid predeclared combinations instead of silently changing
   spacing when the number of classes changes.
5. **Temporal features and efficient ablations.** R&F rate, normalized TTFS,
   cosine phase, and sine phase remain separate named features. O1, Oz, O2,
   O1-Oz, O2-Oz and R&F/delta/LIF ablations are evaluated from the same fitted
   outer-fold models, avoiding repeated parameter selection. These are clearly
   labelled test-time dependence analyses, not retrained-model accuracy.

## Interpreting results

Older exploratory checkpoints may report `apparent_same_data` accuracy because
parameter selection and scoring reused the same segments. These are useful for
neuron-behaviour studies but are not publishable generalization estimates. The
dashboard preserves this label. Only metrics marked `outer_test` from the nested
runner should be used as final classification evidence.

ITR is reported two ways:

- neural-window ITR uses decision time plus the 140 ms visual-onset latency;
- practical ITR additionally includes configurable gaze/command overhead.

This distinction tests whether spike encoding is most useful at 250–500 ms even
when a longer window gives slightly higher accuracy.

## Repository layout

```text
configs/                 versioned experiment and preprocessing YAML
docs/                    architecture, audit, and scientific decisions
scripts/                 resumable study entry points and figure builders
src/ssvep_toolkit/
  algorithms/            classifiers, encoders, fusion, and ITR
  preprocessing/         filtering, epoching, and downsampling
  evaluation/            nested validation and comparable reports
  registry/              SQLite run/metric/artifact index
  visualization/         reusable evidence and paper-figure functions
tests/                   unit and integration tests
outputs/                 ignored checkpoints, figures, registry, and dashboard
```

The core package contains reusable code; `scripts/` contains named experiments.
Do not add ambiguous folders such as `paper_figures`. Use a study-specific path,
for example `outputs/experiments/nested_multi_encoder/figures/`.

See [`docs/project_development_consolidation.md`](docs/project_development_consolidation.md)
for experiment history and known biases, and
[`docs/encoder_architecture_and_gui_plan.md`](docs/encoder_architecture_and_gui_plan.md)
for the application architecture.
