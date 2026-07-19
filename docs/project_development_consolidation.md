# SSVEP spike-encoding project consolidation

## Current answer: is there a linear classifier?

There are now two distinct linear mechanisms:

1. The running R&F five-channel experiment uses a constrained linear score
   fusion. It searches nonnegative simplex weights over O1, Oz, O2, O1-Oz, and
   O2-Oz calibrated score matrices. This is linear but optimistic because the
   weights and accuracy use the same segments.
2. `algorithms/linear_fusion.py` provides a general standardized multiclass
   ridge classifier for arbitrary named R&F, delta, LIF, channel, rate, TTFS,
   and phase feature blocks. Its regularization can be selected with grouped
   inner validation. This is the intended later encoder-fusion layer.

The general classifier must not be used to report final accuracy until encoder
parameters, scaling, feature selection, and ridge regularization are nested
inside training folds.

## Development sequence

### 1. Dataset and MATLAB translation

- Audited 30 MATLAB/HDF5 subject files, electrode metadata, sampling rate, and
  physical amplitude scale.
- Created a Python package, configuration loader, CLI, chunked data access,
  preprocessing, spectral features, reference-study classifiers, plots, and
  tests.
- Preserved MATLAB-compatible paths where reproduction required them while
  separating reusable library code from paper-specific experiments.

### 2. Initial R&F translation

- Analyzed the original notebook and reproduced its resonate-and-fire dynamics.
- Established normalized time `tau = f_res * t`; resonance frequency therefore
  scales the time axis while normalized damping remains comparable across
  frequencies.
- Replaced abrupt legacy level/reset behavior with upward threshold crossing
  and zero reset.
- Added exact integration for piecewise-constant input and verified solver
  convergence against Euler/substep variants.
- Added spike rate, first-spike latency, phase, interval, internal-state, raster,
  resonance-width, and overlap diagnostics.

### 3. Class scaling and population coding

- Progressed from 2-class demonstrations to 2/4/8/16/32-class problems.
- Added 1/2/4/8 spread voters, frequency offsets, harmonics, voting, and
  class-template scoring.
- Used rounded equally spaced targets from 8 through 39 Hz for the class-scaling
  studies.
- Tested 0.5–5 second windows, then emphasized 250/500/750/1000 ms endpoints.

### 4. Subject-specific amplitude and neuron tuning

- Stopped normalizing all segments to one common amplitude.
- Measured raw segment/channel RMS in inferred microvolts and applied adaptive
  per-segment/channel gain to a selected operating RMS.
- Ran subject-specific apparent parameter searches over damping, threshold,
  operating RMS, harmonics, and resonance spread.
- Generated 12 representative real-EEG state/spike cases spanning strong,
  medium, weak, correct, incorrect, and parameter-boundary conditions.

### 5. Spatial references and fusion

- Confirmed that the main R&F classifier used O1/Oz/O2; Oz-only traces were
  visualization examples.
- Tested O1-Oz and O2-Oz. Bipolar replacement reduced population mean but
  improved selected subjects, especially at higher class counts.
- Began a subject-wise five-channel fusion search with separately tuned bipolar
  parameters and nonnegative score weights.

### 6. Delta and LIF encoders

- Added target-specific order-5 Butterworth bands, default `[f-1, f+1] Hz`.
- Added delta UP/DN threshold crossings with a configurable asymmetry ratio.
- Added exact-discrete leaky integrate-and-fire encoding with threshold, tau,
  input gain, crossing, and reset.
- Unified R&F, delta, and LIF behind one `EncoderConfig` and `encode-spikes`
  command.
- Preserved UP/DN as separate delta features.
- Corrected the first delta/LIF run: filtering a cropped one-second epoch caused
  strong edge effects. Invalid checkpoints were archived; the corrected
  exploratory run filters 5.14 seconds before cropping.

## Current parameter spaces

### R&F deep subject-wise search

| Parameter | Values/bounds |
|---|---|
| Normalized damping alpha | 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4 |
| Threshold | 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 |
| Operating RMS | 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5 |
| Harmonics | `(1)`, `(1,2)`, `(1,2,3)` |
| Harmonic weights | `1/h` |
| Spread | `0`; `±0.25 Hz`; `-0.5,-0.167,0.167,0.5 Hz` |
| Fixed input gain | 0.8 after adaptive amplitude scaling |
| Refractory period | 0.5 resonance cycles |
| Solver | exact, four integration substeps configured |
| Spike/reset | upward crossing, reset to zero |

The median selected alpha in the complete deep search is 0.005, the lower
search boundary. Twenty-nine of 30 subjects selected that boundary for the
32-class problem. The damping range therefore remains censored and should be
extended downward only inside a validation-safe search.

### Bipolar R&F refinement and fusion

| Parameter | Values/bounds |
|---|---|
| Damping alpha | 0.0025–0.4, eight log-like levels |
| Threshold | 0.0005–0.2, nine levels |
| Operating RMS | 0.05–8, eleven levels |
| Bipolar candidates | 792 per subject/class cell |
| Fusion candidates | fixed baselines plus 1,024 seeded Dirichlet weights |
| Channels | O1, Oz, O2, O1-Oz, O2-Oz |

The partial fused run selects alpha=0.0025 at the median, again the lower
boundary. This is both a scientific clue and an overfitting warning.

### Delta

| Parameter | Values/bounds |
|---|---|
| Band | target `f ± 1 Hz`; 1 Hz target uses 0.1–2 Hz |
| Filter | order 5, zero phase in the current offline exploration |
| Threshold scale | 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3 |
| Threshold reference | subject/class median RMS of filtered first differences |
| UP/DN asymmetry | 0.5, 0.75, 1, 1.25, 1.5, 2 |
| Features | target × UP/DN; EEG channels currently summed after encoding |

### LIF

| Parameter | Values/bounds |
|---|---|
| Band/filter | same target bank as delta |
| Threshold scale | 0.1–3 on subject/class median filtered RMS |
| Tau | 2, 5, 10, 20, 40, 75, 100 ms |
| Input gain | 1.0 fixed; threshold and gain are otherwise partly degenerate |
| Reset | zero after upward threshold crossing |
| Features | one count per target; EEG channels currently summed after encoding |

### General linear fusion

| Parameter | Initial values |
|---|---|
| Classifier | standardized multiclass ridge least squares |
| L2 grid | 0.001, 0.01, 0.1, 1, 10, 100 |
| Inner split | leave one trial/block group out |
| Candidate blocks | encoder × channel/reference × stream × timing feature |

## Experiment ledger and recorded runtime

Times below are sums of checkpoint-recorded cell runtimes. They are not clean
wall-clock benchmarks because jobs ran concurrently and compilation/cache state
varied.

| Experiment | Completion represented | Recorded runtime |
|---|---:|---:|
| 30-subject early apparent R&F | 150 cells | 0.41 h; median 3.54 s/cell |
| Deep R&F gain/harmonic/spread | 150 cells | 2.50 h; median 25.7 s; max 216.8 s |
| Decision endpoints | 268 endpoint cells currently | 6.05 h; median 23.5 s; max 34.97 min |
| Bipolar fixed-parameter comparison | 150 cells | runtime not stored |
| Five-channel fused R&F | 64 recorded cells at audit | 5.02 h; median 99.2 s; max 37.8 min |
| Corrected individual delta/LIF | evolving | typically seconds/cell; high-class filtering dominates |

The complete machine-readable inventory is regenerated by
`scripts/audit_experiment_ledger.py` into
`outputs/audits/project_experiment_ledger.json`.

## Hidden biases, starting assumptions, and blind spots

### Direct optimism

1. Most recent headline results are apparent accuracy: parameter selection,
   template calibration, feature weighting, and scoring use the same segments.
2. Searching hundreds of neuron settings and more than 1,000 fusion weights
   increases winner's-curse optimism even when each individual model is simple.
3. The current fused score weights are not nested and should not be interpreted
   as generalizable improvement.
4. Subject-specific tuning is intentional, but final claims still require
   held-out blocks within each subject.

### Temporal and filtering assumptions

1. The fixed 140 ms start offset may not be optimal for every subject or
   frequency and was not estimated inside validation folds.
2. Full-epoch zero-phase filtering uses samples after a one-second decision
   endpoint. It is an offline condition, not causal one-second evidence.
3. Prefix-only zero-phase filtering avoids later samples but still has endpoint
   edge behavior. A deployable analysis requires causal SOS filtering with
   state carried from pre-stimulus context.
4. A ±1 Hz filter is only weakly resolvable in a one-second window. Adjacent
   target banks overlap strongly in dense 32-class layouts.

### Frequency and class construction

1. Equal rounded targets from 8–39 Hz are a chosen subset, not the entire 1–60
   Hz task distribution.
2. Results confound number of classes with decreasing class spacing.
3. Harmonic collisions change with the selected subset; performance cannot be
   attributed only to class count.
4. The 1 Hz filter requires a special 0.1 Hz lower bound and much longer context
   than the current one-second decision window.

### Parameter identifiability

1. R&F input gain, operating RMS, and firing threshold are partially
   interchangeable.
2. LIF input gain and threshold are similarly degenerate.
3. Threshold does not move the linear R&F eigenfrequency, but reset, refractory
   behavior, finite duration, and threshold crossings move the firing-rate peak.
4. Boundary selections for alpha indicate either genuinely low damping or an
   objective exploiting dense, noisy, same-data templates.

### Spatial and data assumptions

1. O1/Oz/O2 ordering is assumed from dataset structure and should be asserted
   against metadata at load time.
2. Signal units appear to be microvolts but should be verified from acquisition
   documentation rather than only amplitude plausibility.
3. Current delta/LIF studies sum EEG channels after encoding. This can hide
   subject-specific lateralization and destructive cancellation in score space.
4. Bipolar subtraction removes common noise and common SSVEP signal; its value
   is subject and frequency dependent.

### Engineering and reporting

1. Earlier scripts hard-code grids and duplicate data-loading/adaptive-gain
   code. Reusable logic is now being moved into the package, but old scripts
   remain for reproducibility.
2. Recorded elapsed time excludes some preprocessing and old experiments did
   not store timing.
3. Concurrent long jobs make elapsed-time comparisons hardware-load dependent.
4. Multiple exploratory figures and analyses increase researcher degrees of
   freedom; the final protocol and primary metrics should be frozen before the
   confirmatory run.

## Feature-space visualization framework

Use embeddings as diagnostics, never accuracy evidence.

1. Start with standardized PCA: deterministic, global variance visible, and
   loadings traceable to encoder/channel/frequency features.
2. Add t-SNE for local neighborhood inspection. Repeat several seeds and
   perplexities; do not interpret cluster size, global distance, or empty space.
3. Add UMAP as a faster complementary nonlinear view, again with repeated seeds
   and neighborhood settings.
4. Color the same embedding separately by class, subject, trial block,
   correctness, amplitude, and selected parameter family. A class-separated
   plot that is actually subject-separated is a failure, not success.
5. Show train and held-out points distinctly. Fit scaling/PCA on training data;
   avoid using supervised labels to construct the display.
6. Pair every embedding with quantitative original-space diagnostics:
   nearest-neighbor accuracy, silhouette, trustworthiness, class-centroid
   distance, within-class covariance, and target/impostor margin.
7. Add linked views: selected point → EEG trace → filtered band → internal state
   → spikes → feature vector → linear contribution.

## Proposed optimization framework

### Objective hierarchy

1. Primary: nested held-out block balanced accuracy at a declared endpoint.
2. Secondary: information transfer rate including decision and gaze-shift time.
3. Robustness: fifth-percentile subject accuracy and parameter stability.
4. Efficiency: spikes/s, neuron count, runtime, and memory.
5. Calibration: target/impostor margin and expected calibration error.

### Search stages

1. **Sanity stage:** synthetic resonance, solver convergence, unit tests, and a
   small fixed subject set; no scientific selection.
2. **Multi-fidelity stage:** 2/4/8 classes, fewer inner folds, shorter endpoints,
   and pruning of weak configurations.
3. **Subject-stage optimization:** Bayesian/TPE search over non-degenerate
   parameterizations inside inner folds.
4. **Fusion stage:** freeze each encoder's inner-selected parameters, generate
   out-of-fold features, then train ridge linear fusion.
5. **Confirmatory stage:** locked search space and pipeline on untouched outer
   blocks; no new plot-driven parameter changes.

Every trial should store configuration hash, code version, subject, fold,
endpoint, seed, objective values, runtime, failure state, selected boundaries,
and feature provenance. A study database replaces scattered anonymous grids.

## Prioritized next big leaps

1. **Nested within-subject evaluation and out-of-fold feature generation.** This
   is the largest scientific improvement and must precede encoder fusion claims.
2. **Causal online pipeline.** Carry filter and neuron state from pre-stimulus
   context; compare causal and offline zero-phase performance at 250–1000 ms.
3. **General ridge fusion.** Combine frozen R&F/delta/LIF/channel/timing blocks
   using only out-of-fold training features; inspect standardized coefficients.
4. **Multi-objective Bayesian optimization.** Optimize accuracy, latency,
   neuron count, and robustness with pruning rather than Cartesian expansion.
5. **Factorial class design.** Separate class count, frequency range, minimum
   spacing, and harmonic collision instead of changing them together.
6. **Spatiotemporal channel model.** Retain O1/Oz/O2/bipolar blocks and learn
   regularized weights; add leave-one-channel-out ablations.
7. **Rate + TTFS + phase/event features.** Add them as named blocks and test
   incremental held-out value rather than assuming more features help.
8. **Embedding and error atlas.** PCA/t-SNE/UMAP linked to waveforms, states,
   spikes, margins, subjects, and parameter boundaries.
9. **Sensitivity and identifiability analysis.** Profile or Sobol-style effects
   for alpha, normalized threshold/gain ratio, tau, band width, and endpoint.
10. **Configuration-driven GUI.** The GUI edits validated YAML, estimates cost,
    launches/resumes studies, monitors folds, and renders the same audit plots;
    algorithms remain in the package rather than UI callbacks.
