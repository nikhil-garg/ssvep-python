# Resonate-and-fire paper figure plan

## Figure 1 - Method and evaluation design

- Raw O1/Oz/O2 signal in inferred microvolts.
- Endpoint-specific adaptive channel gain.
- Fundamental, harmonic, and spread oscillator bank.
- Internal state, spikes, rate, TTFS, and phase features.
- Subject-specific calibration and explicit same-data versus held-out branches.

## Figure 2 - Accuracy, latency, and practical BCI performance

- Subject distributions at 250, 500, 750, and 1000 ms.
- Accuracy-versus-window curves with paired subject trajectories.
- Chance-normalized accuracy and information transfer rate.
- Mark the best accuracy endpoint and best ITR endpoint separately.

## Figure 3 - Subject variability and frequency difficulty

- Subject-by-class-count accuracy heatmap.
- Frequency-resolved confusion and harmonic-collision matrices.
- Low-, middle-, and high-frequency summaries.
- Rank subjects by median performance while retaining subject identifiers.

## Figure 4 - Parameter identifiability and optimization evidence

- Median alpha-threshold landscapes, maximizing over gain.
- Gain-threshold ratio and boundary-hit diagnostics.
- Selected parameter atlas across subjects and class counts.
- Improvement from harmonics and resonance spread relative to the coarse bank.

## Figure 5 - Adaptive gain and raw-amplitude behavior

- Raw RMS distribution by subject and channel.
- Adaptive-gain distribution by channel and endpoint.
- Raw amplitude versus selected gain and accuracy.
- Show whether fitted gain tracks amplitude alone or subject performance.

## Figure 6 - Mechanistic resonance examples

- Weak, medium, and strong spectral-SNR examples at low, middle, and high frequencies.
- Raw input, internal state, threshold, reset events, and spike raster on aligned time axes.
- Neighboring and harmonic neuron responses on the same trial.
- Include successful and failed classifications rather than only favorable examples.

## Figure 7 - Decision separation and failure modes

- True-class versus best-impostor score distributions.
- Accuracy versus overlap coefficient.
- Target-margin distributions by subject and endpoint.
- Example trials illustrating harmonic collision, suppressed firing, and ties.

## Figure 8 - Robustness and comparison

- Fundamental-only versus harmonic and spread banks.
- R&F versus FFT, CCA, FBCCA, TRCA, and FBTRCA under identical windows.
- Apparent accuracy and held-out-block accuracy presented separately.
- Sensitivity to solver substeps and endpoint-specific gain estimation.
