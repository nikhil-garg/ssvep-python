"""Show that raw spike counts are no longer flattened by input-frequency compensation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters, simulate_trace
from ssvep_toolkit.visualization import plot_rf_input_compensation_ablation

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "outputs/examples/neuron_behavior"
paths = sorted(EXAMPLES.glob("*.npz"))
labels = []; compensated = []; raw_counts = []
for path in paths:
    metadata = __import__("json").loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path) as source:
        raw = source["raw_uV"].astype(float)
    centered = raw - raw.mean(); rms = np.sqrt(np.mean(centered**2)); adapted = centered * (.75 / max(rms, 1e-9))
    labels.append(f"{metadata['frequency_hz']} Hz · {metadata['electrode']}")
    for normalize, destination in ((True, compensated), (False, raw_counts)):
        parameters = ResonateAndFireParameters(
            damping_alpha=.025, threshold=.01, input_gain=.05,
            normalize_input_by_resonance=normalize, integration_substeps=4,
            refractory_cycles=.5, solver="exact", reset_mode="zero",
            spike_detection="upward_crossing",
        )
        destination.append(len(simulate_trace(adapted, metadata["frequency_hz"], 1000, parameters)[0]))
output = ROOT / "outputs/examples/neuron_behavior/02_rf_input_frequency_compensation_ablation.png"
print(plot_rf_input_compensation_ablation(labels, compensated, raw_counts, output))
np.savez_compressed(output.with_suffix(".npz"), labels=np.asarray(labels),
                    compensated_counts=compensated, uncompensated_raw_counts=raw_counts)
