"""Render the six-panel evidence figure for nested multi-encoder checkpoints."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ssvep_toolkit.visualization import plot_nested_checkpoint_comparison

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "outputs/experiments/nested_multi_encoder/checkpoints"
OUTPUT = ROOT / "outputs/experiments/nested_multi_encoder/figures/01_outer_block_evidence.png"

paths = sorted(CHECKPOINTS.glob("subject_01_02_classes_*.npz"))
if len(paths) != 2:
    raise FileNotFoundError("expected completed offline and causal Subject 1 pilot checkpoints")
print(plot_nested_checkpoint_comparison(paths, OUTPUT))
