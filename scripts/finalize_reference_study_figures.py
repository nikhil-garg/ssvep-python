import csv
import time
from pathlib import Path

import numpy as np

from ssvep_toolkit.visualization import load_reference_study_figure_data, render_reference_study_figure


ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "outputs/reference_study"
figure10 = base / "results/figure_10_fbcca.npz"
figure12 = base / "results/figure_12_fbtrca.npz"
while not (figure10.exists() and figure12.exists()):
    time.sleep(30)

data10 = load_reference_study_figure_data(figure10)
data12 = load_reference_study_figure_data(figure12)
render_reference_study_figure(10, data10, base / "figures/figure_10_fbcca.png")
render_reference_study_figure(12, data12, base / "figures/figure_12_fbtrca.png")

score_file = ROOT.parent / "Sub_score.csv"
with score_file.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.reader(handle))
header = rows[0]
scores = np.empty((30, 2, 60, 3))
for frequency, row in enumerate(rows[1:61]):
    for condition, token in enumerate(("Low_Depth", "High_Depth")):
        for subject in range(1, 31):
            for category, name in enumerate(("Comfort_level", "Flicker_perception", "Preference")):
                scores[subject - 1, condition, frequency, category] = float(row[header.index(f"Sub{subject}_{token}_{name}")])
subjective = scores.mean(axis=3)

bands = np.asarray([[2, 3, 4, 5], [12, 13, 14, 15], [40, 41, 42, 43]]) - 1
band_scores = np.stack([subjective[:, :, frequencies].mean(axis=2) for frequencies in bands], axis=2)
accuracy10 = data10["accuracy"]
composite11 = np.empty((2, 30, 2, 3, 4))
for duration_index, time_index in enumerate((9, 19, 29, 39)):
    performance = accuracy10[..., time_index]
    composite11[0, ..., duration_index] = 0.7 * performance + 0.3 * band_scores / 5
    composite11[1, ..., duration_index] = np.where(performance < 0.7, performance, band_scores / 5)
np.savez_compressed(base / "results/figure_11_composite.npz", composite_scores=composite11)
render_reference_study_figure(11, {"composite_scores": composite11}, base / "figures/figure_11_composite.png")

performance12 = data12["accuracy"]
composite13 = np.empty((2, 30, 2, 60))
composite13[0] = 0.7 * performance12 + 0.3 * subjective / 5
composite13[1] = np.where(performance12 < 0.7, performance12, subjective / 5)
np.savez_compressed(base / "results/figure_13_composite.npz", composite_scores=composite13)
render_reference_study_figure(13, {"composite_scores": composite13}, base / "figures/figure_13_composite.png")
