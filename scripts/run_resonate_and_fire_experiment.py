from pathlib import Path

import numpy as np

from ssvep_toolkit.evaluation import load_raw_resonate_and_fire_data, run_grouped_resonate_and_fire_experiment
from ssvep_toolkit.visualization import render_resonate_and_fire_suite


ROOT = Path(__file__).resolve().parents[1]
frequencies = tuple(range(1, 61))
experiment = ROOT / "outputs/experiments/resonate_and_fire_all_1-60hz"
cache = experiment / "cache/raw_high_depth_o1_oz_o2_1000hz_1-60hz.npz"
cache.parent.mkdir(parents=True, exist_ok=True)
raw_files = sorted(ROOT.parent.glob("data_s*_64.mat"), key=lambda p: int(p.stem.split("s")[1].split("_")[0]))
if cache.exists():
    with np.load(cache) as source:
        data = source["data"]
        sampling_rate = float(source["sampling_rate_hz"])
else:
    data, sampling_rate = load_raw_resonate_and_fire_data(raw_files, frequencies, condition=2)
    np.savez_compressed(cache, data=data.astype(np.float32), sampling_rate_hz=sampling_rate)

result = run_grouped_resonate_and_fire_experiment(
    data,
    sampling_rate,
    frequencies,
    experiment / "results/grouped_5fold_nested_parameters.npz",
    spread_hz=(-0.5, 0.0, 0.5), harmonics=(1, 2, 3), integration_substeps=4,
)
render_resonate_and_fire_suite(result, experiment / "figures", raw_data=data)
print(result)
