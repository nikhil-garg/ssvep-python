from pathlib import Path

import numpy as np
import pytest

from ssvep_toolkit.visualization.reference_study_figures import render_reference_study_figure


@pytest.mark.parametrize("number", range(4, 14))
def test_every_paper_figure_renders(number: int, tmp_path: Path) -> None:
    x = np.linspace(1, 60, 60)
    contracts = {
        4: {"signal": np.sin(np.linspace(0, 20, 1000)), "sampling_rate_hz": np.array(250)},
        5: {"frequencies_hz": x, "amplitude": np.abs(np.sin(x / 5))},
        6: {"frequencies_hz": x, "amplitude": np.abs(np.cos(x / 5))},
        7: {"matrix": np.arange(120).reshape(2, 60)},
        8: {"frequencies_hz": x, "amplitude": np.abs(np.sin(x / 5)), "snr": np.cos(x / 7)},
        9: {"frequencies_hz": x, "scores": np.tile(np.sin(x / 10), (5, 1))},
        10: {"x": x, "accuracy": np.linspace(0.25, 1, 60), "itr": np.linspace(0, 50, 60)},
        11: {"scores": np.arange(10), "accuracy": np.linspace(0.2, 0.9, 10)},
        12: {"frequencies_hz": x, "accuracy": np.linspace(0.25, 1, 60), "itr": np.linspace(0, 50, 60)},
        13: {"scores": np.arange(10), "accuracy": np.linspace(0.2, 0.9, 10)},
    }
    output = tmp_path / f"figure{number}.png"
    figure = render_reference_study_figure(number, contracts[number], output)
    assert output.stat().st_size > 0
    figure.clear()
