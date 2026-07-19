import numpy as np

from ssvep_toolkit.algorithms import (
    DeltaEncoderParameters, LIFEncoderParameters, delta_state_trace, lif_state_trace,
)


def test_delta_state_trace_exposes_both_polarities():
    trace = delta_state_trace([0, 2, 0], DeltaEncoderParameters(1.0))
    assert trace["up_spikes"].tolist() == [False, True, False]
    assert trace["down_spikes"].tolist() == [False, False, True]


def test_lif_state_trace_exposes_membrane_and_crossings():
    trace = lif_state_trace(np.ones(100), 1000, LIFEncoderParameters(.1, .01, 10))
    assert trace["membrane"].shape == (100,)
    assert trace["spikes"].any()
