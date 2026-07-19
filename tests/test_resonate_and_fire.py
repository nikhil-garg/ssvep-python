import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import (
    OscillatorBankClassifier,
    ResonateAndFireParameters,
    simulate_bank,
    simulate_bank_event_features,
    simulate_trace,
)


def test_oscillator_bank_prefers_matching_sine() -> None:
    fs = 1000
    time = np.arange(5000) / fs
    signals = np.stack((np.sin(2*np.pi*16*time), np.sin(2*np.pi*20*time)))[:, None, :]
    parameters = ResonateAndFireParameters(damping_alpha=.3, threshold=.05, transient_seconds=.1)
    scores = simulate_bank(signals, (16, 20), fs, parameters, (5000,))[0]
    assert np.argmax(scores[0]) == 0
    assert np.argmax(scores[1]) == 1


def test_classifier_scaling_uses_training_data() -> None:
    rng = np.random.default_rng(4)
    training = rng.normal(size=(10, 3, 100)) * np.array([1, 2, 4])[None, :, None]
    model = OscillatorBankClassifier((16, 20), 1000, ResonateAndFireParameters()).fit_scaler(training)
    transformed = model.transform(training)
    assert np.allclose(np.std(transformed, axis=(0, 2), ddof=1), 1, atol=.02)


def test_duration_snapshots_are_monotonic() -> None:
    fs = 1000
    signal = np.sin(2*np.pi*16*np.arange(2000)/fs)[None, None, :]
    scores = simulate_bank(signal, (16,), fs, ResonateAndFireParameters(threshold=.01), (500,1000,2000))
    assert np.all(np.diff(scores[:, 0, 0]) >= 0)


def test_normalized_dynamics_match_dimensional_equations() -> None:
    fs = 1000
    signal = np.sin(2*np.pi*14*np.arange(1000)/fs)[None, None, :]
    common = dict(damping_alpha=.3, threshold=.02, integration_substeps=4)
    normalized = simulate_bank(signal, (14,), fs, ResonateAndFireParameters(**common), (1000,))
    dimensional = simulate_bank(signal, (14,), fs, ResonateAndFireParameters(**common, normalized_dynamics=False), (1000,))
    assert np.array_equal(normalized, dimensional)


def test_spread_and_harmonic_bank_maps_neurons_back_to_classes() -> None:
    fs = 1000
    time = np.arange(1500)/fs
    signals = np.sin(2*np.pi*14*time)[None, None, :]
    model = OscillatorBankClassifier((14, 20), fs, ResonateAndFireParameters(threshold=.02),
                                     spread_hz=(-.5, 0, .5), harmonics=(1, 2)).fit_scaler(signals)
    assert model.neuron_frequencies_hz[:6] == (13.5, 14.0, 14.5, 27.5, 28.0, 28.5)
    assert model.scores(signals, (1500,)).shape == (1, 1, 2)


def test_response_calibration_removes_class_rate_bias() -> None:
    fs=1000; time=np.arange(1000)/fs; frequencies=(8, 14, 20)
    signals=np.stack([np.sin(2*np.pi*f*time) for f in frequencies for _ in range(2)])[:,None,:]
    labels=np.repeat(np.arange(3),2)
    model=OscillatorBankClassifier(frequencies,fs,ResonateAndFireParameters(threshold=.02),spread_hz=(-.5,0,.5)).fit_scaler(signals)
    model.fit_calibration(signals,labels,1000)
    assert np.mean(model.predict(signals,(1000,))[0]==labels) >= 2/3


def test_event_features_have_physical_ranges() -> None:
    fs=1000;time=np.arange(1000)/fs;signal=np.sin(2*np.pi*16*time)[None,None,:]
    values=simulate_bank_event_features(signal,(16,),fs,ResonateAndFireParameters(threshold=.02),(1000,))
    assert values.shape==(1,1,1,4)
    assert values[0,0,0,0]>=0
    assert 0<=values[0,0,0,1]<=1
    assert np.hypot(values[0,0,0,2],values[0,0,0,3])<=1.0001


def test_event_rate_matches_exact_upward_crossing_spike_counter() -> None:
    fs=1000;time=np.arange(1000)/fs;signal=np.sin(2*np.pi*16*time)[None,None,:]
    parameters=ResonateAndFireParameters(
        damping_alpha=.025,threshold=.01,input_gain=.05,solver="exact",
        reset_mode="zero",spike_detection="upward_crossing",
    )
    counts=simulate_bank(signal,(16,),fs,parameters,(1000,))[0,0,0]
    events=simulate_bank_event_features(signal,(16,),fs,parameters,(1000,))[0,0,0]
    available_seconds=1-parameters.transient_seconds
    assert np.isclose(events[0]*available_seconds,counts)


def test_uncompensated_normalized_input_preserves_frequency_dependent_spike_count() -> None:
    counts = []
    for frequency in (8, 16, 32):
        signal = .75 * np.sin(2*np.pi*frequency*np.arange(1000)/1000)
        parameters = ResonateAndFireParameters(
            damping_alpha=.025, threshold=.01, input_gain=.05,
            normalize_input_by_resonance=False, integration_substeps=4,
            refractory_cycles=.5, solver="exact", reset_mode="zero",
            spike_detection="upward_crossing",
        )
        counts.append(len(simulate_trace(signal, frequency, 1000, parameters)[0]))
    assert counts[0] < counts[1] < counts[2]
