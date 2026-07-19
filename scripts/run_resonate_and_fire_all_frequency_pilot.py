"""Fast real-data pilot over all 60 classes; blocks 1-8 train, 9-12 test.

This is an exploratory within-subject check, not the final subject-held-out result.
"""
from pathlib import Path

import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier, ResonateAndFireParameters
from ssvep_toolkit.evaluation.resonate_and_fire_experiment import _fft_baseline, load_raw_resonate_and_fire_data
from ssvep_toolkit.visualization import render_resonate_and_fire_suite


ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "outputs/experiments/resonate_and_fire_all_1-60hz/pilot_subject_01"
cache = output_dir / "cache/raw_o1_oz_o2_1000hz.npz"
frequencies = np.arange(1, 61)
durations = np.arange(.5, 5.01, .5)
stops = np.rint(durations*1000).astype(int)
cache.parent.mkdir(parents=True, exist_ok=True)
if cache.exists():
    with np.load(cache) as source: data=source["data"]
else:
    data,_=load_raw_resonate_and_fire_data([ROOT.parent/"data_s1_64.mat"],frequencies,condition=2)
    np.savez_compressed(cache,data=np.asarray(data,dtype=np.float32),sampling_rate_hz=1000.)

train=data[0,:,:8].reshape(-1,3,5000); train_labels=np.repeat(np.arange(60),8)
test=data[0,:,8:].reshape(-1,3,5000); test_labels=np.repeat(np.arange(60),4)
parameters=ResonateAndFireParameters(damping_alpha=.1,threshold=.01,integration_substeps=4,refractory_cycles=.5)
model=OscillatorBankClassifier(frequencies,1000,parameters,spread_hz=(-.5,0,.5),harmonics=(1,2,3)).fit_scaler(train)
model.fit_calibration(train,train_labels,stops)
decision=model.decision_scores(test,stops); predicted=np.argmax(decision,axis=-1)
predictions=predicted.reshape(len(durations),1,60,4)
truth=np.broadcast_to(np.arange(60)[None,None,:,None],predictions.shape)
subject_accuracy=np.mean(predictions==truth,axis=(2,3)).T
fft_predictions=_fft_baseline(data[:,:,:,],frequencies,1000,durations)[:,:, :,8:]
fft_subject_accuracy=np.mean(fft_predictions==truth,axis=(2,3)).T
result=output_dir/"results/within_subject_blocks_1-8_train_9-12_test.npz"; result.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(result,frequencies_hz=frequencies,durations_seconds=durations,predictions=predictions,
    spike_scores=decision.reshape(len(durations),1,60,4,60),subject_accuracy=subject_accuracy,
    accuracy=subject_accuracy.mean(0),fft_predictions=fft_predictions,fft_subject_accuracy=fft_subject_accuracy,
    selected_parameters=np.array([[.1,.01]]),parameter_scores=np.zeros((1,1,1)),damping_grid=np.array([.1]),
    threshold_grid=np.array([.01]),sampling_rate_hz=1000.,spread_hz=np.array([-.5,0,.5]),
    harmonics=np.array([1,2,3]),harmonic_weights=np.array([1,.5,1/3]),integration_substeps=4,refractory_cycles=.5,
    normalized_dynamics=True,evaluation_design="within_subject_blocks_1-8_train_9-12_test")
render_resonate_and_fire_suite(result,output_dir/"figures",raw_data=data)
print("R&F",np.round(subject_accuracy[0]*100,2).tolist())
print("FFT",np.round(fft_subject_accuracy[0]*100,2).tolist())
print(result)
