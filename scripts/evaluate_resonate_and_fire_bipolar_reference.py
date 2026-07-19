"""Compare O1/Oz/O2 with bipolar O1-Oz and O2-Oz R&F readouts."""
from pathlib import Path
import re
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.data.matlab import Matlab73Dataset
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;BASE=ROOT/'outputs/experiments/resonate_and_fire_deep_gain_search';OUT=ROOT/'outputs/experiments/resonate_and_fire_bipolar_reference';CP=OUT/'checkpoints';CP.mkdir(parents=True,exist_ok=True)
counts=(2,4,8,16,32);sets=[np.rint(np.linspace(8,39,n)).astype(int) for n in counts]
def adapt(raw,G):
 centered=raw-raw.mean(axis=-1,keepdims=True);rms=np.sqrt(np.mean(centered.astype(float)**2,axis=-1));gain=G/np.maximum(rms,1e-6);return (centered*gain[...,None]).reshape(-1,raw.shape[-2],1000),rms.reshape(-1,raw.shape[-2]),gain.reshape(-1,raw.shape[-2])
for sid in range(1,31):
 with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source:chunk=source.read_channel_chunk(60,64)
 original=chunk[1,:3,140:1140,:,:].transpose(2,3,0,1).astype(np.float32)
 bipolar=np.stack((original[:,:,0]-original[:,:,1],original[:,:,2]-original[:,:,1]),axis=2)
 for ci,count in enumerate(counts):
  path=CP/f'subject_{sid:02d}_{count:02d}_classes.npz'
  if path.exists():continue
  with np.load(BASE/'checkpoints'/f'subject_{sid:02d}_{count:02d}_classes.npz') as d:
   alpha=float(d['selected_alpha']);threshold=float(d['selected_threshold']);G=float(d['selected_operating_rms']);harmonics=tuple(map(int,d['selected_harmonics']));spread=tuple(map(float,d['selected_spread_hz']));baseline=float(d['accuracy'])
  freqs=sets[ci];raw=bipolar[freqs-1];signals,rms,gains=adapt(raw,G);labels=np.repeat(np.arange(count),12)
  p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,input_gain=.8,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
  model=OscillatorBankClassifier(freqs,1000,p,harmonics=harmonics,harmonic_weights=tuple(1/h for h in harmonics),spread_hz=spread);model.channel_scale_=np.ones(2)
  raw_counts=model.scores(signals,(1000,))[0];direct=np.argmax(raw_counts,axis=1);direct_accuracy=float(np.mean(direct==labels))
  model.fit_calibration(signals,labels,(1000,));scores=model.decision_scores(signals,(1000,))[0];prediction=np.argmax(scores,axis=1);template_accuracy=float(np.mean(prediction==labels))
  np.savez_compressed(path,subject_id=sid,class_count=count,frequencies_hz=freqs,baseline_three_channel_accuracy=baseline,bipolar_direct_spike_accuracy=direct_accuracy,bipolar_template_accuracy=template_accuracy,bipolar_raw_spike_scores=raw_counts,bipolar_template_scores=scores,bipolar_prediction=prediction,bipolar_direct_prediction=direct,raw_bipolar_rms_uV=rms,adaptive_gain_per_uV=gains,selected_alpha=alpha,selected_threshold=threshold,selected_operating_rms=G,selected_harmonics=np.asarray(harmonics),selected_spread_hz=np.asarray(spread),reference_definition=np.array(('O1-Oz','O2-Oz')),evaluation_design='reuse_three_channel_selected_parameters_same_data_bipolar_recalibration')
  print(f'S{sid:02d} C{count:02d} base={100*baseline:.1f} bipolar-template={100*template_accuracy:.1f} direct={100*direct_accuracy:.1f}',flush=True)
print('BIPOLAR EVALUATION COMPLETE')
