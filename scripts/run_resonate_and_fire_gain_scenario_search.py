"""Wide 1 s R&F search with raw-amplitude-derived per-segment/channel gain."""
from pathlib import Path
from itertools import product
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.data.matlab import Matlab73Dataset

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent
OUT=ROOT/'outputs/experiments/resonate_and_fire_gain_optimization';CP=OUT/'checkpoints';CP.mkdir(parents=True,exist_ok=True)
PROFILE=np.load(OUT/'raw_amplitude_and_resonance_profile.npz');SUBJECTS=PROFILE['selected_subject_ids'].astype(int)
BANDS=(np.array((8,9,10)),np.array((22,23,24)),np.array((37,38,39)));BAND_NAMES=('low_8-10hz','medium_22-24hz','high_37-39hz')
ALPHAS=np.array((.01,.025,.05,.1,.2));THRESHOLDS=np.array((.002,.005,.01,.02,.05,.1));OPERATING_RMS=np.array((.25,.5,.75,1.,1.5,2.,3.))
BASE=np.array(list(product(ALPHAS,THRESHOLDS,OPERATING_RMS)),float);HARMONICS=((1,),(1,2),(1,2,3))

def load_subject(sid):
 with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source:chunk=source.read_channel_chunk(60,64)
 return chunk[1,:3,140:1140,:,:].transpose(2,3,0,1).astype(np.float32)

def evaluate(raw,frequencies,alpha,threshold,operating_rms,harmonics):
 centered=raw-raw.mean(axis=-1,keepdims=True);amplitude=np.sqrt(np.mean(centered.astype(float)**2,axis=-1))
 gains=operating_rms/np.maximum(amplitude,1e-6);signals=(centered*gains[...,None]).reshape(-1,3,1000)
 labels=np.repeat(np.arange(len(frequencies)),raw.shape[1])
 p=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),input_gain=.8,integration_substeps=4,
  refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
 model=OscillatorBankClassifier(frequencies,1000,p,harmonics=harmonics,harmonic_weights=tuple(1/h for h in harmonics))
 model.channel_scale_=np.ones(3);model.fit_calibration(signals,labels,(1000,));scores=model.decision_scores(signals,(1000,))[0]
 prediction=np.argmax(scores,axis=-1);return np.mean(prediction==labels),scores,prediction,amplitude.reshape(-1,3),gains.reshape(-1,3)

for sid in SUBJECTS:
 subject=load_subject(int(sid))
 for band_name,frequencies in zip(BAND_NAMES,BANDS):
  path=CP/f'subject_{sid:02d}_{band_name}.npz'
  if path.exists():print(path.name,'resumed',flush=True);continue
  raw=subject[frequencies-1];base_accuracy=np.zeros(len(BASE));top_payload={}
  for i,(alpha,threshold,operating_rms) in enumerate(BASE):
   accuracy,*_=evaluate(raw,frequencies,alpha,threshold,operating_rms,(1,));base_accuracy[i]=accuracy
   if (i+1)%30==0:print(f'S{sid:02d} {band_name} fundamental {i+1}/{len(BASE)}',flush=True)
  top=np.argsort(base_accuracy)[-8:][::-1];expanded=[];expanded_accuracy=[];best=None
  for base_index in top:
   alpha,threshold,operating_rms=BASE[base_index]
   for hi,harmonics in enumerate(HARMONICS):
    accuracy,scores,prediction,amplitude,gains=evaluate(raw,frequencies,alpha,threshold,operating_rms,harmonics)
    expanded.append((base_index,alpha,threshold,operating_rms,hi));expanded_accuracy.append(accuracy)
    if best is None or accuracy>best[0]:best=(accuracy,scores,prediction,amplitude,gains,base_index,alpha,threshold,operating_rms,hi)
  accuracy,scores,prediction,amplitude,gains,base_index,alpha,threshold,operating_rms,hi=best
  labels=np.repeat(np.arange(3),12);target=scores[np.arange(len(labels)),labels];masked=scores.copy();masked[np.arange(len(labels)),labels]=-np.inf;impostor=masked.max(1)
  np.savez_compressed(path,subject_id=sid,band_name=band_name,frequencies_hz=frequencies,base_candidates=BASE,base_accuracy=base_accuracy,
   expanded_candidates=np.asarray(expanded),expanded_accuracy=np.asarray(expanded_accuracy),selected_base_index=base_index,selected_alpha=alpha,
   selected_threshold=threshold,selected_operating_rms=operating_rms,selected_harmonic_index=hi,selected_harmonics=np.asarray(HARMONICS[hi]),
   accuracy=accuracy,prediction=prediction,scores=scores,raw_rms_uV=amplitude,adaptive_gain_per_uV=gains,target_scores=target,
   best_impostor_scores=impostor,target_margin=target-impostor,evaluation_design='same_segment_gain_and_same_data_parameter_optimization_1s')
  print(f'S{sid:02d} {band_name} complete {100*accuracy:.2f}% alpha={alpha} th={threshold} G={operating_rms} H={HARMONICS[hi]}',flush=True)
print('complete',SUBJECTS.tolist())
