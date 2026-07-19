"""Multi-hour all-subject R&F search at the 1-second endpoint.

The run is deliberately resumable. Every subject/class-count cell is written
atomically only after completion. Parameters and accuracy use the same data,
so all reported values are apparent/training accuracy.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import os,time
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.data.matlab import Matlab73Dataset

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent
OUT=ROOT/'outputs/experiments/resonate_and_fire_deep_gain_search';CP=OUT/'checkpoints';CP.mkdir(parents=True,exist_ok=True)
ALL_COUNTS=np.array((2,4,8,16,32));CLASS_SETS=[np.rint(np.linspace(8,39,n)).astype(int) for n in ALL_COUNTS]
SUBJECTS=np.asarray([int(x) for x in os.environ.get('RF_SUBJECTS',','.join(map(str,range(1,31)))).split(',')]);COUNTS=np.asarray([int(x) for x in os.environ.get('RF_CLASS_COUNTS','2,4,8,16,32').split(',')])
ALPHAS=np.array((.005,.01,.025,.05,.1,.2,.4));THRESHOLDS=np.array((.001,.002,.005,.01,.02,.05,.1,.2));OPERATING=np.array((.1,.25,.5,.75,1.,1.5,2.,3.,5.))
COARSE=np.array(list(product(ALPHAS,THRESHOLDS,OPERATING)),float);HARMONICS=((1,),(1,2),(1,2,3));SPREADS=((0.,),(-.25,.25),(-.5,-.167,.167,.5))

def load_subject(sid):
 with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source:chunk=source.read_channel_chunk(60,64)
 return chunk[1,:3,140:1140,:,:].transpose(2,3,0,1).astype(np.float32)

def adaptive_signals(raw,operating):
 centered=raw-raw.mean(axis=-1,keepdims=True);rms=np.sqrt(np.mean(centered.astype(float)**2,axis=-1));gain=operating/np.maximum(rms,1e-6)
 return (centered*gain[...,None]).reshape(-1,3,1000),rms.reshape(-1,3),gain.reshape(-1,3)

def evaluate(raw,freqs,alpha,threshold,operating,harmonics=(1,),spread=(0.,),return_scores=False):
 signals,rms,gains=adaptive_signals(raw,operating);labels=np.repeat(np.arange(len(freqs)),raw.shape[1])
 p=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),input_gain=.8,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
 model=OscillatorBankClassifier(freqs,1000,p,harmonics=harmonics,harmonic_weights=tuple(1/h for h in harmonics),spread_hz=spread);model.channel_scale_=np.ones(3);model.fit_calibration(signals,labels,(1000,));scores=model.decision_scores(signals,(1000,))[0];pred=np.argmax(scores,axis=1);acc=float(np.mean(pred==labels))
 return (acc,scores,pred,rms,gains) if return_scores else acc

def atomic_save(path,**payload):
 temporary=path.with_suffix('.partial.npz');np.savez_compressed(temporary,**payload);os.replace(temporary,path)

def run_cell(sid,count,subject):
 path=CP/f'subject_{sid:02d}_{count:02d}_classes.npz'
 if path.exists():print(f'S{sid:02d} C{count:02d} resumed',flush=True);return
 freqs=CLASS_SETS[np.flatnonzero(ALL_COUNTS==count)[0]];raw=subject[freqs-1];started=time.perf_counter();coarse_accuracy=np.zeros(len(COARSE),np.float32)
 for i,(a,t,g) in enumerate(COARSE):
  coarse_accuracy[i]=evaluate(raw,freqs,a,t,g)
  if (i+1)%72==0:print(f'S{sid:02d} C{count:02d} coarse {i+1}/{len(COARSE)}',flush=True)
 # Refine around the 12 best distinct coarse operating points, then test
 # harmonic and resonance-spread banks without an exponential full grid.
 top=np.argsort(coarse_accuracy)[-12:][::-1];expanded=[];expanded_acc=[];best=None
 for base in top:
  a,t,g=COARSE[base]
  for hi,h in enumerate(HARMONICS):
   for spi,spread in enumerate(SPREADS):
    acc=evaluate(raw,freqs,a,t,g,h,spread);expanded.append((base,a,t,g,hi,spi));expanded_acc.append(acc)
    if best is None or acc>best[0]:best=(acc,base,a,t,g,hi,spi)
 acc,base,a,t,g,hi,spi=best;scores_acc,scores,pred,rms,gains=evaluate(raw,freqs,a,t,g,HARMONICS[hi],SPREADS[spi],True)
 labels=np.repeat(np.arange(count),12);target=scores[np.arange(len(labels)),labels];masked=scores.copy();masked[np.arange(len(labels)),labels]=-np.inf;impostor=masked.max(1)
 atomic_save(path,subject_id=sid,class_count=count,frequencies_hz=freqs,coarse_candidates=COARSE,coarse_accuracy=coarse_accuracy,
  expanded_candidates=np.asarray(expanded),expanded_accuracy=np.asarray(expanded_acc),selected_base_index=base,selected_alpha=a,selected_threshold=t,
  selected_operating_rms=g,selected_harmonic_index=hi,selected_harmonics=np.asarray(HARMONICS[hi]),selected_spread_index=spi,selected_spread_hz=np.asarray(SPREADS[spi]),
  accuracy=scores_acc,prediction=pred,scores=scores,raw_rms_uV=rms,adaptive_gain_per_uV=gains,target_scores=target,best_impostor_scores=impostor,target_margin=target-impostor,
  elapsed_seconds=time.perf_counter()-started,evaluation_design='same_subject_same_segments_parameter_optimization_and_accuracy_1s_no_holdout')
 print(f'S{sid:02d} C{count:02d} complete {100*scores_acc:.2f}% a={a} t={t} G={g} H={HARMONICS[hi]} spread={SPREADS[spi]} elapsed={time.perf_counter()-started:.1f}s',flush=True)

for sid in SUBJECTS:
 print(f'Loading subject {sid}/30',flush=True);subject=load_subject(int(sid))
 for count in COUNTS:run_cell(int(sid),int(count),subject)
print('DEEP SEARCH COMPLETE',flush=True)
