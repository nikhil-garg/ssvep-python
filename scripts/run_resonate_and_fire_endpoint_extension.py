"""Independent 250/500/750 ms optimization after the 1 s deep search."""
from pathlib import Path
from itertools import product
import os,time
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.data.matlab import Matlab73Dataset

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;BASE=ROOT/'outputs/experiments/resonate_and_fire_deep_gain_search'
OUT=ROOT/'outputs/experiments/resonate_and_fire_decision_endpoints';CP=OUT/'checkpoints';CP.mkdir(parents=True,exist_ok=True)
SUBJECTS=np.arange(1,31);COUNTS=np.array((2,4,8,16,32));SETS=[np.rint(np.linspace(8,39,n)).astype(int) for n in COUNTS];ENDPOINTS=(250,500,750)
ALPHAS=np.array((.005,.01,.025,.05,.1,.2,.4));THRESHOLDS=np.array((.001,.002,.005,.01,.02,.05,.1,.2));OPERATING=np.array((.1,.25,.5,.75,1.,1.5,2.,3.,5.))
CANDIDATES=np.array(list(product(ALPHAS,THRESHOLDS,OPERATING)),float);HARMONICS=((1,),(1,2),(1,2,3));SPREADS=((0.,),(-.25,.25),(-.5,-.167,.167,.5))

def load_subject(sid,endpoint):
 with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source:chunk=source.read_channel_chunk(60,64)
 return chunk[1,:3,140:140+endpoint,:,:].transpose(2,3,0,1).astype(np.float32)
def evaluate(raw,freqs,endpoint,a,t,g,h=(1,),spread=(0.,),full=False):
 centered=raw-raw.mean(axis=-1,keepdims=True);rms=np.sqrt(np.mean(centered.astype(float)**2,axis=-1));gain=g/np.maximum(rms,1e-6);signals=(centered*gain[...,None]).reshape(-1,3,endpoint);labels=np.repeat(np.arange(len(freqs)),12)
 p=ResonateAndFireParameters(damping_alpha=float(a),threshold=float(t),input_gain=.8,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
 model=OscillatorBankClassifier(freqs,1000,p,harmonics=h,harmonic_weights=tuple(1/x for x in h),spread_hz=spread);model.channel_scale_=np.ones(3);model.fit_calibration(signals,labels,(endpoint,));scores=model.decision_scores(signals,(endpoint,))[0];pred=np.argmax(scores,1);acc=float(np.mean(pred==labels))
 return (acc,scores,pred,rms.reshape(-1,3),gain.reshape(-1,3)) if full else acc
def save(path,**data):
 tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,**data);os.replace(tmp,path)

for endpoint in ENDPOINTS:
 for sid in SUBJECTS:
  subject=load_subject(int(sid),endpoint)
  for ci,count in enumerate(COUNTS):
   path=CP/f'endpoint_{endpoint:04d}ms_subject_{sid:02d}_{count:02d}_classes.npz'
   if path.exists():continue
   freqs=SETS[ci];raw=subject[freqs-1];started=time.perf_counter();coarse=np.zeros(len(CANDIDATES),np.float32)
   for i,(a,t,g) in enumerate(CANDIDATES):coarse[i]=evaluate(raw,freqs,endpoint,a,t,g)
   top=np.argsort(coarse)[-12:][::-1];expanded=[];expanded_accuracy=[];best=None
   for base in top:
    a,t,g=CANDIDATES[base]
    for hi,h in enumerate(HARMONICS):
     for spi,spread in enumerate(SPREADS):
      acc=evaluate(raw,freqs,endpoint,a,t,g,h,spread);expanded.append((base,a,t,g,hi,spi));expanded_accuracy.append(acc)
      if best is None or acc>best[0]:best=(acc,base,a,t,g,hi,spi)
   acc,base,a,t,g,hi,spi=best;acc,scores,pred,rms,gains=evaluate(raw,freqs,endpoint,a,t,g,HARMONICS[hi],SPREADS[spi],True);labels=np.repeat(np.arange(count),12);target=scores[np.arange(len(labels)),labels];masked=scores.copy();masked[np.arange(len(labels)),labels]=-np.inf;impostor=masked.max(1)
   save(path,endpoint_ms=endpoint,subject_id=sid,class_count=count,frequencies_hz=freqs,candidates=CANDIDATES,coarse_accuracy=coarse,expanded_candidates=np.asarray(expanded),expanded_accuracy=np.asarray(expanded_accuracy),selected_alpha=a,selected_threshold=t,selected_operating_rms=g,selected_harmonics=np.asarray(HARMONICS[hi]),selected_spread_hz=np.asarray(SPREADS[spi]),accuracy=acc,prediction=pred,scores=scores,raw_rms_uV=rms,adaptive_gain_per_uV=gains,target_scores=target,best_impostor_scores=impostor,target_margin=target-impostor,elapsed_seconds=time.perf_counter()-started,evaluation_design='endpoint_specific_gain_and_same_data_optimization_no_future_samples_no_holdout')
   print(f'{endpoint}ms S{sid:02d} C{count:02d} {100*acc:.2f}% elapsed={time.perf_counter()-started:.1f}s',flush=True)
print('ENDPOINT EXTENSION COMPLETE',flush=True)
