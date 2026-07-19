"""Five-subject R&F pilot: S1-3 fit, S4 tune, S5 held-out test."""
from pathlib import Path
from itertools import product
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.evaluation.resonate_and_fire_experiment import load_raw_resonate_and_fire_data,_fft_baseline
from ssvep_toolkit.visualization.resonate_and_fire_multisubject import render_multisubject_suite

ROOT=Path(__file__).resolve().parents[1];outdir=ROOT/"outputs/experiments/resonate_and_fire_five_subject_pilot";outdir.mkdir(parents=True,exist_ok=True);cache=outdir/"cache/subjects_01-05_8-39hz_o1_oz_o2_1000hz.npz";cache.parent.mkdir(parents=True,exist_ok=True)
pool=np.arange(8,40)
if cache.exists(): data=np.load(cache)["data"]
else:
 files=[ROOT.parent/f"data_s{i}_64.mat" for i in range(1,6)];data,_=load_raw_resonate_and_fire_data(files,pool,condition=2);np.savez_compressed(cache,data=np.asarray(data,np.float32),frequencies_hz=pool,sampling_rate_hz=1000.)
counts=np.array((2,4,8,16,32));class_sets=[tuple(np.rint(np.linspace(8,39,n)).astype(int)) for n in counts];durations=np.arange(.5,5.01,.5);stops=np.rint(1000*durations).astype(int)
# Compact grid chosen before seeing S4/S5: only the four focused parameters.
candidates=np.array(list(product((.05,.1),(.01,.02,.05),(.5,1.),(1,2,4))),float);validation=np.zeros((len(counts),len(candidates)));selected=np.zeros((len(counts),4));accuracy=np.zeros((len(counts),len(durations)));fft_accuracy=np.zeros_like(accuracy);subject5_predictions=[]

def flat(values,subjects):
 x=values[subjects].reshape(-1,3,5000);y=np.tile(np.repeat(np.arange(values.shape[1]),12),len(subjects));return x,y
def voted(freqs,p,width,voter_count,train,ytrain,test,calibration_stops):
 offsets=np.array([0.]) if voter_count==1 else np.linspace(-width,width,int(voter_count));decisions=[]
 for offset in offsets:
  m=OscillatorBankClassifier(freqs,1000,p,spread_hz=(float(offset),),harmonics=(1,2,3),harmonic_weights=(1,.5,1/3)).fit_scaler(train);m.fit_calibration(train,ytrain,calibration_stops);decisions.append(m.decision_scores(test,calibration_stops))
 decisions=np.asarray(decisions);individual=np.argmax(decisions,axis=-1);total=decisions.sum(0);final=np.empty(individual.shape[1:],int)
 for di in range(final.shape[0]):
  for trial in range(final.shape[1]):
   ballot=np.bincount(individual[:,di,trial],minlength=len(freqs));tie=np.flatnonzero(ballot==ballot.max());final[di,trial]=tie[np.argmax(total[di,trial,tie])]
 return final

checkdir=outdir/"checkpoints";checkdir.mkdir(exist_ok=True)
for ci,freqs in enumerate(class_sets):
 checkpoint=checkdir/f"{len(freqs)}classes.npz"
 if checkpoint.exists():
  with np.load(checkpoint) as s:validation[ci]=s['validation'];selected[ci]=s['selected'];accuracy[ci]=s['accuracy'];fft_accuracy[ci]=s['fft_accuracy'];pred=s['predictions']
  subject5_predictions.append(pred);print(len(freqs),"classes resumed");continue
 indexes=np.asarray(freqs)-8;values=data[:,indexes];fit_x,fit_y=flat(values,np.array((0,1,2)));val_x,val_y=flat(values,np.array((3,)))
 for pi,(alpha,threshold,width,voter_count) in enumerate(candidates):
  p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5,solver="exact");validation[ci,pi]=np.mean(voted(freqs,p,width,int(voter_count),fit_x,fit_y,val_x,(1000,))[0]==val_y)
 best=int(np.argmax(validation[ci]));alpha,threshold,width,voter_count=candidates[best];selected[ci]=candidates[best];train_x,train_y=flat(values,np.array((0,1,2,3)));test_x,test_y=flat(values,np.array((4,)));p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5,solver="exact")
 pred=voted(freqs,p,width,int(voter_count),train_x,train_y,test_x,stops);accuracy[ci]=np.mean(pred==test_y[None,:],axis=1);fft=_fft_baseline(values[4:5],freqs,1000,durations)[:,0].reshape(len(durations),-1);fft_accuracy[ci]=np.mean(fft==test_y[None,:],axis=1);subject5_predictions.append(pred)
 np.savez_compressed(checkpoint,validation=validation[ci],selected=selected[ci],accuracy=accuracy[ci],fft_accuracy=fft_accuracy[ci],predictions=pred);print(len(freqs),"classes",selected[ci].tolist(),np.round(100*accuracy[ci],2).tolist())
result=outdir/"subjects1-3_fit_subject4_tune_subject5_test.npz";np.savez_compressed(result,class_counts=counts,durations_seconds=durations,accuracy=accuracy,fft_accuracy=fft_accuracy,candidates=candidates,validation_scores=validation,selected_parameters=selected,evaluation_design="S1-S3_fit_S4_parameter_selection_S1-S4_refit_S5_test",subject_ids=np.arange(1,6))
render_multisubject_suite(result,outdir/"figures");print(result)
