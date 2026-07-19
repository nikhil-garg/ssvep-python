"""Jointly tune alpha, threshold, bank width and voter count; analyze spikes."""
from pathlib import Path
from itertools import product
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters,simulate_bank,simulate_trace
from ssvep_toolkit.evaluation.resonate_and_fire_experiment import _fft_baseline
from ssvep_toolkit.visualization.resonate_and_fire_focused import render_focused_suite

ROOT=Path(__file__).resolve().parents[1]; raw=np.load(ROOT/"outputs/experiments/resonate_and_fire_all_1-60hz/pilot_subject_01/cache/raw_o1_oz_o2_1000hz.npz")["data"][0]
counts=np.array((2,4,8,16,32)); class_sets=[tuple(np.rint(np.linspace(8,39,n)).astype(int)) for n in counts]; durations=np.arange(.5,5.01,.5); stops=np.rint(1000*durations).astype(int)
alphas=np.array((.05,.1,.2)); thresholds=np.array((.005,.01,.02,.05)); widths=np.array((.1,.5,1.,2.)); voters=np.array((1,2,4,8)); candidates=np.array(list(product(alphas,thresholds,widths,voters)),float)
stage1=np.zeros((len(counts),len(candidates))); stage2=np.full_like(stage1,np.nan); selected=np.zeros((len(counts),4)); accuracy=np.zeros((len(counts),len(durations))); fft_accuracy=np.zeros_like(accuracy)
checkpoint_dir=ROOT/"outputs/experiments/resonate_and_fire_focused_tuning/checkpoints";checkpoint_dir.mkdir(parents=True,exist_ok=True)

def voted(frequencies,p,width,voter_count,train,train_y,test,calibration_stops):
 offsets=np.array([0.]) if voter_count==1 else np.linspace(-width,width,int(voter_count)); decisions=[]
 for offset in offsets:
  model=OscillatorBankClassifier(frequencies,1000,p,spread_hz=(float(offset),),harmonics=(1,2,3),harmonic_weights=(1,.5,1/3)).fit_scaler(train);model.fit_calibration(train,train_y,calibration_stops);decisions.append(model.decision_scores(test,calibration_stops))
 decisions=np.asarray(decisions); individual=np.argmax(decisions,axis=-1); total=decisions.sum(0); final=np.empty(individual.shape[1:],int)
 for di in range(final.shape[0]):
  for trial in range(final.shape[1]):
   ballot=np.bincount(individual[:,di,trial],minlength=len(frequencies));tie=np.flatnonzero(ballot==ballot.max());final[di,trial]=tie[np.argmax(total[di,trial,tie])]
 return final

for ci,frequencies in enumerate(class_sets):
 checkpoint=checkpoint_dir/f"subject01_{len(frequencies)}classes.npz"
 if checkpoint.exists():
  with np.load(checkpoint) as saved: stage1[ci]=saved['stage1'];stage2[ci]=saved['stage2'];selected[ci]=saved['selected'];accuracy[ci]=saved['accuracy'];fft_accuracy[ci]=saved['fft_accuracy']
  print(len(frequencies),"classes resumed",selected[ci].tolist());continue
 values=raw[np.asarray(frequencies)-1];n=len(frequencies)
 fit1=values[:,:4].reshape(-1,3,5000);y1=np.repeat(np.arange(n),4);val1=values[:,4:6].reshape(-1,3,5000);yv1=np.repeat(np.arange(n),2)
 for pi,(alpha,threshold,width,voter_count) in enumerate(candidates):
  p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5,solver="exact");stage1[ci,pi]=np.mean(voted(frequencies,p,width,int(voter_count),fit1,y1,val1,(1000,))[0]==yv1)
 top=np.argsort(stage1[ci])[-12:];fit2=values[:,:6].reshape(-1,3,5000);y2=np.repeat(np.arange(n),6);val2=values[:,6:8].reshape(-1,3,5000);yv2=np.repeat(np.arange(n),2)
 for pi in top:
  alpha,threshold,width,voter_count=candidates[pi];p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5,solver="exact");stage2[ci,pi]=np.mean(voted(frequencies,p,width,int(voter_count),fit2,y2,val2,(1000,))[0]==yv2)
 best=top[np.nanargmax(stage2[ci,top])];alpha,threshold,width,voter_count=candidates[best];selected[ci]=(alpha,threshold,width,voter_count)
 train=values[:,:8].reshape(-1,3,5000);yt=np.repeat(np.arange(n),8);test=values[:,8:].reshape(-1,3,5000);ytest=np.repeat(np.arange(n),4);p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5,solver="exact")
 pred=voted(frequencies,p,width,int(voter_count),train,yt,test,stops);accuracy[ci]=np.mean(pred==ytest[None,:],axis=1);fft=_fft_baseline(values[None],frequencies,1000,durations)[:,0,:,8:].reshape(len(durations),-1);fft_accuracy[ci]=np.mean(fft==ytest[None,:],axis=1)
 np.savez_compressed(checkpoint,stage1=stage1[ci],stage2=stage2[ci],selected=selected[ci],accuracy=accuracy[ci],fft_accuracy=fft_accuracy[ci])
 print(n,"classes selected",selected[ci].tolist(),"R&F",np.round(100*accuracy[ci],2).tolist())

# Spike timing analysis on a held-out 16 Hz Oz trial, standardized by training blocks only.
scale=np.std(raw[:,0:8],axis=(0,1,3),ddof=1);signal=(raw[15,8,1]-raw[15,8,1].mean())/scale[1];resonances=np.arange(12,20.001,.1);rates=np.zeros((len(alphas),len(thresholds),len(resonances)));timing=np.full((len(alphas),len(thresholds),5),np.nan)
for ai,alpha in enumerate(alphas):
 for ti,threshold in enumerate(thresholds):
  p=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),integration_substeps=4,refractory_cycles=.5,solver="exact");rates[ai,ti]=simulate_bank(signal[None,None,:],resonances,1000,p,(5000,))[0,0]/4.9
  spikes,_,_=simulate_trace(signal,16,1000,p);times=spikes/1000;times=times[times>=.1];isi=np.diff(times)
  if len(times):
   timing[ai,ti,0]=len(times)/4.9;timing[ai,ti,1]=times[0]-.1;timing[ai,ti,2]=np.median(isi) if len(isi) else np.nan;timing[ai,ti,3]=np.std(isi)/np.mean(isi) if len(isi) and np.mean(isi)>0 else np.nan;timing[ai,ti,4]=abs(np.mean(np.exp(1j*2*np.pi*16*times))) if len(times)>=5 else np.nan
raster=np.full((3,2000),np.nan)
p=ResonateAndFireParameters(damping_alpha=.1,threshold=.02,integration_substeps=4,refractory_cycles=.5,solver="exact")
for i,res in enumerate((15.,16.,17.)):
 sp,_,_=simulate_trace(signal,res,1000,p);raster[i,:min(len(sp),2000)]=sp[:2000]/1000

outdir=ROOT/"outputs/experiments/resonate_and_fire_focused_tuning/pilot_subject_01";outdir.mkdir(parents=True,exist_ok=True);result=outdir/"focused_joint_tuning_and_spikes.npz"
np.savez_compressed(result,class_counts=counts,durations_seconds=durations,accuracy=accuracy,fft_accuracy=fft_accuracy,candidates=candidates,stage1_scores=stage1,stage2_scores=stage2,selected_parameters=selected,alpha_grid=alphas,threshold_grid=thresholds,width_grid_hz=widths,voter_grid=voters,
 resonance_frequencies_hz=resonances,spike_rates_hz=rates,timing_metrics=timing,timing_metric_names=np.array(("rate_hz","first_spike_latency_s","median_isi_s","isi_cv","phase_vector_strength")),raster_resonances_hz=np.array((15.,16.,17.)),raster_times_s=raster,sampling_rate_hz=1000.,signal_unit="training_standard_deviations",threshold_unit="dimensionless_state")
render_focused_suite(result,outdir/"figures");print(result)
