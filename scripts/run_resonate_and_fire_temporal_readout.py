"""Compare rate+TTFS+phase templates with count-only R&F on held-out S5."""
from pathlib import Path
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import TemporalOscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.visualization.resonate_and_fire_multisubject import render_temporal_comparison
ROOT=Path(__file__).resolve().parents[1];base=ROOT/"outputs/experiments/resonate_and_fire_five_subject_pilot";data=np.load(base/"cache/subjects_01-05_8-39hz_o1_oz_o2_1000hz.npz")["data"]
with np.load(base/"subjects1-3_fit_subject4_tune_subject5_test.npz") as prior:
 counts=prior['class_counts'];durations=prior['durations_seconds'];selected=prior['selected_parameters'];count_accuracy=prior['accuracy'];fft_accuracy=prior['fft_accuracy']
sets=[tuple(np.rint(np.linspace(8,39,int(n))).astype(int)) for n in counts];stops=np.rint(1000*durations).astype(int);accuracy=np.zeros_like(count_accuracy)
def flat(values,subjects):
 x=values[subjects].reshape(-1,3,5000);y=np.tile(np.repeat(np.arange(values.shape[1]),12),len(subjects));return x,y
for ci,freqs in enumerate(sets):
 values=data[:,np.asarray(freqs)-8];train,ytrain=flat(values,np.arange(4));test,ytest=flat(values,np.array((4,)));alpha,threshold,width,voters=selected[ci];p=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),integration_substeps=4,refractory_cycles=.5,solver='exact');offsets=np.array([0.]) if voters==1 else np.linspace(-width,width,int(voters));decisions=[]
 for offset in offsets:
  model=TemporalOscillatorBankClassifier(freqs,1000,p,spread_hz=(float(offset),),harmonics=(1,2,3)).fit_scaler(train);model.fit_calibration(train,ytrain,stops);decisions.append(model.decision_scores(test,stops))
 decisions=np.asarray(decisions);individual=np.argmax(decisions,axis=-1);total=decisions.sum(0);final=np.empty(individual.shape[1:],int)
 for di in range(len(durations)):
  for trial in range(len(ytest)):
   ballot=np.bincount(individual[:,di,trial],minlength=len(freqs));tie=np.flatnonzero(ballot==ballot.max());final[di,trial]=tie[np.argmax(total[di,trial,tie])]
 accuracy[ci]=np.mean(final==ytest[None,:],axis=1);print(len(freqs),'classes temporal',np.round(100*accuracy[ci],2).tolist())
result=base/"subject5_temporal_rate_ttfs_phase.npz";np.savez_compressed(result,class_counts=counts,durations_seconds=durations,temporal_accuracy=accuracy,count_accuracy=count_accuracy,fft_accuracy=fft_accuracy,selected_parameters=selected,features=np.array(('rate_spikes_per_s','ttfs_fraction_of_available_window','mean_cos_resonance_phase','mean_sin_resonance_phase')),evaluation_design='S1-S4_templates_S5_test_no_temporal_parameter_retuning')
render_temporal_comparison(result,base/"figures/temporal_readout");print(result)
