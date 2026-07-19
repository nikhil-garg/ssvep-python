"""Plot gain distributions, parameter landscapes, and R&F internal states."""
from pathlib import Path
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters,simulate_trace
from ssvep_toolkit.data.matlab import Matlab73Dataset

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;OUT=ROOT/'outputs/experiments/resonate_and_fire_gain_optimization';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
profile=np.load(OUT/'raw_amplitude_and_resonance_profile.npz');subjects=profile['selected_subject_ids'].astype(int)
bands=(np.array((8,9,10)),np.array((22,23,24)),np.array((37,38,39)));names=('low_8-10hz','medium_22-24hz','high_37-39hz');band_labels=('Low 8–10 Hz','Medium 22–24 Hz','High 37–39 Hz')

def cell(sid,name):return np.load(OUT/'checkpoints'/f'subject_{sid:02d}_{name}.npz')
def raw_segment(sid,freq,block):
 with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source:chunk=source.read_channel_chunk(60,64)
 return chunk[1,:3,140:1140,freq-1,block].astype(float)

accuracy=np.zeros((7,3));alpha=np.zeros_like(accuracy);threshold=np.zeros_like(accuracy);operating=np.zeros_like(accuracy);harmonic=np.zeros_like(accuracy,dtype=int)
all_raw=[];all_gain=[];all_channel=[];all_subject=[];all_band=[];all_margins=[]
for si,sid in enumerate(subjects):
 for bi,name in enumerate(names):
  with cell(sid,name) as d:
   accuracy[si,bi]=d['accuracy'];alpha[si,bi]=d['selected_alpha'];threshold[si,bi]=d['selected_threshold'];operating[si,bi]=d['selected_operating_rms'];harmonic[si,bi]=d['selected_harmonic_index'];all_margins.extend(d['target_margin'])
   raw=d['raw_rms_uV'];gain=d['adaptive_gain_per_uV'];all_raw.extend(raw.ravel());all_gain.extend(gain.ravel());all_channel.extend(np.tile(np.arange(3),len(raw)));all_subject.extend(np.repeat(sid,raw.size));all_band.extend(np.repeat(bi,raw.size))
all_raw=np.asarray(all_raw);all_gain=np.asarray(all_gain);all_channel=np.asarray(all_channel);all_subject=np.asarray(all_subject);all_band=np.asarray(all_band)
np.savez_compressed(OUT/'selected_scenario_summary.npz',subject_ids=subjects,band_labels=np.asarray(band_labels),accuracy=accuracy,selected_alpha=alpha,selected_threshold=threshold,selected_operating_rms=operating,selected_harmonic_index=harmonic,raw_rms_uV=all_raw,adaptive_gain_per_uV=all_gain,gain_channel_index=all_channel,gain_subject_id=all_subject,gain_band_index=all_band)

import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-ticks')
# 1 all-subject raw amplitude profile
fig,ax=plt.subplots(figsize=(10,5));x=np.arange(1,31);y=profile['subject_median_rms_uV'];ax.plot(x,y,'o-',color='.6',lw=1);div=set(profile['diverse_subject_ids']);sim=set(profile['similar_subject_ids'])
for sid in x:
 if sid in div:ax.scatter(sid,y[sid-1],s=90,marker='D',label='Amplitude-diverse' if sid==min(div) else None)
 if sid in sim:ax.scatter(sid,y[sid-1],s=75,marker='s',label='Amplitude-matched' if sid==min(sim) else None)
ax.set(xlabel='Subject',ylabel='Median raw 1 s RMS (µV)',xticks=np.arange(1,31,2));ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'01_subject_amplitude_profile.png',dpi=180);plt.close(fig)
# 2 adaptive gain distributions
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ch,label in enumerate(('O1','Oz','O2')):
 axes[ch].hist(all_gain[all_channel==ch],bins=35,alpha=.8);axes[ch].set_title(label);axes[ch].set_xlabel('Adaptive gain (per µV)');axes[ch].set_yscale('log')
axes[0].set_ylabel('Segment count');fig.tight_layout();fig.savefig(FIG/'02_gain_distribution_by_channel.png',dpi=180);plt.close(fig)
# 3 selected parameter distributions
fig,axes=plt.subplots(1,4,figsize=(13,4));items=((alpha,'Damping α'),(threshold,'Threshold'),(operating,'Target RMS'),(harmonic,'Harmonic index'))
for ax,(values,label) in zip(axes,items):
 ax.boxplot([values[:,i] for i in range(3)],tick_labels=('Low','Medium','High'));ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(FIG/'03_selected_parameter_distributions.png',dpi=180);plt.close(fig)
# 4 accuracy heatmap
fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(100*accuracy,aspect='auto',cmap='viridis',vmin=33.3,vmax=100)
for i in range(7):
 for j in range(3):ax.text(j,i,f'{100*accuracy[i,j]:.1f}',ha='center',va='center',color='white' if accuracy[i,j]<.7 else 'black')
ax.set(xticks=np.arange(3),xticklabels=('Low','Medium','High'),yticks=np.arange(7),yticklabels=[f'S{s}' for s in subjects],xlabel='Frequency band',ylabel='Subject');fig.colorbar(im,ax=ax,label='Apparent accuracy at 1 s (%)');fig.tight_layout();fig.savefig(FIG/'04_accuracy_subject_band.png',dpi=180);plt.close(fig)
# 5 gain versus raw amplitude
fig,ax=plt.subplots(figsize=(8,5));
for bi,label in enumerate(('Low','Medium','High')):ax.scatter(all_raw[all_band==bi],all_gain[all_band==bi],s=8,alpha=.25,label=label)
ax.set(xlabel='Raw segment/channel RMS (µV)',ylabel='Adaptive gain (per µV)',yscale='log');ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'05_gain_vs_raw_amplitude.png',dpi=180);plt.close(fig)
# Select weak/medium/strong target-SNR examples independently in each band.
selected_index={sid:i for i,sid in enumerate(profile['subject_ids'].astype(int))};scenario=[]
for bi,freqs in enumerate(bands):
 candidates=[]
 for sid in subjects:
  si=selected_index[sid]
  for freq in freqs:
   for block in range(12):candidates.append((np.median(profile['target_spectral_snr'][si,freq-8,block]),sid,freq,block))
 candidates=sorted(candidates);positions=(round(.1*(len(candidates)-1)),round(.5*(len(candidates)-1)),round(.9*(len(candidates)-1)))
 scenario.append([candidates[p] for p in positions])
# 6 internal y states
fig,axes=plt.subplots(3,3,figsize=(14,9),sharex=True);trace_payload=[]
for bi in range(3):
 for strength in range(3):
  snr,sid,freq,block=scenario[bi][strength];raw=raw_segment(sid,freq,block);centered=raw[1]-raw[1].mean()
  with cell(sid,names[bi]) as d:a=float(d['selected_alpha']);th=float(d['selected_threshold']);g0=float(d['selected_operating_rms'])/np.sqrt(np.mean(centered**2))
  p=ResonateAndFireParameters(damping_alpha=a,threshold=th,input_gain=.8,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
  spikes,xstate,ystate=simulate_trace(centered*g0,float(freq),1000,p);t=np.arange(1000)/1000;ax=axes[strength,bi];ax.plot(t,ystate,lw=.8);ax.axhline(th,color='r',ls='--',lw=1);ax.vlines(spikes/1000,th,th*1.12,color='k',lw=.6)
  ax.set_title(f'{band_labels[bi]} · S{sid} · {freq} Hz\nSNR {snr:.2f}, {len(spikes)} spikes');trace_payload.append((bi,strength,snr,sid,freq,block,g0,a,th,len(spikes)))
  if bi==0:ax.set_ylabel(('Weak','Medium','Strong')[strength]+'\nstate y')
  if strength==2:ax.set_xlabel('Time (s)')
fig.tight_layout();fig.savefig(FIG/'06_internal_states_weak_medium_strong.png',dpi=180);plt.close(fig)
# 7 spike selectivity for target and neighboring neurons
fig,axes=plt.subplots(3,3,figsize=(14,8),sharex=True)
for bi in range(3):
 for strength in range(3):
  snr,sid,freq,block=scenario[bi][strength];raw=raw_segment(sid,freq,block);centered=raw[1]-raw[1].mean()
  with cell(sid,names[bi]) as d:a=float(d['selected_alpha']);th=float(d['selected_threshold']);g0=float(d['selected_operating_rms'])/np.sqrt(np.mean(centered**2))
  p=ResonateAndFireParameters(damping_alpha=a,threshold=th,input_gain=.8,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
  ax=axes[strength,bi]
  for row,res in enumerate((freq-1,freq,freq+1)):
   spikes,_,_=simulate_trace(centered*g0,float(res),1000,p);ax.vlines(spikes/1000,row-.35,row+.35,lw=.7,label=f'{res} Hz' if strength==0 else None)
  ax.set(yticks=(0,1,2),yticklabels=(f'{freq-1}',f'{freq}',f'{freq+1}'),title=f'{band_labels[bi]} · S{sid} · target {freq} Hz')
  if bi==0:ax.set_ylabel(('Weak','Medium','Strong')[strength]+'\nresonator (Hz)')
  if strength==2:ax.set_xlabel('Spike time (s)')
fig.tight_layout();fig.savefig(FIG/'07_spike_rasters_weak_medium_strong.png',dpi=180);plt.close(fig)
# 8 alpha-threshold landscape for median-amplitude representative S7
fig,axes=plt.subplots(1,3,figsize=(12,4))
for bi,name in enumerate(names):
 with cell(7,name) as d:z=d['base_accuracy'].reshape(5,6,7).max(axis=2)
 im=axes[bi].imshow(100*z,origin='lower',aspect='auto',cmap='magma',extent=(0,6,0,5));axes[bi].set_title(band_labels[bi]);axes[bi].set_xticks(np.arange(6)+.5,('.002','.005','.01','.02','.05','.1'));axes[bi].set_yticks(np.arange(5)+.5,('.01','.025','.05','.1','.2'));axes[bi].set_xlabel('Threshold')
axes[0].set_ylabel('Damping α');fig.colorbar(im,ax=axes.tolist(),label='Best gain accuracy (%)',shrink=.8);fig.savefig(FIG/'08_alpha_threshold_landscapes_s07.png',dpi=180,bbox_inches='tight');plt.close(fig)
np.savez_compressed(OUT/'weak_medium_strong_examples.npz',examples=np.asarray(trace_payload,float),columns=np.array(('band_index','strength_index','spectral_snr','subject_id','frequency_hz','block_zero_based','adaptive_gain_per_uV','alpha','threshold','spike_count')))
print('accuracy mean by band',np.round(100*accuracy.mean(0),2));print('gain percentiles',np.round(np.percentile(all_gain,(1,5,25,50,75,95,99)),4));print('figures',FIG)
