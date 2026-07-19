"""Twelve real-EEG R&F u/v/spike cases illustrating parameter tuning."""
from pathlib import Path
import re
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters,simulate_trace
from ssvep_toolkit.data.matlab import Matlab73Dataset
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;EXP=ROOT/'outputs/experiments/resonate_and_fire_deep_gain_search';FIG=EXP/'figures/tuning_cases';FIG.mkdir(parents=True,exist_ok=True)
cells=[]
for path in sorted((EXP/'checkpoints').glob('*.npz')):
 m=re.search(r'subject_(\d+)_(\d+)_classes',path.stem)
 with np.load(path) as d:
  cells.append(dict(path=path,sid=int(m.group(1)),count=int(m.group(2)),freqs=d['frequencies_hz'],pred=d['prediction'],margin=d['target_margin'],alpha=float(d['selected_alpha']),threshold=float(d['selected_threshold']),operating=float(d['selected_operating_rms']),harmonics=d['selected_harmonics'],spread=d['selected_spread_hz'],accuracy=float(d['accuracy'])))
criteria=(
 ('2-class strong correct',lambda c:c['count']==2,'correct_high'),('4-class strong correct',lambda c:c['count']==4,'correct_high'),('8-class strong correct',lambda c:c['count']==8,'correct_high'),('16-class strong correct',lambda c:c['count']==16,'correct_high'),
 ('32-class strong correct',lambda c:c['count']==32,'correct_high'),('High-overlap error',lambda c:c['count']>=16,'error_low'),('Lowest damping',lambda c:c['alpha']==.005,'correct_high'),('Higher damping',lambda c:c['alpha']>=.1,'correct_high'),
 ('High threshold',lambda c:c['threshold']==.2,'correct_high'),('Low threshold',lambda c:c['threshold']<=.002,'correct_high'),('High operating gain',lambda c:c['operating']==5.,'correct_high'),('Low operating gain',lambda c:c['operating']==.1,'error_low'))
chosen=[];used=set()
for label,accept,mode in criteria:
 candidates=[c for c in cells if accept(c)]
 candidates.sort(key=lambda c:(c['accuracy'],c['sid']),reverse=mode=='correct_high')
 pick=next((c for c in candidates if c['path'] not in used),candidates[0]);used.add(pick['path']);labels=np.repeat(np.arange(pick['count']),12);correct=pick['pred']==labels
 eligible=np.flatnonzero(correct if mode=='correct_high' else ~correct)
 if not len(eligible):eligible=np.arange(len(labels))
 trial=int(eligible[np.argmax(pick['margin'][eligible])] if mode=='correct_high' else eligible[np.argmin(pick['margin'][eligible])]);chosen.append((label,pick,trial))
cache={}
def raw_subject(sid):
 if sid not in cache:
  with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source:cache[sid]=source.read_channel_chunk(60,64)[1,1,140:1140,:,:]
 return cache[sid]
payload=[]
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-ticks')
for page in range(3):
 fig,axes=plt.subplots(4,4,figsize=(16,11),sharex='col');page_cases=chosen[page*4:(page+1)*4]
 for column,(label,c,trial) in enumerate(page_cases):
  class_index=trial//12;block=trial%12;frequency=float(c['freqs'][class_index]);raw=raw_subject(c['sid'])[:,int(frequency)-1,block].astype(float);centered=raw-raw.mean();raw_rms=np.sqrt(np.mean(centered**2));gain=c['operating']/max(raw_rms,1e-9);signal=centered*gain
  p=ResonateAndFireParameters(damping_alpha=c['alpha'],threshold=c['threshold'],input_gain=.8,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing');spikes,u,v=simulate_trace(signal,frequency,1000,p);t=np.arange(1000)/1000;prediction=int(c['pred'][trial]);correct=prediction==class_index
  axes[0,column].plot(t,signal,lw=.7);axes[0,column].set_title(f'{label}\nS{c["sid"]} · {c["count"]} classes · target {frequency:g} Hz · '+('correct' if correct else 'error'))
  axes[1,column].plot(t,u,lw=.75);axes[2,column].plot(t,v,lw=.75);axes[2,column].axhline(c['threshold'],color='r',ls='--',lw=1)
  axes[3,column].vlines(spikes/1000,0,1,lw=.8);axes[3,column].set(ylim=(-.05,1.05),yticks=(0,1),yticklabels=('quiet','spike'),xlabel='Time (s)')
  axes[0,column].text(.01,.97,f'α={c["alpha"]:g}, θ={c["threshold"]:g}, G={c["operating"]:g}, g={gain:.3f}/µV\nH={c["harmonics"].tolist()}, spread={np.round(c["spread"],3).tolist()}, spikes={len(spikes)}',transform=axes[0,column].transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
  payload.append((page,column,c['sid'],c['count'],frequency,block,c['alpha'],c['threshold'],c['operating'],gain,len(spikes),int(correct),float(c['margin'][trial])))
 for row,name in enumerate(('Gained Oz input','Internal state u','Internal state v','Output spikes')):axes[row,0].set_ylabel(name)
 fig.tight_layout();fig.savefig(FIG/f'{page+1:02d}_four_real_eeg_tuning_cases.png',dpi=180);plt.close(fig)
np.savez_compressed(EXP/'twelve_tuning_case_metadata.npz',cases=np.asarray(payload,float),columns=np.array(('page','column','subject_id','class_count','target_frequency_hz','block_zero_based','alpha','threshold','operating_rms','adaptive_gain_per_uV','spike_count','correct','target_margin')))
print('created',len(payload),'cases in',FIG)
