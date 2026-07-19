from __future__ import annotations
from pathlib import Path
from .plots import _finish,_pyplot

def render_controlled_suite(result_path: str|Path,output_dir: str|Path)->list[Path]:
 import numpy as np
 output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 with np.load(result_path) as s:d={k:s[k] for k in s.files}
 funcs=[("01_class_frequency_layout.png",_layout),("02_accuracy_vs_duration.png",_accuracy),("03_rf_vs_fft.png",_comparison),("04_five_second_scaling.png",_scaling),("05_span_voter_validation.png",_structure),("06_damping_threshold_validation.png",_parameters),("07_selected_parameters.png",_selected),("08_confusion_matrices_5s.png",_confusions),("09_eeg_units_audit.png",_units),("10_solver_convergence.png",_solver)]
 return [fn(d,output_dir/name) for name,fn in funcs]

def _layout(d,p):
 plt=_pyplot();fig,ax=plt.subplots(figsize=(10,4.5))
 for i,n in enumerate(d['class_counts']):
  f=d['class_frequencies_hz'][i];f=f[~__import__('numpy').isnan(f)];ax.scatter(f,[i]*len(f),s=20)
 ax.set(xlabel='Target frequency (Hz)',ylabel='Number of classes',yticks=range(len(d['class_counts'])),yticklabels=d['class_counts']);ax.grid(axis='x',alpha=.2);_finish(fig,p);return p

def _accuracy(d,p):
 plt=_pyplot();fig,ax=plt.subplots(figsize=(9,5))
 for i,n in enumerate(d['class_counts']):ax.plot(d['durations_seconds'],100*d['accuracy'][i],marker='o',label=f'{n} classes')
 ax.set(xlabel='Epoch duration (s)',ylabel='Held-out block accuracy (%)',ylim=(0,102));ax.grid(alpha=.2);ax.legend(frameon=False,ncols=2);_finish(fig,p);return p

def _comparison(d,p):
 plt=_pyplot();fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True);axes=axes.ravel()
 for i,n in enumerate(d['class_counts']):
  axes[i].plot(d['durations_seconds'],100*d['accuracy'][i],marker='o',label='R&F');axes[i].plot(d['durations_seconds'],100*d['fft_accuracy'][i],marker='s',label='FFT + harmonic');axes[i].axhline(100/n,color='.5',ls='--',lw=1);axes[i].set_title(f'{n} classes');axes[i].grid(alpha=.2)
 axes[5].axis('off');axes[0].legend(frameon=False);axes[0].set_ylabel('Accuracy (%)');axes[3].set_ylabel('Accuracy (%)');axes[3].set_xlabel('Epoch duration (s)');axes[4].set_xlabel('Epoch duration (s)');_finish(fig,p);return p

def _scaling(d,p):
 import numpy as np
 plt=_pyplot();fig,ax=plt.subplots(figsize=(8,4.8));n=d['class_counts'];rf=100*d['accuracy'][:,-1];fft=100*d['fft_accuracy'][:,-1];chance=100/n;x=np.arange(len(n));w=.36
 ax.bar(x-w/2,rf,w,label='R&F');ax.bar(x+w/2,fft,w,label='FFT + harmonic');ax.plot(x,chance,'k--o',label='Chance');ax.set(xlabel='Number of classes',ylabel='5 s accuracy (%)',xticks=x,xticklabels=n);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);_finish(fig,p);return p

def _structure(d,p):
 plt=_pyplot();fig,axes=plt.subplots(1,5,figsize=(16,3.6),sharex=True,sharey=True)
 for i,n in enumerate(d['class_counts']):
  im=axes[i].imshow(100*d['structure_validation_scores'][i],origin='lower',aspect='auto',vmin=0,vmax=100);axes[i].set(title=f'{n} classes',xlabel='Voters',xticks=range(len(d['voter_grid'])),xticklabels=d['voter_grid'],yticks=range(len(d['spread_half_width_grid_hz'])),yticklabels=d['spread_half_width_grid_hz'])
 axes[0].set_ylabel('Half-width (Hz)');fig.colorbar(im,ax=axes,label='Validation accuracy (%)',shrink=.75);_finish(fig,p);return p

def _parameters(d,p):
 plt=_pyplot();fig,axes=plt.subplots(1,5,figsize=(16,3.6),sharex=True,sharey=True)
 for i,n in enumerate(d['class_counts']):
  im=axes[i].imshow(100*d['parameter_validation_scores'][i],origin='lower',aspect='auto',vmin=0,vmax=100);axes[i].set(title=f'{n} classes',xlabel='Threshold',xticks=range(len(d['threshold_grid'])),xticklabels=d['threshold_grid'],yticks=range(len(d['damping_grid'])),yticklabels=d['damping_grid']);axes[i].tick_params(axis='x',rotation=45)
 axes[0].set_ylabel('Damping α');fig.colorbar(im,ax=axes,label='Validation accuracy (%)',shrink=.75);_finish(fig,p);return p

def _selected(d,p):
 plt=_pyplot();fig,axes=plt.subplots(2,2,figsize=(9,7));labels=('Voters','Half-width (Hz)','Damping α','Threshold')
 for j,ax in enumerate(axes.ravel()):ax.plot(d['class_counts'],d['selected_parameters'][:,j],marker='o');ax.set(xlabel='Classes',ylabel=labels[j],xscale='log',xticks=d['class_counts']);ax.get_xaxis().set_major_formatter('{x:g}');ax.grid(alpha=.2)
 _finish(fig,p);return p

def _confusions(d,p):
 import numpy as np
 plt=_pyplot();fig,axes=plt.subplots(1,5,figsize=(16,3.6))
 for i,n in enumerate(d['class_counts']):
  m=np.zeros((n,n),int);valid=d['truths'][i]>=0;np.add.at(m,(d['truths'][i,valid],d['predictions'][i,-1,valid]),1);im=axes[i].imshow(m/np.maximum(m.sum(1,keepdims=True),1),vmin=0,vmax=1,aspect='auto');axes[i].set(title=f'{n} classes',xlabel='Predicted',ylabel='Actual')
 fig.colorbar(im,ax=axes,label='Row proportion',shrink=.75);_finish(fig,p);return p

def _units(d,p):
 import numpy as np
 plt=_pyplot();fig,axes=plt.subplots(1,2,figsize=(12,4.5));t=np.arange(len(d['example_signal']))/float(d['sampling_rate_hz']);axes[0].plot(t,d['example_signal']);axes[0].set(xlabel='Time (s)',ylabel='EEG amplitude (µV, inferred)')
 centers=(d['unit_edges'][:-1]+d['unit_edges'][1:])/2;axes[1].plot(centers,d['unit_hist']);axes[1].set(xlabel='EEG amplitude (µV, inferred)',ylabel='Samples',yscale='log');axes[1].axvline(float(d['raw_mean']),color='.4',ls='--');_finish(fig,p);return p

def _per_frequency(d,p):
 import numpy as np
 plt=_pyplot();fig,axes=plt.subplots(1,5,figsize=(16,4),sharey=True)
 for i,n in enumerate(d['class_counts']):
  valid=d['truths'][i]>=0;truth=d['truths'][i,valid];pred=d['predictions'][i,:,valid];values=np.stack([np.mean(pred[:,truth==c]==c,axis=1) for c in range(n)])*100;im=axes[i].imshow(values,origin='lower',aspect='auto',vmin=0,vmax=100);axes[i].set(title=f'{n} classes',xlabel='Duration index',ylabel='Target class')
 fig.colorbar(im,ax=axes,label='Accuracy (%)',shrink=.75);_finish(fig,p);return p

def _solver(d,p):
 plt=_pyplot();fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for si,name in enumerate(d['solver_names']):
  for ui,substeps in enumerate(d['solver_substeps']):axes[0].plot(d['solver_input_frequencies_hz'],d['solver_spike_rates'][si,ui],label=f'{name}, {substeps} step(s)',alpha=.8)
 for si,name in enumerate(d['solver_names']):axes[1].plot(d['solver_substeps'],d['solver_mae_spikes_per_second'][si],marker='o',label=str(name))
 axes[0].set(xlabel='Input sinusoid frequency (Hz)',ylabel='Spike rate (spikes/s)');axes[0].axvline(16,color='.4',ls='--');axes[0].legend(frameon=False,ncols=2)
 axes[1].set(xlabel='Internal substeps per 1 ms sample',ylabel='MAE from exact 8-step reference (spikes/s)',xscale='log',yscale='log',xticks=d['solver_substeps']);axes[1].get_xaxis().set_major_formatter('{x:g}');axes[1].legend(frameon=False);axes[0].grid(alpha=.2);axes[1].grid(alpha=.2);_finish(fig,p);return p
