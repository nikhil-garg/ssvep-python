from __future__ import annotations
from pathlib import Path
from .plots import _finish,_pyplot

def render_focused_suite(result_path: str|Path,output_dir: str|Path):
 import numpy as np
 output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 with np.load(result_path) as s:d={k:s[k] for k in s.files}
 jobs=[('01_accuracy.png',_accuracy),('02_fft_comparison.png',_fft),('03_alpha_threshold.png',_alpha_threshold),('04_width_voters.png',_width_voters),('05_selected_parameters.png',_selected),('06_spike_tuning.png',_tuning),('07_spike_timing_metrics.png',_timing),('08_spike_raster.png',_raster)]
 return [fn(d,output_dir/name) for name,fn in jobs]

def _accuracy(d,p):
 plt=_pyplot();fig,ax=plt.subplots(figsize=(9,5))
 for i,n in enumerate(d['class_counts']):ax.plot(d['durations_seconds'],100*d['accuracy'][i],marker='o',label=f'{n} classes')
 ax.set(xlabel='Epoch duration (s)',ylabel='Held-out block accuracy (%)',ylim=(0,102));ax.grid(alpha=.2);ax.legend(frameon=False,ncols=2);_finish(fig,p);return p

def _fft(d,p):
 import numpy as np
 plt=_pyplot();fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(len(d['class_counts']));w=.36;ax.bar(x-w/2,100*d['accuracy'][:,-1],w,label='R&F');ax.bar(x+w/2,100*d['fft_accuracy'][:,-1],w,label='FFT + harmonic');ax.plot(x,100/d['class_counts'],'k--o',label='Chance');ax.set(xlabel='Classes',ylabel='5 s accuracy (%)',xticks=x,xticklabels=d['class_counts']);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);_finish(fig,p);return p

def _projection(d,xindex,yindex,xvalues,yvalues,p,xlabel,ylabel):
 import numpy as np
 plt=_pyplot();fig,axes=plt.subplots(1,5,figsize=(16,3.6),sharex=True,sharey=True)
 for ci,n in enumerate(d['class_counts']):
  z=np.full((len(yvalues),len(xvalues)),np.nan)
  for yi,y in enumerate(yvalues):
   for xi,x in enumerate(xvalues):
    mask=(d['candidates'][:,xindex]==x)&(d['candidates'][:,yindex]==y);z[yi,xi]=np.max(d['stage1_scores'][ci,mask])
  im=axes[ci].imshow(100*z,origin='lower',aspect='auto',vmin=0,vmax=100);axes[ci].set(title=f'{n} classes',xlabel=xlabel,xticks=range(len(xvalues)),xticklabels=xvalues,yticks=range(len(yvalues)),yticklabels=yvalues)
 axes[0].set_ylabel(ylabel);fig.colorbar(im,ax=axes,label='Best stage-1 validation accuracy (%)',shrink=.75);_finish(fig,p);return p

def _alpha_threshold(d,p):return _projection(d,1,0,d['threshold_grid'],d['alpha_grid'],p,'Threshold','Damping α')
def _width_voters(d,p):return _projection(d,3,2,d['voter_grid'],d['width_grid_hz'],p,'Voters','Half-width (Hz)')

def _selected(d,p):
 plt=_pyplot();fig,axes=plt.subplots(2,2,figsize=(9,7));labels=('Damping α','Threshold','Half-width (Hz)','Voters')
 for j,ax in enumerate(axes.ravel()):ax.plot(d['class_counts'],d['selected_parameters'][:,j],marker='o');ax.set(xlabel='Classes',ylabel=labels[j],xscale='log',xticks=d['class_counts']);ax.get_xaxis().set_major_formatter('{x:g}');ax.grid(alpha=.2)
 _finish(fig,p);return p

def _tuning(d,p):
 plt=_pyplot();fig,axes=plt.subplots(1,len(d['alpha_grid']),figsize=(14,4),sharey=True)
 for ai,alpha in enumerate(d['alpha_grid']):
  for ti,threshold in enumerate(d['threshold_grid']):axes[ai].plot(d['resonance_frequencies_hz'],d['spike_rates_hz'][ai,ti],label=f'th={threshold:g}')
  axes[ai].axvline(16,color='.4',ls='--');axes[ai].set(title=f'α={alpha:g}',xlabel='Neuron resonance frequency (Hz)',ylabel='Spike rate (spikes/s)');axes[ai].grid(alpha=.2)
 axes[-1].legend(frameon=False);_finish(fig,p);return p

def _timing(d,p):
 plt=_pyplot();fig,axes=plt.subplots(1,5,figsize=(16,3.6))
 labels=('Rate (spikes/s)','First-spike latency (s)','Median ISI (s)','ISI CV','Phase vector strength')
 for mi,ax in enumerate(axes):
  im=ax.imshow(d['timing_metrics'][:,:,mi],origin='lower',aspect='auto');ax.set(title=labels[mi],xlabel='Threshold',xticks=range(len(d['threshold_grid'])),xticklabels=d['threshold_grid'],yticks=range(len(d['alpha_grid'])),yticklabels=d['alpha_grid']);ax.tick_params(axis='x',rotation=45);fig.colorbar(im,ax=ax,shrink=.7)
 axes[0].set_ylabel('Damping α');_finish(fig,p);return p

def _raster(d,p):
 import numpy as np
 plt=_pyplot();fig,ax=plt.subplots(figsize=(11,4))
 for i,res in enumerate(d['raster_resonances_hz']):
  t=d['raster_times_s'][i];t=t[~np.isnan(t)];ax.vlines(t,i-.35,i+.35,lw=.7)
 ax.set(xlabel='Spike time (s)',ylabel='Neuron resonance frequency (Hz)',yticks=range(3),yticklabels=d['raster_resonances_hz']);_finish(fig,p);return p
