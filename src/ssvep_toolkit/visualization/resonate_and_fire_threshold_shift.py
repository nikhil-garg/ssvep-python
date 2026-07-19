from pathlib import Path
from .plots import _finish,_pyplot
def render_threshold_shift(result_path,output_dir):
 import numpy as np
 output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
 with np.load(result_path) as s:d={k:s[k] for k in s.files}
 plt=_pyplot();fig,axes=plt.subplots(1,3,figsize=(15,4.5));axes[0].plot(d['input_frequencies_hz'],d['subthreshold_rms']);axes[0].axvline(d['resonance_hz'],color='.4',ls='--');axes[0].set(xlabel='Input frequency (Hz)',ylabel='Subthreshold y RMS (state units)',title='Linear oscillator')
 for mi,title in enumerate(('Legacy level/reset-to-threshold','Upward crossing/zero reset')):
  for ti,threshold in enumerate(d['thresholds']):axes[mi+1].plot(d['input_frequencies_hz'],d['spike_rates'][mi,ti],label=f'th={threshold:g}')
  axes[mi+1].axvline(d['resonance_hz'],color='.4',ls='--');axes[mi+1].set(xlabel='Input frequency (Hz)',ylabel='Spike rate (spikes/s)',title=title);axes[mi+1].legend(frameon=False)
 for ax in axes:ax.grid(alpha=.2)
 p=output_dir/'linear_vs_spiking_resonance.png';_finish(fig,p)
 fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(d['thresholds'],d['peak_frequencies_hz'][0],marker='o',label='Legacy');ax.plot(d['thresholds'],d['peak_frequencies_hz'][1],marker='o',label='Upward crossing + zero reset');ax.axhline(d['resonance_hz'],color='.4',ls='--');ax.set(xlabel='Threshold (dimensionless state)',ylabel='Frequency of maximum firing (Hz)',xscale='log');ax.legend(frameon=False);ax.grid(alpha=.2);q=output_dir/'apparent_resonance_vs_threshold.png';_finish(fig,q);return[p,q]
