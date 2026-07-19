"""Separate linear resonance from threshold/reset-induced firing resonance."""
from pathlib import Path
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters,simulate_bank,simulate_trace
from ssvep_toolkit.visualization.resonate_and_fire_threshold_shift import render_threshold_shift
ROOT=Path(__file__).resolve().parents[1];fs=1000;inputs=np.arange(12,20.001,.1);time=np.arange(5000)/fs;signals=np.sin(2*np.pi*inputs[:,None]*time)[:,None,:];thresholds=np.array((.005,.01,.02,.05,.1));rates=np.zeros((2,len(thresholds),len(inputs)))
for mi,(reset,detection) in enumerate((('notebook_compatible','level'),('zero','upward_crossing'))):
 for ti,threshold in enumerate(thresholds):
  p=ResonateAndFireParameters(damping_alpha=.1,threshold=float(threshold),integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode=reset,spike_detection=detection);rates[mi,ti]=simulate_bank(signals,(16,),fs,p,(5000,))[0,:,0]/4.9
subthreshold=np.zeros(len(inputs));p=ResonateAndFireParameters(damping_alpha=.1,threshold=1e9,integration_substeps=4,solver='exact')
for i,signal in enumerate(signals[:,0]):
 _,_,y=simulate_trace(signal,16,fs,p);subthreshold[i]=np.sqrt(np.mean(y[1000:]**2))
peaks=inputs[np.argmax(rates,axis=-1)];outdir=ROOT/'outputs/experiments/resonate_and_fire_threshold_shift';outdir.mkdir(parents=True,exist_ok=True);result=outdir/'threshold_reset_crossing_analysis.npz';np.savez_compressed(result,input_frequencies_hz=inputs,resonance_hz=16.,thresholds=thresholds,subthreshold_rms=subthreshold,spike_rates=rates,mode_names=np.array(('legacy_level_threshold_reset','upward_crossing_zero_reset')),peak_frequencies_hz=peaks)
render_threshold_shift(result,outdir/'figures');print('peaks',peaks);print(result)
