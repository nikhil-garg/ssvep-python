"""Six concrete state/spike figures for corrected R&F dynamics."""
from pathlib import Path
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters,simulate_bank,simulate_trace
from ssvep_toolkit.visualization.resonate_and_fire_evidence import render_evidence_suite
ROOT=Path(__file__).resolve().parents[1];fs=1000;time=np.arange(5000)/fs;resonators=np.array((15.,16.,17.));p_good=ResonateAndFireParameters(damping_alpha=.1,threshold=.1,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing')
synthetic=np.sin(2*np.pi*16*time);sx=np.zeros((3,5000));sy=np.zeros_like(sx);sspikes=np.full((3,1000),np.nan)
for i,res in enumerate(resonators):
 sp,x,y=simulate_trace(synthetic,res,fs,p_good);sx[i]=x;sy[i]=y;sspikes[i,:min(len(sp),1000)]=sp[:1000]/fs
input_freqs=np.arange(12,20.001,.1);signals=np.sin(2*np.pi*input_freqs[:,None]*time)[:,None,:];thresholds=np.array((.02,.05,.1));synthetic_rates=np.zeros((3,len(input_freqs)))
for i,th in enumerate(thresholds):
 p=ResonateAndFireParameters(damping_alpha=.1,threshold=float(th),integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing');synthetic_rates[i]=simulate_bank(signals,(16,),fs,p,(5000,))[0,:,0]/4.9
# Held-out subject 5, block 9, 16 Hz, Oz; scaling uses only S5 blocks 1-8.
data=np.load(ROOT/'outputs/experiments/resonate_and_fire_five_subject_pilot/cache/subjects_01-05_8-39hz_o1_oz_o2_1000hz.npz')['data'];scale=np.std(data[4,:,0:8],axis=(0,1,3),ddof=1);real=(data[4,16-8,8,1]-data[4,16-8,8,1].mean())/scale[1];p_real=ResonateAndFireParameters(damping_alpha=.05,threshold=.02,integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing');rx=np.zeros_like(sx);ry=np.zeros_like(sy);rspikes=np.full((3,1000),np.nan)
for i,res in enumerate(resonators):
 sp,x,y=simulate_trace(real,res,fs,p_real);rx[i]=x;ry[i]=y;rspikes[i,:min(len(sp),1000)]=sp[:1000]/fs
real_res=np.arange(12,20.001,.1);real_rates=np.zeros((len(thresholds),len(real_res)))
for i,th in enumerate(thresholds):
 p=ResonateAndFireParameters(damping_alpha=.05,threshold=float(th),integration_substeps=4,refractory_cycles=.5,solver='exact',reset_mode='zero',spike_detection='upward_crossing');real_rates[i]=simulate_bank(real[None,None,:],real_res,fs,p,(5000,))[0,0]/4.9
shift=np.load(ROOT/'outputs/experiments/resonate_and_fire_threshold_shift/threshold_reset_crossing_analysis.npz');outdir=ROOT/'outputs/experiments/resonate_and_fire_evidence';outdir.mkdir(parents=True,exist_ok=True);result=outdir/'corrected_internal_states_and_spikes.npz'
np.savez_compressed(result,time_s=time,resonator_frequencies_hz=resonators,synthetic_signal=synthetic,synthetic_x=sx,synthetic_y=sy,synthetic_spike_times_s=sspikes,input_frequencies_hz=input_freqs,thresholds=thresholds,synthetic_spike_rates_hz=synthetic_rates,subthreshold_rms=shift['subthreshold_rms'],real_signal_standardized=real,real_x=rx,real_y=ry,real_spike_times_s=rspikes,real_resonances_hz=real_res,real_spike_rates_hz=real_rates,real_signal_unit='subject5_training_standard_deviations',state_unit='dimensionless',spike_rule='upward_crossing_zero_reset')
render_evidence_suite(result,outdir/'figures');print(result)
