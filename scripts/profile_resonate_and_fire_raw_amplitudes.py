"""Profile raw 1 s occipital amplitudes and target-frequency resonance strength."""
from pathlib import Path
import numpy as np
from ssvep_toolkit.data.matlab import Matlab73Dataset

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT.parent
OUT=ROOT/'outputs/experiments/resonate_and_fire_gain_optimization';OUT.mkdir(parents=True,exist_ok=True)
subjects=np.arange(1,31);freqs=np.arange(8,40);channels=('O1','Oz','O2');fs=1000
rms=np.zeros((30,32,12,3));ptp=np.zeros_like(rms);snr=np.zeros((30,32,12,3))
for si,sid in enumerate(subjects):
 print(f'profiling subject {sid}/30',flush=True)
 with Matlab73Dataset(DATA/f'data_s{sid}_64.mat') as source: chunk=source.read_channel_chunk(60,64)
 # condition 2, channels O1/Oz/O2, 140 ms latency, first 1 s; -> f,b,c,t
 x=chunk[1,:3,140:1140,7:39,:].transpose(2,3,0,1).astype(float)
 x=x-x.mean(axis=-1,keepdims=True);rms[si]=np.sqrt(np.mean(x*x,axis=-1));
 q=np.percentile(x,(5,95),axis=-1);ptp[si]=q[1]-q[0]
 spec=np.abs(np.fft.rfft(x,axis=-1));bins=np.fft.rfftfreq(1000,1/fs)
 for fi,f in enumerate(freqs):
  target=np.argmin(abs(bins-f));neighbors=np.r_[max(1,target-3):max(1,target-1),target+2:target+4]
  noise=np.median(np.take(spec[fi],neighbors,axis=-1),axis=-1)
  snr[si,fi]=spec[fi,:,:,target]/np.maximum(noise,1e-12)
subject_median_rms=np.median(rms,axis=(1,2,3));order=np.argsort(subject_median_rms);median=np.median(subject_median_rms)
diverse=np.array((order[0],np.argmin(abs(subject_median_rms-median)),order[-1]))
similar=np.array([index for index in np.argsort(abs(subject_median_rms-median)) if index not in set(diverse)][:4])
selected=np.unique(np.r_[diverse,similar]);bands=np.array(((8,10),(22,24),(37,39)))
np.savez_compressed(OUT/'raw_amplitude_and_resonance_profile.npz',subject_ids=subjects,frequencies_hz=freqs,channel_names=channels,
 rms_uV=rms,robust_peak_to_peak_uV=ptp,target_spectral_snr=snr,subject_median_rms_uV=subject_median_rms,
 diverse_subject_ids=subjects[diverse],similar_subject_ids=subjects[similar],selected_subject_ids=subjects[selected],frequency_bands_hz=bands,
 raw_unit_inference='microvolts_from_dataset_scale_not_explicit_mat_attribute')
print('diverse',subjects[diverse].tolist(),np.round(subject_median_rms[diverse],3).tolist())
print('similar',subjects[similar].tolist(),np.round(subject_median_rms[similar],3).tolist())
