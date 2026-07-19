"""Controlled 8--39 Hz class scaling with nested block-wise parameter tuning."""
from pathlib import Path
import numpy as np
from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters,simulate_bank
from ssvep_toolkit.evaluation.resonate_and_fire_experiment import _fft_baseline
from ssvep_toolkit.visualization.resonate_and_fire_controlled import render_controlled_suite

ROOT=Path(__file__).resolve().parents[1]; raw_path=ROOT/"outputs/experiments/resonate_and_fire_all_1-60hz/pilot_subject_01/cache/raw_o1_oz_o2_1000hz.npz"
all_data=np.load(raw_path)["data"][0]; counts=np.array((2,4,8,16,32)); class_sets=[tuple(np.rint(np.linspace(8,39,n)).astype(int)) for n in counts]
durations=np.arange(.5,5.01,.5); stops=np.rint(1000*durations).astype(int); tuning_stop=1000
alphas=np.array((.1,.2,.3)); thresholds=np.array((.005,.01,.02,.05,.1)); voters=np.array((1,2,4,8)); half_widths=np.array((.1,.25,.5,1.,2.))
structure_scores=np.zeros((len(counts),len(half_widths),len(voters))); parameter_scores=np.zeros((len(counts),len(alphas),len(thresholds))); selected=np.zeros((len(counts),4)); accuracy=np.zeros((len(counts),len(durations))); fft_accuracy=np.zeros_like(accuracy)
max_trials=counts.max()*4; predictions=np.full((len(counts),len(durations),max_trials),-1,int); truths=np.full((len(counts),max_trials),-1,int)

def vote_models(frequencies,parameters,voter_count,half_width,train,train_y,test,calibration_stops):
    offsets=np.array([0.]) if voter_count==1 else np.linspace(-half_width,half_width,int(voter_count)); decisions=[]
    for offset in offsets:
        model=OscillatorBankClassifier(frequencies,1000,parameters,spread_hz=(float(offset),),harmonics=(1,2,3),harmonic_weights=(1,.5,1/3)).fit_scaler(train)
        model.fit_calibration(train,train_y,calibration_stops); decisions.append(model.decision_scores(test,calibration_stops))
    decisions=np.asarray(decisions); individual=np.argmax(decisions,axis=-1); summed=decisions.sum(0); final=np.empty(individual.shape[1:],int)
    for di in range(final.shape[0]):
        for trial in range(final.shape[1]):
            ballot=np.bincount(individual[:,di,trial],minlength=len(frequencies)); tied=np.flatnonzero(ballot==ballot.max()); final[di,trial]=tied[np.argmax(summed[di,trial,tied])]
    return final

for ci,frequencies in enumerate(class_sets):
    values=all_data[np.asarray(frequencies)-1]; n=len(frequencies); tune=values[:,:4].reshape(-1,3,5000); tune_y=np.repeat(np.arange(n),4); validation=values[:,4:8].reshape(-1,3,5000); validation_y=np.repeat(np.arange(n),4)
    # Stage A: select bank span and voter count without adapting neuron dynamics.
    central=ResonateAndFireParameters(damping_alpha=.2,threshold=.02,integration_substeps=4,refractory_cycles=.5)
    for wi,half_width in enumerate(half_widths):
      for vi,voter_count in enumerate(voters):
        structure_scores[ci,wi,vi]=np.mean(vote_models(frequencies,central,voter_count,float(half_width),tune,tune_y,validation,(tuning_stop,))[0]==validation_y)
    structure_best=np.unravel_index(np.argmax(structure_scores[ci]),structure_scores[ci].shape); half_width=float(half_widths[structure_best[0]]); voter_count=int(voters[structure_best[1]])
    # Stage B: tune damping and threshold jointly for the selected bank.
    for ai,alpha in enumerate(alphas):
      for ti,threshold in enumerate(thresholds):
        p=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),integration_substeps=4,refractory_cycles=.5)
        parameter_scores[ci,ai,ti]=np.mean(vote_models(frequencies,p,voter_count,half_width,tune,tune_y,validation,(tuning_stop,))[0]==validation_y)
    best=np.unravel_index(np.argmax(parameter_scores[ci]),parameter_scores[ci].shape); alpha=float(alphas[best[0]]); threshold=float(thresholds[best[1]]); selected[ci]=(voter_count,half_width,alpha,threshold)
    train=values[:,:8].reshape(-1,3,5000); train_y=np.repeat(np.arange(n),8); test=values[:,8:].reshape(-1,3,5000); test_y=np.repeat(np.arange(n),4); p=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5)
    predicted=vote_models(frequencies,p,voter_count,half_width,train,train_y,test,stops); accuracy[ci]=np.mean(predicted==test_y[None,:],axis=1); predictions[ci,:,:len(test_y)]=predicted; truths[ci,:len(test_y)]=test_y
    fft=_fft_baseline(values[None],frequencies,1000,durations)[:,0,:,8:].reshape(len(durations),-1); fft_accuracy[ci]=np.mean(fft==test_y[None,:],axis=1)
    print(n,"classes",frequencies,"selected",selected[ci].tolist(),"R&F",np.round(100*accuracy[ci],2).tolist())

# Unit audit. MAT epochs have no unit attribute; magnitude is consistent with microvolts.
flat=all_data.ravel(); unit_edges=np.linspace(-50,50,201); unit_hist,_=np.histogram(flat,bins=unit_edges); channel_std=np.std(all_data,axis=(0,1,3)); example=all_data[15,0,1,:1000]
# Solver/event convergence on a dense deterministic frequency sweep.
solver_inputs=np.arange(12,20.001,.1); solver_time=np.arange(3000)/1000; solver_signals=np.sin(2*np.pi*solver_inputs[:,None]*solver_time)[:,None,:]; substep_grid=np.array((1,2,4,8)); solver_rates=np.zeros((2,len(substep_grid),len(solver_inputs)))
for si,solver in enumerate(("euler","exact")):
 for ui,substeps in enumerate(substep_grid):
  q=ResonateAndFireParameters(damping_alpha=.2,threshold=.05,integration_substeps=int(substeps),refractory_cycles=.5,solver=solver)
  solver_rates[si,ui]=simulate_bank(solver_signals,(16,),1000,q,(3000,))[0,:,0]/2.9
reference=solver_rates[1,-1];solver_mae=np.mean(abs(solver_rates-reference[None,None,:]),axis=-1)
outdir=ROOT/"outputs/experiments/resonate_and_fire_controlled_spacing/pilot_subject_01"; outdir.mkdir(parents=True,exist_ok=True); result=outdir/"controlled_8-39hz_nested_tuning.npz"
freqpad=np.full((len(counts),counts.max()),np.nan)
for i,x in enumerate(class_sets):freqpad[i,:len(x)]=x
np.savez_compressed(result,class_counts=counts,class_frequencies_hz=freqpad,durations_seconds=durations,accuracy=accuracy,fft_accuracy=fft_accuracy,predictions=predictions,truths=truths,
    structure_validation_scores=structure_scores,parameter_validation_scores=parameter_scores,voter_grid=voters,spread_half_width_grid_hz=half_widths,damping_grid=alphas,threshold_grid=thresholds,selected_parameters=selected,frequency_range_hz=np.array((8,39)),
    harmonics=np.array((1,2,3)),harmonic_weights=np.array((1,.5,1/3)),refractory_cycles=.5,integration_substeps=4,sampling_rate_hz=1000.,raw_unit="microvolts_inferred_from_magnitude_no_MAT_attribute",
    raw_min=float(flat.min()),raw_max=float(flat.max()),raw_mean=float(flat.mean()),raw_std=float(flat.std()),channel_std=channel_std,unit_hist=unit_hist,unit_edges=unit_edges,example_signal=example,
    solver_names=np.array(("euler","exact")),solver_substeps=substep_grid,solver_input_frequencies_hz=solver_inputs,solver_spike_rates=solver_rates,solver_mae_spikes_per_second=solver_mae,solver_reference="exact_8_substeps",
    evaluation_design="subject_01_blocks_1-4_fit_5-8_validate_9-12_test")
render_controlled_suite(result,outdir/"figures"); print(result)
