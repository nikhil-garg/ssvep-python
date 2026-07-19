"""Tune and test R&F banks with 2, 4, 8, 16, and 32 classes on subject 1."""
from pathlib import Path
import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters,simulate_bank
from ssvep_toolkit.evaluation.resonate_and_fire_experiment import _fft_baseline
from ssvep_toolkit.visualization import render_resonate_and_fire_scaling

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/"outputs/experiments/resonate_and_fire_all_1-60hz/pilot_subject_01/cache/raw_o1_oz_o2_1000hz.npz"
if not source.exists(): raise FileNotFoundError(f"Run the all-frequency pilot once to create {source}")
all_data=np.load(source)["data"][0]
class_sets=((16,20),tuple(range(12,16)),tuple(range(12,20)),tuple(range(8,24)),tuple(range(8,40)))
durations=np.arange(.5,5.01,.5); stops=np.rint(1000*durations).astype(int)
damping_grid=np.array((.1,.2,.3,.4)); threshold_grid=np.array((.002,.005,.01,.02,.05,.1,.2))
parameter_scores=np.zeros((len(class_sets),len(damping_grid),len(threshold_grid)))
selected=np.zeros((len(class_sets),2)); accuracy=np.zeros((len(class_sets),len(durations))); fft_accuracy=np.zeros_like(accuracy)
predictions=[]
for set_index,frequencies in enumerate(class_sets):
    indexes=np.asarray(frequencies)-1; values=all_data[indexes]; count=len(frequencies)
    tune_x=values[:,:4].reshape(-1,3,5000); tune_y=np.repeat(np.arange(count),4)
    validation_x=values[:,4:8].reshape(-1,3,5000); validation_y=np.repeat(np.arange(count),4)
    for ai,alpha in enumerate(damping_grid):
        for ti,threshold in enumerate(threshold_grid):
            parameters=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),integration_substeps=4,refractory_cycles=.5)
            model=OscillatorBankClassifier(frequencies,1000,parameters,spread_hz=(-.5,0,.5),harmonics=(1,2,3)).fit_scaler(tune_x)
            model.fit_calibration(tune_x,tune_y,1000)
            parameter_scores[set_index,ai,ti]=np.mean(model.predict(validation_x,(1000,))[0]==validation_y)
    best=np.unravel_index(np.argmax(parameter_scores[set_index]),parameter_scores[set_index].shape)
    alpha=float(damping_grid[best[0]]); threshold=float(threshold_grid[best[1]]); selected[set_index]=(alpha,threshold)
    train_x=values[:,:8].reshape(-1,3,5000); train_y=np.repeat(np.arange(count),8); test_x=values[:,8:].reshape(-1,3,5000); test_y=np.repeat(np.arange(count),4)
    parameters=ResonateAndFireParameters(damping_alpha=alpha,threshold=threshold,integration_substeps=4,refractory_cycles=.5)
    model=OscillatorBankClassifier(frequencies,1000,parameters,spread_hz=(-.5,0,.5),harmonics=(1,2,3)).fit_scaler(train_x)
    model.fit_calibration(train_x,train_y,stops); predicted=model.predict(test_x,stops); predictions.append(predicted)
    accuracy[set_index]=np.mean(predicted==test_y[None,:],axis=1)
    fft=_fft_baseline(values[None],frequencies,1000,durations)[:,0,:,8:].reshape(len(durations),-1)
    fft_accuracy[set_index]=np.mean(fft==test_y[None,:],axis=1)
    print(count,"classes",frequencies,"params",selected[set_index].tolist(),"accuracy",np.round(100*accuracy[set_index],2).tolist())

# Directly test how threshold changes the tuning curve of a 16 Hz neuron.
input_frequencies=np.arange(12,20.001,.1); time=np.arange(5000)/1000
signals=np.sin(2*np.pi*input_frequencies[:,None]*time)[...,None,:].reshape(len(input_frequencies),1,5000)
specificity=np.zeros((len(threshold_grid),len(input_frequencies)))
for ti,threshold in enumerate(threshold_grid):
    parameters=ResonateAndFireParameters(damping_alpha=.2,threshold=float(threshold),integration_substeps=4,refractory_cycles=.5)
    specificity[ti]=simulate_bank(signals,(16,),1000,parameters,(5000,))[0,:,0]/4.9

output_dir=ROOT/"outputs/experiments/resonate_and_fire_class_scaling/pilot_subject_01"; output_dir.mkdir(parents=True,exist_ok=True)
result=output_dir/"class_scaling_blocks_1-4_tune_5-8_validate_9-12_test.npz"
np.savez_compressed(result,class_counts=np.array([len(x) for x in class_sets]),class_frequencies=np.array([",".join(map(str,x)) for x in class_sets]),durations_seconds=durations,
    accuracy=accuracy,fft_accuracy=fft_accuracy,selected_parameters=selected,parameter_scores=parameter_scores,damping_grid=damping_grid,threshold_grid=threshold_grid,
    specificity_input_frequencies_hz=input_frequencies,specificity_resonance_hz=16.,specificity_spike_rates=specificity,
    evaluation_design="subject_01_blocks_1-4_tune_5-8_validate_9-12_test")
render_resonate_and_fire_scaling(result,output_dir/"figures"); print(result)
