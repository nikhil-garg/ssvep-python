"""Compare 1, 2, 4, and 8 equally spaced R&F voters around every target."""
from pathlib import Path
import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import OscillatorBankClassifier,ResonateAndFireParameters
from ssvep_toolkit.visualization import render_resonate_and_fire_voting

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/"outputs/experiments/resonate_and_fire_class_scaling/pilot_subject_01"
prior_path=base/"class_scaling_blocks_1-4_tune_5-8_validate_9-12_test.npz"
raw_path=ROOT/"outputs/experiments/resonate_and_fire_all_1-60hz/pilot_subject_01/cache/raw_o1_oz_o2_1000hz.npz"
with np.load(prior_path) as prior:
    selected=prior["selected_parameters"]; durations=prior["durations_seconds"]; class_sets=[tuple(map(int,text.split(','))) for text in prior["class_frequencies"]]
all_data=np.load(raw_path)["data"][0]; stops=np.rint(1000*durations).astype(int); voter_counts=np.array((1,2,4,8)); accuracy=np.zeros((len(class_sets),len(voter_counts),len(durations)))
for ci,frequencies in enumerate(class_sets):
    values=all_data[np.asarray(frequencies)-1]; count=len(frequencies); train=values[:,:8].reshape(-1,3,5000); train_y=np.repeat(np.arange(count),8); test=values[:,8:].reshape(-1,3,5000); test_y=np.repeat(np.arange(count),4)
    alpha,threshold=selected[ci]; parameters=ResonateAndFireParameters(damping_alpha=float(alpha),threshold=float(threshold),integration_substeps=4,refractory_cycles=.5)
    for vi,voter_count in enumerate(voter_counts):
        offsets=np.array([0.]) if voter_count==1 else np.linspace(-.5,.5,voter_count)
        voter_predictions=[]; decision_sum=np.zeros((len(durations),len(test_y),count))
        for offset in offsets:
            model=OscillatorBankClassifier(frequencies,1000,parameters,spread_hz=(float(offset),),harmonics=(1,2,3)).fit_scaler(train)
            model.fit_calibration(train,train_y,stops); decision=model.decision_scores(test,stops); decision_sum+=decision; voter_predictions.append(np.argmax(decision,axis=-1))
        voter_predictions=np.asarray(voter_predictions); final=np.empty(voter_predictions.shape[1:],dtype=int)
        for di in range(len(durations)):
            for trial in range(len(test_y)):
                votes=np.bincount(voter_predictions[:,di,trial],minlength=count); tied=np.flatnonzero(votes==votes.max()); final[di,trial]=tied[np.argmax(decision_sum[di,trial,tied])]
        accuracy[ci,vi]=np.mean(final==test_y[None,:],axis=1); print(count,"classes",voter_count,"voters",np.round(100*accuracy[ci,vi],2).tolist())
output=base/"spread_voting_1_2_4_8_neurons.npz"
np.savez_compressed(output,class_counts=np.array([len(x) for x in class_sets]),class_frequencies=np.array([','.join(map(str,x)) for x in class_sets]),voter_counts=voter_counts,durations_seconds=durations,accuracy=accuracy,spread_half_width_hz=.5,selected_parameters=selected,evaluation_design="subject_01_blocks_1-8_train_9-12_test_training_only_templates")
render_resonate_and_fire_voting(output,base/"figures/spread_voting"); print(output)
