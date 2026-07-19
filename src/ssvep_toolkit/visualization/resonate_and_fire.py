from __future__ import annotations

from pathlib import Path

from ssvep_toolkit.algorithms.resonate_and_fire import ResonateAndFireParameters, simulate_trace
from .plots import _finish, _pyplot


def render_resonate_and_fire_suite(result_path: str | Path, output_dir: str | Path, raw_data=None) -> list[Path]:
    """Render binary or multiclass R&F diagnostics from a saved experiment."""
    import numpy as np

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(result_path) as source:
        data = {key: source[key] for key in source.files}
    outputs = [
        _accuracy_plot(data, output_dir / "accuracy_vs_duration.png"),
        _subject_heatmap(data, output_dir / "subject_accuracy_heatmap.png"),
        _confusion_matrices(data, output_dir / "confusion_matrices.png"),
        _per_frequency_accuracy(data, output_dir / "per_frequency_accuracy.png"),
        _frequency_error(data, output_dir / "frequency_error_distribution.png"),
        _parameter_surface(data, output_dir / "parameter_validation_surface.png"),
        _margin_plot(data, output_dir / "classification_margins.png"),
        _bank_layout(data, output_dir / "oscillator_bank_layout.png"),
    ]
    if raw_data is not None:
        outputs.extend((_trace_plot(data, raw_data, output_dir / "oscillator_traces.png"),
                        _spike_raster(data, raw_data, output_dir / "spike_raster.png")))
    return outputs


def _truth(data):
    import numpy as np
    classes = len(data["frequencies_hz"])
    return np.broadcast_to(np.arange(classes)[None, None, :, None], data["predictions"].shape)


def _accuracy_plot(data, path):
    import numpy as np
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(8,4.8)); x=data["durations_seconds"]
    for key,label in (("subject_accuracy","R&F spread/harmonic bank"),("fft_subject_accuracy","FFT + first harmonic")):
        values=data[key]; mean=values.mean(0)*100; sem=(values.std(0,ddof=1)/np.sqrt(values.shape[0])*100 if values.shape[0]>1 else np.zeros_like(mean))
        line,=ax.plot(x,mean,marker="o",label=label); ax.fill_between(x,mean-sem,mean+sem,color=line.get_color(),alpha=.2)
    chance=100/len(data["frequencies_hz"]); ax.axhline(chance,color="0.5",linestyle="--",linewidth=1,label=f"Chance ({chance:.2g}%)")
    ax.set(xlabel="Epoch duration (s)",ylabel="Held-out subject accuracy (%)",ylim=(0,102)); ax.grid(alpha=.2); ax.legend(frameon=False)
    _finish(fig,path); return path


def _subject_heatmap(data,path):
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(10,7)); image=ax.imshow(data["subject_accuracy"]*100,aspect="auto",origin="lower",vmin=0,vmax=100)
    ax.set(xlabel="Epoch duration (s)",ylabel="Held-out subject",xticks=range(len(data["durations_seconds"])),xticklabels=[f"{x:g}" for x in data["durations_seconds"]]); fig.colorbar(image,ax=ax,label="Accuracy (%)"); _finish(fig,path); return path


def _confusion_matrices(data,path):
    import numpy as np
    durations=data["durations_seconds"]; wanted=(.5,1,2,3,5); classes=len(data["frequencies_hz"]); truth=_truth(data)
    plt=_pyplot(); fig,axes=plt.subplots(1,len(wanted),figsize=(16,3.5),squeeze=False)
    for ax,duration in zip(axes[0],wanted):
        idx=int(np.argmin(abs(durations-duration))); matrix=np.zeros((classes,classes),int)
        np.add.at(matrix,(truth[idx].ravel(),data["predictions"][idx].ravel()),1)
        normalized=matrix/np.maximum(matrix.sum(1,keepdims=True),1); image=ax.imshow(normalized,vmin=0,vmax=1,aspect="auto")
        ax.set(title=f"{durations[idx]:g} s",xlabel="Predicted (Hz)",ylabel="Actual (Hz)")
        ticks=np.linspace(0,classes-1,min(7,classes),dtype=int); ax.set_xticks(ticks,data["frequencies_hz"][ticks],rotation=45); ax.set_yticks(ticks,data["frequencies_hz"][ticks])
    fig.colorbar(image,ax=axes.ravel().tolist(),label="Row proportion",shrink=.75); _finish(fig,path); return path


def _per_frequency_accuracy(data,path):
    import numpy as np
    truth=_truth(data); values=(data["predictions"]==truth).mean(axis=(1,3)).T*100
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(11,6)); image=ax.imshow(values,origin="lower",aspect="auto",vmin=0,vmax=100)
    ax.set(xlabel="Epoch duration (s)",ylabel="Stimulation frequency (Hz)",xticks=range(values.shape[1]),xticklabels=[f"{x:g}" for x in data["durations_seconds"]])
    ticks=np.linspace(0,values.shape[0]-1,min(12,values.shape[0]),dtype=int); ax.set_yticks(ticks,data["frequencies_hz"][ticks]); fig.colorbar(image,ax=ax,label="Accuracy (%)"); _finish(fig,path); return path


def _frequency_error(data,path):
    import numpy as np
    frequencies=data["frequencies_hz"]; truth=_truth(data); actual=frequencies[truth]; predicted=frequencies[data["predictions"]]
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(9,4.8)); values=[]
    for index in range(len(data["durations_seconds"])):
        error=np.abs(predicted[index]-actual[index]).ravel(); values.append(error)
    ax.boxplot(values,tick_labels=[f"{x:g}" for x in data["durations_seconds"]],showfliers=False); ax.set(xlabel="Epoch duration (s)",ylabel="Absolute frequency error (Hz)"); ax.grid(axis="y",alpha=.2); _finish(fig,path); return path


def _parameter_surface(data,path):
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(8,5)); scores=data["parameter_scores"].mean(0)*100; image=ax.imshow(scores,aspect="auto",origin="lower")
    ax.set(xticks=range(len(data["threshold_grid"])),xticklabels=[f"{x:g}" for x in data["threshold_grid"]],yticks=range(len(data["damping_grid"])),yticklabels=[f"{x:g}" for x in data["damping_grid"]],xlabel="Normalized threshold",ylabel="Damping alpha")
    for r in range(scores.shape[0]):
        for c in range(scores.shape[1]): ax.text(c,r,f"{scores[r,c]:.1f}",ha="center",va="center",fontsize=8)
    fig.colorbar(image,ax=ax,label="Inner-fold accuracy (%)"); _finish(fig,path); return path


def _margin_plot(data,path):
    import numpy as np
    scores=data["spike_scores"]; truth=_truth(data); correct=np.take_along_axis(scores,truth[...,None],axis=-1)[...,0]
    masked=scores.copy(); np.put_along_axis(masked,truth[...,None],-np.inf,axis=-1); margins=(correct-masked.max(axis=-1)).reshape(len(data["durations_seconds"]),-1).T
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(9,4.8)); ax.boxplot([margins[:,i] for i in range(margins.shape[1])],tick_labels=[f"{x:g}" for x in data["durations_seconds"]],showfliers=False)
    ax.axhline(0,color="0.4",linestyle="--"); ax.set(xlabel="Epoch duration (s)",ylabel="Correct-class score margin"); ax.grid(axis="y",alpha=.2); _finish(fig,path); return path


def _bank_layout(data,path):
    import numpy as np
    frequencies=data["frequencies_hz"]; spread=data.get("spread_hz",np.array([0.])); harmonics=data.get("harmonics",np.array([1]))
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(11,5))
    for harmonic in harmonics:
        for offset in spread: ax.scatter(frequencies, harmonic*frequencies+offset,s=9,alpha=.55,label=f"h={harmonic}" if offset==spread[0] else None)
    ax.set(xlabel="Target class frequency (Hz)",ylabel="Neuron resonance frequency (Hz)"); ax.grid(alpha=.2); ax.legend(frameon=False,ncols=len(harmonics)); _finish(fig,path); return path


def _parameters(data):
    import numpy as np
    damping,threshold=np.median(data["selected_parameters"],axis=0)
    return ResonateAndFireParameters(damping_alpha=float(damping),threshold=float(threshold),integration_substeps=int(data.get("integration_substeps",4)),refractory_cycles=float(data.get("refractory_cycles",0)))


def _trace_plot(data,raw_data,path):
    import numpy as np
    parameters=_parameters(data); fs=float(data["sampling_rate_hz"]); frequencies=data["frequencies_hz"]; indexes=np.unique(np.linspace(0,len(frequencies)-1,min(3,len(frequencies)),dtype=int)); scale=np.std(raw_data,axis=(0,1,2,4),ddof=1)
    plt=_pyplot(); fig,axes=plt.subplots(len(indexes),2,figsize=(13,3.2*len(indexes)),squeeze=False); samples=min(raw_data.shape[-1],round(1.0*fs)); time=np.arange(samples)/fs
    for row,index in enumerate(indexes):
        signal=raw_data[0,index,0,1,:samples]/scale[1]; spikes,x,y=simulate_trace(signal,float(frequencies[index]),fs,parameters)
        axes[row,0].plot(time,signal); axes[row,0].vlines(spikes/fs,*axes[row,0].get_ylim(),color="tab:red",alpha=.25,linewidth=.5); axes[row,0].set(ylabel=f"{frequencies[index]:g} Hz input",xlabel="Time (s)")
        axes[row,1].plot(time,y); axes[row,1].axhline(parameters.threshold,color="tab:red",linestyle="--"); axes[row,1].scatter(spikes/fs,y[spikes],s=10,color="tab:red"); axes[row,1].set(xlabel="Time (s)",ylabel="Normalized oscillator y")
    _finish(fig,path); return path


def _spike_raster(data,raw_data,path):
    import numpy as np
    parameters=_parameters(data); fs=float(data["sampling_rate_hz"]); frequencies=data["frequencies_hz"]; actual_index=len(frequencies)//2; actual=float(frequencies[actual_index]); scale=np.std(raw_data,axis=(0,1,2,4),ddof=1); signal=raw_data[0,actual_index,0,1]/scale[1]
    spread=data.get("spread_hz",np.array([0.])); harmonics=data.get("harmonics",np.array([1])); resonances=[h*actual+o for h in harmonics for o in spread]
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(11,5))
    for row,resonance in enumerate(resonances):
        spikes,_,_=simulate_trace(signal,float(resonance),fs,parameters); ax.vlines(spikes/fs,row-.35,row+.35,linewidth=.7)
    ax.set(xlabel="Time (s)",ylabel="Neuron resonance (Hz)",yticks=range(len(resonances)),yticklabels=[f"{x:g}" for x in resonances],title=f"Spike raster for a {actual:g} Hz trial"); _finish(fig,path); return path
