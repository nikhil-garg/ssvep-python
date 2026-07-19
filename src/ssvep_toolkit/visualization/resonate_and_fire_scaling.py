from __future__ import annotations

from pathlib import Path

from .plots import _finish, _pyplot


def render_resonate_and_fire_scaling(result_path: str | Path, output_dir: str | Path) -> list[Path]:
    import numpy as np
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    with np.load(result_path) as source: data={key:source[key] for key in source.files}
    return [_scaling(data,output_dir/"accuracy_by_class_count.png"),
            _thresholds(data,output_dir/"selected_threshold_by_class_count.png"),
            _validation(data,output_dir/"threshold_validation_curves.png"),
            _specificity(data,output_dir/"threshold_specificity.png")]


def render_resonate_and_fire_voting(result_path: str | Path, output_dir: str | Path) -> list[Path]:
    import numpy as np
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    with np.load(result_path) as source: data={key:source[key] for key in source.files}
    plt=_pyplot(); fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True); axes=axes.ravel()
    for ci,count in enumerate(data["class_counts"]):
        for vi,voters in enumerate(data["voter_counts"]): axes[ci].plot(data["durations_seconds"],100*data["accuracy"][ci,vi],marker="o",label=f"{voters} neurons")
        axes[ci].axhline(100/count,color="0.5",linestyle="--",linewidth=1); axes[ci].set_title(f"{count} classes"); axes[ci].grid(alpha=.2)
    axes[5].axis("off"); axes[0].set_ylabel("Accuracy (%)"); axes[3].set_ylabel("Accuracy (%)"); axes[3].set_xlabel("Epoch duration (s)"); axes[4].set_xlabel("Epoch duration (s)"); axes[2].legend(frameon=False)
    first=output_dir/"accuracy_by_voter_count.png"; _finish(fig,first)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,duration in zip(axes,(1.,5.)):
        di=int(np.argmin(abs(data["durations_seconds"]-duration))); image=ax.imshow(100*data["accuracy"][:,:,di],aspect="auto",vmin=0,vmax=100)
        ax.set(title=f"{duration:g} s",xlabel="Neurons around target",xticks=range(len(data["voter_counts"])),xticklabels=data["voter_counts"],yticks=range(len(data["class_counts"])),yticklabels=data["class_counts"])
    axes[0].set_ylabel("Classes"); fig.colorbar(image,ax=axes,label="Accuracy (%)"); second=output_dir/"voting_accuracy_heatmaps.png"; _finish(fig,second)
    return [first,second]


def _scaling(data,path):
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(9,5))
    for index,count in enumerate(data["class_counts"]):
        ax.plot(data["durations_seconds"],100*data["accuracy"][index],marker="o",label=f"{count} classes")
        ax.axhline(100/count,color="0.6",linewidth=.5,alpha=.35)
    ax.set(xlabel="Epoch duration (s)",ylabel="Held-out block accuracy (%)",ylim=(0,102)); ax.grid(alpha=.2); ax.legend(frameon=False,ncols=2); _finish(fig,path); return path


def _thresholds(data,path):
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(data["class_counts"],data["selected_parameters"][:,1],marker="o")
    ax.set(xlabel="Number of classes",ylabel="Selected firing threshold",xscale="log",yscale="log",xticks=data["class_counts"]); ax.get_xaxis().set_major_formatter("{x:g}"); ax.grid(alpha=.2); _finish(fig,path); return path


def _validation(data,path):
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(9,5)); scores=data["parameter_scores"].max(axis=1)
    for index,count in enumerate(data["class_counts"]): ax.plot(data["threshold_grid"],100*scores[index],marker="o",label=f"{count} classes")
    ax.set(xlabel="Firing threshold",ylabel="Best validation accuracy over damping (%)",xscale="log"); ax.grid(alpha=.2); ax.legend(frameon=False,ncols=2); _finish(fig,path); return path


def _specificity(data,path):
    import numpy as np
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(9,5)); responses=data["specificity_spike_rates"]
    for index,threshold in enumerate(data["threshold_grid"]):
        peak=max(float(responses[index].max()),1e-12); ax.plot(data["specificity_input_frequencies_hz"],responses[index]/peak, label=f"th={threshold:g}")
    ax.axvline(float(data["specificity_resonance_hz"]),color="0.4",linestyle="--",linewidth=1)
    ax.set(xlabel="Input sinusoid frequency (Hz)",ylabel="Normalized spike rate",ylim=(-.02,1.05)); ax.grid(alpha=.2); ax.legend(frameon=False,ncols=2); _finish(fig,path); return path
