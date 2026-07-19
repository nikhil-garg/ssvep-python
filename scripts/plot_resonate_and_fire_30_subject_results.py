"""Render aggregate evidence for the 30-subject apparent-accuracy run."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/resonate_and_fire_30_subject_apparent_accuracy"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(OUT / "all_30_subjects_apparent_accuracy.npz") as source:
        d = {key: source[key] for key in source.files}
    acc = d["accuracy"]; counts = d["class_counts"]; durations = d["durations_seconds"]
    subjects = d["subject_ids"]; harmonic_choice = d["selected_harmonic_index"]
    harmonic_best = np.zeros((30, 5, 3, 10))
    for si, subject in enumerate(subjects):
        for ci, count in enumerate(counts):
            with np.load(OUT / "checkpoints" / f"subject_{subject:02d}_{count:02d}_classes.npz") as cell:
                candidate_accuracy = cell["candidate_accuracy"]
            for hi in range(3):
                rows = np.arange(hi, len(candidate_accuracy), 3)
                objectives = candidate_accuracy[rows].mean(axis=1) + 1e-3*candidate_accuracy[rows, -1]
                harmonic_best[si, ci, hi] = candidate_accuracy[rows[np.argmax(objectives)]]
    np.savez_compressed(OUT / "harmonic_comparison.npz", harmonic_best_accuracy=harmonic_best,
                        subject_ids=subjects, class_counts=counts, durations_seconds=durations,
                        harmonic_labels=np.array(("f", "f+2f", "f+2f+3f")))

    plt.style.use("seaborn-v0_8-ticks")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for ci, count in enumerate(counts):
        mean = np.nanmean(acc[:, ci], axis=0); sem = np.nanstd(acc[:, ci], axis=0, ddof=1)/np.sqrt(30)
        ax.plot(durations, 100*mean, marker="o", label=f"{count} classes")
        ax.fill_between(durations, 100*(mean-sem), 100*(mean+sem), alpha=.12)
    ax.set(xlabel="Epoch duration (s)", ylabel="Mean apparent accuracy (%)", ylim=(0, 102))
    ax.legend(frameon=False, ncols=2); fig.tight_layout(); fig.savefig(FIG/"01_accuracy_vs_duration.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    positions=np.arange(len(counts)); data=[100*acc[:,ci,-1] for ci in range(len(counts))]
    ax.boxplot(data, positions=positions, showmeans=True)
    ax.plot(positions, 100/counts, "k--o", label="Chance")
    ax.set(xlabel="Number of classes", ylabel="Subject-wise apparent accuracy at 5 s (%)",
           xticks=positions, xticklabels=counts, ylim=(0,102)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG/"02_subject_distribution_5s.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True); axes=axes.ravel()
    for ci, count in enumerate(counts):
        image=axes[ci].imshow(100*acc[:,ci], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100,
                              extent=(durations[0],durations[-1],.5,30.5))
        axes[ci].set_title(f"{count} classes"); axes[ci].set_xlabel("Epoch (s)")
    axes[0].set_ylabel("Subject"); axes[3].set_ylabel("Subject"); axes[5].axis("off")
    fig.colorbar(image, ax=axes.tolist(), label="Apparent accuracy (%)", shrink=.8)
    fig.savefig(FIG/"03_subject_duration_heatmaps.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), sharey=True)
    labels=("f", "f+2f", "f+2f+3f")
    for ci, count in enumerate(counts):
        means=100*harmonic_best[:,ci,:,-1].mean(axis=0)
        axes[ci].bar(np.arange(3),means); axes[ci].set_title(f"{count} classes")
        axes[ci].set_xticks(np.arange(3),labels,rotation=35,ha="right"); axes[ci].grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Mean optimized accuracy at 5 s (%)")
    fig.tight_layout(); fig.savefig(FIG/"04_harmonic_bank_comparison.png", dpi=180); plt.close(fig)

    choice_counts=np.stack([(harmonic_choice==hi).sum(axis=0) for hi in range(3)],axis=1)
    fig, ax=plt.subplots(figsize=(8.5,5)); bottom=np.zeros(len(counts))
    for hi,label in enumerate(labels):
        ax.bar(np.arange(len(counts)),choice_counts[:,hi],bottom=bottom,label=label);bottom+=choice_counts[:,hi]
    ax.set(xlabel="Number of classes",ylabel="Subjects selecting harmonic bank",xticks=np.arange(len(counts)),xticklabels=counts,ylim=(0,30))
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/"05_selected_harmonic_banks.png",dpi=180);plt.close(fig)

    fig, ax=plt.subplots(figsize=(8.5,5));
    for ci,count in enumerate(counts):ax.scatter(100*acc[:,ci,-1],d["overlap_coefficient_5s"][:,ci],label=f"{count}",alpha=.75)
    ax.set(xlabel="Apparent accuracy at 5 s (%)",ylabel="Target/impostor overlap coefficient",ylim=(0,1.02));ax.legend(title="Classes",frameon=False,ncols=2)
    fig.tight_layout();fig.savefig(FIG/"06_accuracy_vs_score_overlap.png",dpi=180);plt.close(fig)

    print("mean_5s",np.round(100*np.nanmean(acc[:,:,-1],axis=0),2))
    print("median_5s",np.round(100*np.nanmedian(acc[:,:,-1],axis=0),2))
    print("mean_overlap",np.round(np.nanmean(d["overlap_coefficient_5s"],axis=0),3))
    print("harmonic_choice_counts",choice_counts.tolist())

if __name__ == "__main__": main()
