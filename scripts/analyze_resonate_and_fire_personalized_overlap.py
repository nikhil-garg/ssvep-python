"""Subject-wise held-out accuracy and score-overlap diagnostics for R&F.

The neuron parameters are fixed before evaluation.  For each subject, blocks
1--8 fit the channel scale and class templates; blocks 9--12 are never used
until the final evaluation.
"""
from pathlib import Path

import numpy as np

from ssvep_toolkit.algorithms.resonate_and_fire import (
    OscillatorBankClassifier,
    ResonateAndFireParameters,
)
from ssvep_toolkit.evaluation.resonate_and_fire_experiment import _fft_baseline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/resonate_and_fire_personalized_overlap"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    edges = np.histogram_bin_edges(np.r_[a, b], bins=bins)
    ha, _ = np.histogram(a, edges, density=True)
    hb, _ = np.histogram(b, edges, density=True)
    return float(np.sum(np.minimum(ha, hb) * np.diff(edges)))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = ROOT / "outputs/experiments/resonate_and_fire_five_subject_pilot/cache/subjects_01-05_8-39hz_o1_oz_o2_1000hz.npz"
    with np.load(cache) as source:
        all_data = source["data"].astype(float)

    frequencies = np.rint(np.linspace(8, 39, 8)).astype(int)
    data = all_data[:, frequencies - 8]
    durations = np.arange(0.5, 5.01, 0.5)
    stops = np.rint(1000 * durations).astype(int)
    n_subjects, n_classes, _, n_channels, n_samples = data.shape
    train_blocks = np.arange(8)
    test_blocks = np.arange(8, 12)
    parameters = ResonateAndFireParameters(
        damping_alpha=0.05,
        threshold=0.02,
        integration_substeps=4,
        refractory_cycles=0.5,
        solver="exact",
        reset_mode="zero",
        spike_detection="upward_crossing",
    )

    subject_accuracy = np.zeros((n_subjects, len(durations)))
    fft_subject_accuracy = np.zeros_like(subject_accuracy)
    predictions = np.zeros((len(durations), n_subjects, n_classes, len(test_blocks)), dtype=int)
    scores = np.zeros(predictions.shape + (n_classes,), dtype=np.float32)
    truth = np.broadcast_to(np.arange(n_classes)[:, None], (n_classes, len(test_blocks)))

    for subject in range(n_subjects):
        train = data[subject, :, train_blocks].transpose(1, 0, 2, 3).reshape(-1, n_channels, n_samples)
        train_y = np.repeat(np.arange(n_classes), len(train_blocks))
        test = data[subject, :, test_blocks].transpose(1, 0, 2, 3).reshape(-1, n_channels, n_samples)
        model = OscillatorBankClassifier(
            frequencies,
            1000.0,
            parameters,
            spread_hz=(0.0,),
            harmonics=(1, 2, 3),
            harmonic_weights=(1.0, 0.5, 1 / 3),
        ).fit_scaler(train)
        model.fit_calibration(train, train_y, stops)
        subject_scores = model.decision_scores(test, stops)
        scores[:, subject] = subject_scores.reshape(len(durations), n_classes, len(test_blocks), n_classes)
        predictions[:, subject] = np.argmax(scores[:, subject], axis=-1)
        subject_accuracy[subject] = np.mean(predictions[:, subject] == truth[None], axis=(1, 2))

        fft = _fft_baseline(data[subject : subject + 1, :, test_blocks], frequencies, 1000.0, durations)
        fft_pred = fft[:, 0]
        fft_subject_accuracy[subject] = np.mean(fft_pred == truth[None], axis=(1, 2))
        print(f"Subject {subject + 1}: R&F 5 s {100*subject_accuracy[subject,-1]:.2f}%")

    accuracy = subject_accuracy.mean(axis=0)
    fft_accuracy = fft_subject_accuracy.mean(axis=0)
    last_scores = scores[-1]
    target_scores = np.take_along_axis(last_scores, truth[None, ..., None], axis=-1)[..., 0]
    masked = last_scores.copy()
    for class_index in range(n_classes):
        masked[:, class_index, :, class_index] = -np.inf
    best_impostor_scores = masked.max(axis=-1)
    margins = target_scores - best_impostor_scores
    overlap = overlap_coefficient(target_scores.ravel(), best_impostor_scores.ravel())

    result = OUT / "subjectwise_blocks01-08_train_blocks09-12_test_8class.npz"
    np.savez_compressed(
        result,
        frequencies_hz=frequencies,
        durations_seconds=durations,
        subject_accuracy=subject_accuracy,
        accuracy=accuracy,
        fft_subject_accuracy=fft_subject_accuracy,
        fft_accuracy=fft_accuracy,
        predictions=predictions,
        decision_scores=scores,
        target_scores=target_scores,
        best_impostor_scores=best_impostor_scores,
        margins=margins,
        overlap_coefficient=overlap,
        train_blocks_1based=train_blocks + 1,
        test_blocks_1based=test_blocks + 1,
        damping_alpha=parameters.damping_alpha,
        threshold=parameters.threshold,
        reset_mode=parameters.reset_mode,
        spike_detection=parameters.spike_detection,
    )

    plt.style.use("seaborn-v0_8-ticks")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(durations, 100 * accuracy, "o-", label="R&F personalized")
    ax.plot(durations, 100 * fft_accuracy, "s-", label="FFT + harmonic")
    ax.axhline(100 / n_classes, color="0.5", ls="--", label="Chance")
    ax.set(xlabel="Epoch duration (s)", ylabel="Mean held-out accuracy (%)", ylim=(0, 102))
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "01_accuracy_vs_duration.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for subject in range(n_subjects):
        ax.plot(durations, 100 * subject_accuracy[subject], marker="o", label=f"S{subject+1}")
    ax.axhline(100 / n_classes, color="0.5", ls="--")
    ax.set(xlabel="Epoch duration (s)", ylabel="Held-out R&F accuracy (%)", ylim=(0, 102))
    ax.legend(frameon=False, ncols=3); fig.tight_layout(); fig.savefig(FIG / "02_subject_accuracy.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    edges = np.histogram_bin_edges(np.r_[target_scores.ravel(), best_impostor_scores.ravel()], bins=24)
    ax.hist(target_scores.ravel(), bins=edges, density=True, alpha=0.45, label="True-class score")
    ax.hist(best_impostor_scores.ravel(), bins=edges, density=True, alpha=0.45, label="Best competing score")
    ax.set(xlabel="Calibrated template score at 5 s", ylabel="Density", title=f"Overlap coefficient = {overlap:.3f}")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "03_target_vs_impostor_overlap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for subject in range(n_subjects):
        rows.extend((f"S{subject+1}", float(value)) for value in margins[subject].ravel())
    values = [margins[subject].ravel() for subject in range(n_subjects)]
    ax.violinplot(values, showmedians=True, showextrema=False)
    ax.set_xticks(np.arange(1, n_subjects + 1), [f"S{x}" for x in range(1, n_subjects + 1)])
    ax.axhline(0, color="0.4", ls="--"); ax.set(ylabel="True score − best competing score", title="Positive margin means correct separation")
    fig.tight_layout(); fig.savefig(FIG / "04_margin_by_subject.png", dpi=180); plt.close(fig)

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for target, predicted in zip(np.broadcast_to(truth, (n_subjects,) + truth.shape).ravel(), predictions[-1].ravel()):
        confusion[target, predicted] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    normalized_confusion = confusion / confusion.sum(axis=1, keepdims=True)
    image = ax.imshow(normalized_confusion, cmap="viridis", vmin=0, vmax=1)
    for row in range(n_classes):
        for column in range(n_classes):
            ax.text(column, row, f"{normalized_confusion[row,column]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normalized_confusion[row,column] < .2 else "black")
    ax.set_xticks(np.arange(n_classes), frequencies); ax.set_yticks(np.arange(n_classes), frequencies)
    fig.colorbar(image, ax=ax, label="Fraction")
    ax.set(xlabel="Predicted frequency (Hz)", ylabel="True frequency (Hz)")
    fig.tight_layout(); fig.savefig(FIG / "05_normalized_confusion_matrix.png", dpi=180); plt.close(fig)

    mean_matrix = last_scores.mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(7, 6))
    limit = np.max(np.abs(mean_matrix))
    image = ax.imshow(mean_matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(n_classes), frequencies); ax.set_yticks(np.arange(n_classes), frequencies)
    fig.colorbar(image, ax=ax, label="Mean calibrated score")
    ax.set(xlabel="Candidate class (Hz)", ylabel="True class (Hz)")
    fig.tight_layout(); fig.savefig(FIG / "06_mean_score_overlap_matrix.png", dpi=180); plt.close(fig)

    print(f"Mean R&F accuracy at 5 s: {100*accuracy[-1]:.2f}%")
    print(f"Mean FFT accuracy at 5 s: {100*fft_accuracy[-1]:.2f}%")
    print(f"Target/impostor overlap coefficient: {overlap:.3f}")
    print(result)


if __name__ == "__main__":
    main()
