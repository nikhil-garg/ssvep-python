from pathlib import Path

from ssvep_toolkit.runners import compute_figure10_cohort, compute_figure12_cohort


ROOT = Path(__file__).resolve().parents[1]
inputs = sorted((ROOT / "outputs/reference_study/preprocessed").glob("subject_*_preprocessed.h5"))
parameters = ROOT.parent / "dataset_code/dataset_code/temporary data"

compute_figure10_cohort(
    inputs,
    parameters,
    ROOT / "outputs/reference_study/results/figure_10_fbcca.npz",
    checkpoint_dir=ROOT / "outputs/reference_study/checkpoints/figure_10_fbcca",
)
compute_figure12_cohort(
    inputs,
    parameters,
    ROOT / "outputs/reference_study/results/figure_12_fbtrca.npz",
    checkpoint_dir=ROOT / "outputs/reference_study/checkpoints/figure_12_fbtrca",
)
