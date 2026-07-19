import json
import numpy as np

from ssvep_toolkit.preprocessing import butterworth_bandpass_stream, butterworth_sos
from ssvep_toolkit.progress import ProgressJournal, WorkEstimate, latest_progress


def test_streaming_bandpass_matches_one_shot_causal_filter() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(2, 3, 700))
    sos = butterworth_sos(1000, 11, 13, order=5)
    whole, _ = butterworth_bandpass_stream(values, sos)
    first, state = butterworth_bandpass_stream(values[..., :140], sos)
    second, _ = butterworth_bandpass_stream(values[..., 140:], sos, state)
    assert np.allclose(np.concatenate((first, second), axis=-1), whole, atol=1e-12)


def test_progress_journal_records_plan_eta_and_provenance(tmp_path) -> None:
    path = tmp_path / "progress.jsonl"
    journal = ProgressJournal(
        path, config={"study": "test"}, study="transparent-test",
        estimate=WorkEstimate(1, 2, 2, 4, 3, 2, 3),
    )
    journal.write("features", 1, 4, "first candidate")
    final = journal.write("features", 4, 4, "done", status="complete")
    assert final["fraction"] == 1
    assert latest_progress(path)["status"] == "complete"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["extra"]["work_estimate"]["cells"] == 4
    assert len(first["config_sha256"]) == 64
