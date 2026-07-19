import os
import time

from ssvep_toolkit.storage import old_gui_run_candidates, prune_old_gui_runs, remove_stale_partial_files


def test_gui_run_retention_keeps_newest_and_removes_only_old_candidates(tmp_path):
    root = tmp_path / "gui_runs"; root.mkdir()
    runs = []
    for index in range(5):
        path = root / f"run_{index}"; path.mkdir(); runs.append(path)
        old = time.time() - (40 + index) * 86400
        os.utime(path, (old, old))
    candidates = old_gui_run_candidates(root, keep=2, older_than_days=30)
    assert len(candidates) == 3
    removed = prune_old_gui_runs(root, keep=2, older_than_days=30)
    assert removed == candidates
    assert len(tuple(root.iterdir())) == 2


def test_stale_partial_cleanup_preserves_recent_file(tmp_path):
    old = tmp_path / "old.partial.npz"; old.write_bytes(b"old")
    recent = tmp_path / "recent.partial.npz"; recent.write_bytes(b"new")
    timestamp = time.time() - 48 * 3600; os.utime(old, (timestamp, timestamp))
    assert remove_stale_partial_files(tmp_path, older_than_hours=24) == (old.resolve(),)
    assert recent.exists() and not old.exists()
