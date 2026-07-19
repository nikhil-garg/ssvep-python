"""Conservative retention helpers for generated, reproducible GUI artifacts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil


def old_gui_run_candidates(root: str | Path, *, keep: int = 3, older_than_days: int = 30) -> tuple[Path, ...]:
    directory = Path(root).resolve()
    if keep < 0 or older_than_days < 0 or not directory.exists():
        return ()
    runs = sorted((item for item in directory.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    return tuple(
        item for item in runs[keep:]
        if datetime.fromtimestamp(item.stat().st_mtime, timezone.utc) < cutoff
    )


def prune_old_gui_runs(root: str | Path, *, keep: int = 3, older_than_days: int = 30) -> tuple[Path, ...]:
    directory = Path(root).resolve()
    removed = old_gui_run_candidates(directory, keep=keep, older_than_days=older_than_days)
    for path in removed:
        resolved = path.resolve()
        if not resolved.is_relative_to(directory) or resolved == directory:
            raise ValueError(f"refusing to remove path outside GUI run root: {resolved}")
        shutil.rmtree(resolved)
    return removed


def remove_stale_partial_files(root: str | Path, *, older_than_hours: int = 24) -> tuple[Path, ...]:
    directory = Path(root).resolve()
    if older_than_hours < 0 or not directory.exists():
        return ()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
    removed = []
    for path in directory.rglob("*.partial.npz"):
        resolved = path.resolve()
        if not resolved.is_relative_to(directory):
            continue
        modified = datetime.fromtimestamp(resolved.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            resolved.unlink(); removed.append(resolved)
    return tuple(removed)
