"""Structured, append-only progress and provenance records for long studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any


@dataclass(frozen=True)
class WorkEstimate:
    subjects: int
    class_tasks: int
    filter_modes: int
    rf_candidates: int
    delta_candidates: int
    lif_candidates: int
    ridge_candidates: int
    outer_folds: int = 12

    @property
    def cells(self) -> int:
        return self.subjects * self.class_tasks * self.filter_modes

    def to_dict(self) -> dict[str, int]:
        result = asdict(self); result["cells"] = self.cells
        result["candidate_feature_sets_per_task"] = self.rf_candidates + self.delta_candidates + self.lif_candidates
        return result


class ProgressJournal:
    """Write human-readable console progress and machine-readable JSONL events."""
    def __init__(self, path: str | Path, *, config: dict[str, Any], study: str,
                 estimate: WorkEstimate | None = None) -> None:
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic(); self.phase_started = self.started
        self.last_phase = ""; self.config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.write("study_started", 0, None, "Study initialized", extra={
            "study": study, "config_sha256": self.config_hash,
            "python": platform.python_version(), "platform": platform.platform(),
            "pid": os.getpid(), "work_estimate": estimate.to_dict() if estimate else None,
        })

    def write(self, phase: str, current: int | None, total: int | None, message: str,
              *, status: str = "running", extra: dict[str, Any] | None = None) -> dict[str, Any]:
        now = time.monotonic()
        if phase != self.last_phase:
            self.phase_started = now; self.last_phase = phase
        phase_elapsed = now - self.phase_started
        fraction = current / total if current is not None and total else None
        eta = phase_elapsed * (1 - fraction) / fraction if fraction and fraction > 0 else None
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "phase": phase, "current": current, "total": total, "fraction": fraction,
            "message": message, "phase_elapsed_seconds": round(phase_elapsed, 3),
            "study_elapsed_seconds": round(now - self.started, 3),
            "eta_seconds": round(eta, 3) if eta is not None else None,
            "config_sha256": self.config_hash,
        }
        if extra: event["extra"] = extra
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
        return event


def latest_progress(path: str | Path) -> dict[str, Any] | None:
    """Read only the last valid event from a possibly interrupted JSONL file."""
    source = Path(path)
    if not source.exists(): return None
    last = None
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        try: last = json.loads(line)
        except json.JSONDecodeError: continue
    return last
