"""SQLite experiment registry shared by CLI, GUI, dashboard, and studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    def fallback(item: Any) -> Any:
        if hasattr(item, "tolist"):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=fallback)


def configuration_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    id: int
    study_id: int
    run_key: str
    status: str
    subject_id: int | None
    class_count: int | None
    outer_fold: int | None
    encoder: str | None
    config_hash: str
    started_utc: str | None
    finished_utc: str | None
    error: str | None


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    run_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    subject_id INTEGER,
    class_count INTEGER,
    outer_fold INTEGER,
    encoder TEXT,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    started_utc TEXT,
    finished_utc TEXT,
    error TEXT,
    UNIQUE(study_id, run_key)
);
CREATE TABLE IF NOT EXISTS parameters (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY(run_id, name)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step REAL,
    split TEXT NOT NULL DEFAULT '',
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_run_name ON metrics(run_id, name);
CREATE INDEX IF NOT EXISTS runs_study_status ON runs(study_id, status);
"""


class ExperimentRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> "ExperimentRegistry":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
        return self

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def create_study(self, name: str, config: Mapping[str, Any], description: str = "") -> int:
        payload = canonical_json(config)
        digest = configuration_hash(config)
        with self.connect() as connection:
            existing = connection.execute("SELECT id,config_hash FROM studies WHERE name=?", (name,)).fetchone()
            if existing is not None:
                if existing["config_hash"] != digest:
                    raise ValueError(f"study {name!r} already exists with a different configuration")
                connection.execute("UPDATE studies SET description=? WHERE id=?", (description, existing["id"]))
                return int(existing["id"])
            cursor = connection.execute(
                "INSERT INTO studies(name, description, config_json, config_hash, created_utc) VALUES(?,?,?,?,?)",
                (name, description, payload, digest, utc_now()),
            )
            return int(cursor.lastrowid)

    def create_run(
        self,
        study_id: int,
        run_key: str,
        config: Mapping[str, Any],
        *,
        subject_id: int | None = None,
        class_count: int | None = None,
        outer_fold: int | None = None,
        encoder: str | None = None,
        status: str = "queued",
    ) -> int:
        payload = canonical_json(config)
        digest = configuration_hash(config)
        started = utc_now() if status == "running" else None
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id,config_hash FROM runs WHERE study_id=? AND run_key=?", (study_id, run_key)
            ).fetchone()
            if existing is not None:
                if existing["config_hash"] != digest:
                    raise ValueError(f"run {run_key!r} already exists with a different configuration")
                connection.execute("UPDATE runs SET status=? WHERE id=?", (status, existing["id"]))
                return int(existing["id"])
            cursor = connection.execute(
                "INSERT INTO runs(study_id,run_key,status,subject_id,class_count,outer_fold,encoder,config_json,config_hash,started_utc) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (study_id, run_key, status, subject_id, class_count, outer_fold, encoder, payload, digest, started),
            )
            return int(cursor.lastrowid)

    def set_status(self, run_id: int, status: str, error: str | None = None) -> None:
        finished = utc_now() if status in {"complete", "failed", "cancelled", "invalid"} else None
        started = utc_now() if status == "running" else None
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?, error=?, finished_utc=COALESCE(?,finished_utc), "
                "started_utc=COALESCE(started_utc,?) WHERE id=?",
                (status, error, finished, started, run_id),
            )

    def log_parameters(self, run_id: int, parameters: Mapping[str, Any]) -> None:
        rows = [(run_id, name, canonical_json(value)) for name, value in parameters.items()]
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO parameters(run_id,name,value_json) VALUES(?,?,?) "
                "ON CONFLICT(run_id,name) DO UPDATE SET value_json=excluded.value_json",
                rows,
            )

    def log_metric(self, run_id: int, name: str, value: float, *, step: float | None = None, split: str = "") -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO metrics(run_id,name,value,step,split,created_utc) VALUES(?,?,?,?,?,?)",
                (run_id, name, float(value), step, split, utc_now()),
            )

    def replace_metric(self, run_id: int, name: str, value: float, *,
                       step: float | None = None, split: str = "") -> None:
        """Idempotently replace a scalar checkpoint metric."""
        with self.connect() as connection:
            if step is None:
                connection.execute(
                    "DELETE FROM metrics WHERE run_id=? AND name=? AND split=? AND step IS NULL",
                    (run_id, name, split),
                )
            else:
                connection.execute(
                    "DELETE FROM metrics WHERE run_id=? AND name=? AND split=? AND step=?",
                    (run_id, name, split, step),
                )
            connection.execute(
                "INSERT INTO metrics(run_id,name,value,step,split,created_utc) VALUES(?,?,?,?,?,?)",
                (run_id, name, float(value), step, split, utc_now()),
            )

    def add_artifact(self, run_id: int, kind: str, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO artifacts(run_id,kind,path,metadata_json,created_utc) VALUES(?,?,?,?,?)",
                (run_id, kind, str(path), canonical_json(metadata or {}), utc_now()),
            )

    def event(self, message: str, *, run_id: int | None = None, level: str = "info") -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO events(run_id,level,message,created_utc) VALUES(?,?,?,?)",
                (run_id, level, message, utc_now()),
            )

    def list_runs(self, study_id: int | None = None) -> list[RunRecord]:
        query = "SELECT id,study_id,run_key,status,subject_id,class_count,outer_fold,encoder,config_hash,started_utc,finished_utc,error FROM runs"
        parameters: tuple[Any, ...] = ()
        if study_id is not None:
            query += " WHERE study_id=?"
            parameters = (study_id,)
        query += " ORDER BY id"
        with self.connect() as connection:
            return [RunRecord(**dict(row)) for row in connection.execute(query, parameters)]

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            studies = connection.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
            run_rows = connection.execute("SELECT status,COUNT(*) AS count FROM runs GROUP BY status").fetchall()
            metrics = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        return {"studies": int(studies), "runs_by_status": {row[0]: int(row[1]) for row in run_rows}, "metrics": int(metrics)}

    def dashboard_snapshot(self) -> dict[str, Any]:
        """Return JSON-ready run, metric, and artifact data for read-only UIs."""
        with self.connect() as connection:
            studies = [dict(row) for row in connection.execute(
                "SELECT id,name,description,status,created_utc,config_hash FROM studies ORDER BY id DESC"
            )]
            runs = [dict(row) for row in connection.execute(
                "SELECT id,study_id,run_key,status,subject_id,class_count,outer_fold,encoder,"
                "started_utc,finished_utc,error FROM runs ORDER BY id DESC"
            )]
            metrics = [dict(row) for row in connection.execute(
                "SELECT m.run_id,m.name,m.value,m.step,m.split,m.created_utc "
                "FROM metrics m ORDER BY m.id"
            )]
            artifacts = [dict(row) for row in connection.execute(
                "SELECT run_id,kind,path,metadata_json,created_utc FROM artifacts ORDER BY id DESC"
            )]
        return {"summary": self.summary(), "studies": studies, "runs": runs,
                "metrics": metrics, "artifacts": artifacts}
