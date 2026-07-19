"""Import resumable NPZ checkpoints into the experiment registry."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import ExperimentRegistry


def import_npz_checkpoints(registry: ExperimentRegistry, root: str | Path, *,
                           study_name: str | None = None) -> dict[str, int]:
    import numpy as np

    source = Path(root).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    study = registry.create_study(
        study_name or source.name, {"source": str(source), "format": "legacy_npz_checkpoint"},
        description="Imported checkpoint tree; evaluation design is retained per run.",
    )
    imported = skipped = failed = 0
    for path in sorted(source.rglob("*.npz")):
        excluded = {"cache", "figures", "invalid_filter_after_crop_checkpoints"}
        if path.name.endswith(".partial.npz") or excluded.intersection(part.lower() for part in path.parts):
            skipped += 1
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                scalar = {key: _scalar(archive[key]) for key in archive.files if archive[key].ndim == 0}
            subject = _integer(scalar.get("subject_id"))
            classes = _integer(scalar.get("class_count"))
            encoder = scalar.get("encoder")
            run = registry.create_run(
                study, path.relative_to(source).as_posix(), scalar,
                subject_id=subject, class_count=classes,
                encoder=str(encoder) if encoder is not None else _encoder_from_name(path.name),
                status="complete",
            )
            parameters = {key: value for key, value in scalar.items()
                          if key.startswith("selected_") or key in {"threshold", "input_gain", "damping_alpha"}}
            if parameters:
                registry.log_parameters(run, parameters)
            for key in ("accuracy", "fused_accuracy", "elapsed_seconds"):
                value = scalar.get(key)
                if isinstance(value, (int, float)):
                    registry.replace_metric(run, key, float(value), split=_split_label(scalar))
            registry.add_artifact(run, "checkpoint", path,
                                  {"evaluation_design": scalar.get("evaluation_design", "unknown")})
            imported += 1
        except (OSError, ValueError, KeyError, TypeError):
            failed += 1
    return {"study_id": study, "imported": imported, "skipped": skipped, "failed": failed}


def _scalar(value: Any) -> Any:
    item = value.item()
    return item.decode() if isinstance(item, bytes) else item


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _encoder_from_name(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith("delta_"):
        return "delta"
    if lower.startswith("lif_"):
        return "lif"
    if "resonate" in lower or lower.startswith("subject_"):
        return "resonate_fire"
    return None


def _split_label(scalar: dict[str, Any]) -> str:
    design = str(scalar.get("evaluation_design", "")).lower()
    if "outer" in design or "held" in design or "test" in design:
        return "reported_test"
    if "same" in design or "apparent" in design:
        return "apparent_same_data"
    return "reported"
