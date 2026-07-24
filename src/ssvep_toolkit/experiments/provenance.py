"""Provenance records for reproducible study results."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class Provenance:
    schema_version: int
    git_commit: str | None
    git_dirty: bool | None
    python_version: str
    platform: str
    dependency_versions: dict[str, str]
    configuration_hash: str
    dataset_manifest_hash: str | None
    random_seed: int | None
    validation_level: str
    filter_mode: str | None
    causal_state_initialized: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configuration_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_manifest_hash(dataset_root: str | Path | None) -> str | None:
    if dataset_root is None:
        return None
    root = Path(dataset_root)
    if not root.exists():
        return None
    manifest = [(item.name, item.stat().st_size) for item in sorted(root.glob("data_s*_64.mat"))]
    return hashlib.sha256(json.dumps(manifest).encode("utf-8")).hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(config: dict[str, Any], *, validation_level: str,
                       filter_mode: str | None, dataset_root: str | Path | None = None) -> Provenance:
    if filter_mode == "causal" and config.get("preprocessing", {}).get("zero_phase") is True:
        raise ValueError("zero-phase filtering cannot be tagged causal")
    versions = {}
    for package in ("numpy", "scipy", "h5py", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    dirty = _git_value(["status", "--porcelain"])
    return Provenance(
        schema_version=1,
        git_commit=_git_value(["rev-parse", "HEAD"]), git_dirty=bool(dirty) if dirty is not None else None,
        python_version=sys.version.split()[0], platform=platform.platform(), dependency_versions=versions,
        configuration_hash=configuration_hash(config), dataset_manifest_hash=dataset_manifest_hash(dataset_root),
        random_seed=config.get("optimization", {}).get("seed"), validation_level=validation_level,
        filter_mode=filter_mode, causal_state_initialized=True if filter_mode == "causal" else None,
    )
