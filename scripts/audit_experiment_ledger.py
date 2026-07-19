"""Create a machine-readable inventory of experiment outputs and runtimes."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs/experiments"
OUTPUT = ROOT / "outputs/audits/project_experiment_ledger.json"
SELECTED_KEYS = (
    "selected_alpha", "selected_threshold", "selected_operating_rms",
    "selected_bipolar_alpha", "selected_bipolar_threshold",
    "selected_bipolar_operating_rms", "selected_threshold_scale",
    "selected_threshold_uV", "selected_asymmetry", "selected_tau_seconds",
    "accuracy", "fused_accuracy", "optimized_bipolar_accuracy",
)


def scalar(value: np.ndarray) -> float | str | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    item = array.item()
    if isinstance(item, (bytes, str)):
        return str(item)
    try:
        return float(item)
    except (TypeError, ValueError):
        return str(item)


def summarize_experiment(path: Path) -> dict[str, object]:
    all_files = list(path.rglob("*.npz"))
    files = [file for file in all_files if not any(part.startswith("invalid_") for part in file.relative_to(path).parts)]
    elapsed = []
    designs: Counter[str] = Counter()
    selected: dict[str, list[float]] = defaultdict(list)
    unreadable = []
    for file in files:
        try:
            with np.load(file, allow_pickle=False) as result:
                if "elapsed_seconds" in result:
                    value = scalar(result["elapsed_seconds"])
                    if isinstance(value, float) and np.isfinite(value):
                        elapsed.append(value)
                if "evaluation_design" in result:
                    value = scalar(result["evaluation_design"])
                    if value is not None:
                        designs[str(value)] += 1
                for key in SELECTED_KEYS:
                    if key in result:
                        value = scalar(result[key])
                        if isinstance(value, float) and np.isfinite(value):
                            selected[key].append(value)
        except Exception as exc:  # audit should identify rather than hide damaged outputs
            unreadable.append({"file": str(file.relative_to(ROOT)), "error": str(exc)})
    runtime = None
    if elapsed:
        runtime = {
            "files_with_runtime": len(elapsed),
            "sum_seconds": float(np.sum(elapsed)),
            "median_seconds": float(np.median(elapsed)),
            "minimum_seconds": float(np.min(elapsed)),
            "maximum_seconds": float(np.max(elapsed)),
        }
    parameter_summary = {
        key: {
            "count": len(values), "minimum": float(np.min(values)),
            "median": float(np.median(values)), "maximum": float(np.max(values)),
            "unique": sorted(set(round(value, 10) for value in values))[:50],
        }
        for key, values in selected.items()
    }
    return {
        "npz_files": len(files),
        "excluded_invalid_npz_files": len(all_files) - len(files),
        "figures": len(list(path.rglob("*.png"))),
        "bytes": sum(file.stat().st_size for file in path.rglob("*") if file.is_file()),
        "runtime": runtime,
        "evaluation_designs": dict(designs),
        "selected_values": parameter_summary,
        "unreadable": unreadable,
    }


def main() -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(EXPERIMENTS),
        "experiments": {
            path.name: summarize_experiment(path)
            for path in sorted(EXPERIMENTS.iterdir()) if path.is_dir()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)
    for name, experiment in payload["experiments"].items():
        runtime = experiment["runtime"]
        hours = runtime["sum_seconds"] / 3600 if runtime else 0
        print(f"{name}: npz={experiment['npz_files']} figures={experiment['figures']} recorded_cpu_cells={hours:.2f}h")


if __name__ == "__main__":
    main()
