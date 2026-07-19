from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration is missing or internally inconsistent."""


@dataclass
class DatasetConfig:
    root: Path
    subject_pattern: str = "data_s{subject}_64.mat"
    subjects: list[int] = field(default_factory=lambda: [1])
    conditions: list[int] = field(default_factory=lambda: [1, 2])
    frequencies: list[int] = field(default_factory=lambda: list(range(1, 61)))
    channels: str | list[int] = "posterior_9"


@dataclass
class DownsamplingConfig:
    enabled: bool = True
    original_rate_hz: int = 1000
    target_rate_hz: int = 250
    method: str = "matlab_compatible"


@dataclass
class LatencyConfig:
    enabled: bool = False
    seconds: float = 0.14


@dataclass
class PreprocessingConfig:
    downsampling: DownsamplingConfig = field(default_factory=DownsamplingConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)


@dataclass
class ExecutionConfig:
    overwrite: bool = False
    resume: bool = True


@dataclass
class OutputConfig:
    root: Path = Path("outputs")
    format: str = "hdf5"


@dataclass
class AppConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source_path: Path | None = None

    def validate(self) -> None:
        d = self.dataset
        if not d.subjects or any(x < 1 or x > 30 for x in d.subjects):
            raise ConfigError("dataset.subjects must contain values from 1 through 30")
        if not d.conditions or any(x not in (1, 2) for x in d.conditions):
            raise ConfigError("dataset.conditions must contain 1 and/or 2")
        if not d.frequencies or any(x < 1 or x > 60 for x in d.frequencies):
            raise ConfigError("dataset.frequencies must contain values from 1 through 60")
        if isinstance(d.channels, list) and (
            not d.channels or any(x < 1 or x > 64 for x in d.channels)
        ):
            raise ConfigError("explicit channel numbers must be from 1 through 64")
        if isinstance(d.channels, str) and d.channels not in {"all", "posterior_9"}:
            raise ConfigError("dataset.channels must be 'all', 'posterior_9', or a list")
        ds = self.preprocessing.downsampling
        if ds.original_rate_hz <= 0 or ds.target_rate_hz <= 0:
            raise ConfigError("sampling rates must be positive")
        if ds.method not in {"matlab_compatible", "polyphase"}:
            raise ConfigError("downsampling.method must be matlab_compatible or polyphase")
        if ds.method == "matlab_compatible" and ds.original_rate_hz % ds.target_rate_hz:
            raise ConfigError("matlab_compatible requires an integer downsampling factor")
        if self.preprocessing.latency.seconds < 0:
            raise ConfigError("latency.seconds cannot be negative")
        if self.output.format != "hdf5":
            raise ConfigError("only HDF5 output is currently implemented")

    def channel_indices(self) -> list[int]:
        value = self.dataset.channels
        if value == "all":
            return list(range(64))
        if value == "posterior_9":
            return [47, 53, 54, 55, 56, 57, 60, 61, 62]
        return [x - 1 for x in value]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        result["dataset"]["root"] = str(self.dataset.root)
        result["output"]["root"] = str(self.output.root)
        return result


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install the project dependencies") from exc

    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    dataset = _section(raw, "dataset")
    if "root" not in dataset:
        raise ConfigError("dataset.root is required")
    dataset_root = Path(dataset["root"])
    if not dataset_root.is_absolute():
        dataset_root = (source.parent / dataset_root).resolve()
    output = _section(raw, "output")
    output_root = Path(output.get("root", "outputs"))
    if not output_root.is_absolute():
        output_root = (source.parent.parent / output_root).resolve()
    pre = _section(raw, "preprocessing")
    cfg = AppConfig(
        dataset=DatasetConfig(
            root=dataset_root,
            subject_pattern=str(dataset.get("subject_pattern", "data_s{subject}_64.mat")),
            subjects=list(dataset.get("subjects", [1])),
            conditions=list(dataset.get("conditions", [1, 2])),
            frequencies=list(dataset.get("frequencies", range(1, 61))),
            channels=dataset.get("channels", "posterior_9"),
        ),
        preprocessing=PreprocessingConfig(
            downsampling=DownsamplingConfig(**_section(pre, "downsampling")),
            latency=LatencyConfig(**_section(pre, "latency")),
        ),
        execution=ExecutionConfig(**_section(raw, "execution")),
        output=OutputConfig(root=output_root, format=str(output.get("format", "hdf5"))),
        source_path=source,
    )
    cfg.validate()
    return cfg
