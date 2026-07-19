from pathlib import Path

import pytest

from ssvep_toolkit.config.models import (
    AppConfig,
    ConfigError,
    DatasetConfig,
    DownsamplingConfig,
    PreprocessingConfig,
)


def test_posterior_channel_conversion() -> None:
    config = AppConfig(dataset=DatasetConfig(root=Path(".")))
    assert config.channel_indices() == [47, 53, 54, 55, 56, 57, 60, 61, 62]


def test_invalid_frequency_is_rejected() -> None:
    config = AppConfig(dataset=DatasetConfig(root=Path("."), frequencies=[0]))
    with pytest.raises(ConfigError):
        config.validate()


def test_matlab_mode_requires_integer_factor() -> None:
    config = AppConfig(
        dataset=DatasetConfig(root=Path(".")),
        preprocessing=PreprocessingConfig(
            downsampling=DownsamplingConfig(original_rate_hz=1000, target_rate_hz=300)
        ),
    )
    with pytest.raises(ConfigError):
        config.validate()
