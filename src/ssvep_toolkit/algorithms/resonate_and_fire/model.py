from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResonateAndFireParameters:
    """Parameters shared by the neurons in an oscillator bank."""

    damping_alpha: float = 0.3
    threshold: float = 0.01
    input_gain: float = 0.8
    transient_seconds: float = 0.1
    refractory_seconds: float = 0.0
    refractory_cycles: float = 0.0
    reset_mode: str = "notebook_compatible"
    integration_substeps: int = 4
    normalized_dynamics: bool = True
    normalize_input_by_resonance: bool = True
    solver: str = "exact"
    spike_detection: str = "level"

    def validate(self) -> None:
        if self.damping_alpha <= 0 or self.threshold <= 0 or self.input_gain <= 0:
            raise ValueError("damping, threshold, and input gain must be positive")
        if self.transient_seconds < 0 or self.refractory_seconds < 0 or self.refractory_cycles < 0:
            raise ValueError("time parameters cannot be negative")
        if self.reset_mode not in {"notebook_compatible", "zero"}:
            raise ValueError("reset_mode must be notebook_compatible or zero")
        if not isinstance(self.integration_substeps, int) or self.integration_substeps < 1:
            raise ValueError("integration_substeps must be a positive integer")
        if self.solver not in {"exact", "euler"}:
            raise ValueError("solver must be exact or euler")
        if self.spike_detection not in {"level", "upward_crossing"}:
            raise ValueError("spike_detection must be level or upward_crossing")
