"""Auditable stimulus-frequency selection for class-scaling experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HarmonicCollision:
    source_hz: int
    harmonic: int
    target_hz: int


@dataclass(frozen=True)
class ClassSetDesign:
    frequencies_hz: tuple[int, ...]
    class_count: int
    spacing_hz: int
    start_hz: int
    span_hz: int
    harmonic_collisions: tuple[HarmonicCollision, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frequencies_hz": list(self.frequencies_hz), "class_count": self.class_count,
            "spacing_hz": self.spacing_hz, "start_hz": self.start_hz, "span_hz": self.span_hz,
            "harmonic_collisions": [collision.__dict__ for collision in self.harmonic_collisions],
        }


def factorial_class_sets(
    class_counts: Sequence[int], spacings_hz: Sequence[int], *,
    available_hz: Sequence[int] = (8, 60), starts_hz: Sequence[int] | None = None,
    interference_harmonics: Sequence[int] = (2, 3), maximum_collisions: int | None = 0,
) -> tuple[ClassSetDesign, ...]:
    """Generate an explicit class-count × spacing × start factorial design.

    Invalid combinations are omitted rather than silently changing spacing or
    range. The returned metadata makes class count, spacing, span, and harmonic
    collisions separable experimental factors.
    """
    if len(available_hz) != 2: raise ValueError("available_hz must contain [minimum, maximum]")
    low, high = map(int, available_hz); designs = []
    for count in sorted({int(value) for value in class_counts}):
        if count < 2: raise ValueError("class counts must be at least two")
        for spacing in sorted({int(value) for value in spacings_hz}):
            if spacing < 1: raise ValueError("spacings must be positive")
            span = spacing * (count - 1)
            starts = tuple(int(value) for value in starts_hz) if starts_hz is not None else tuple(range(low, high - span + 1))
            for start in starts:
                frequencies = tuple(start + spacing * index for index in range(count))
                if frequencies[0] < low or frequencies[-1] > high: continue
                collisions = harmonic_collisions(frequencies, interference_harmonics)
                if maximum_collisions is not None and len(collisions) > int(maximum_collisions): continue
                designs.append(ClassSetDesign(frequencies, count, spacing, start, span, collisions))
    return tuple(designs)


def harmonic_collisions(
    frequencies_hz: Sequence[int],
    interference_harmonics: Sequence[int] = (2, 3),
) -> tuple[HarmonicCollision, ...]:
    """Return exact harmonic relations between distinct stimulus classes."""
    selected = {int(value) for value in frequencies_hz}
    collisions = []
    for source in sorted(selected):
        for harmonic in sorted({int(value) for value in interference_harmonics}):
            if harmonic <= 1:
                raise ValueError("interference harmonics must be greater than one")
            target = source * harmonic
            if target in selected:
                collisions.append(HarmonicCollision(source, harmonic, target))
    return tuple(collisions)


def select_class_frequencies(
    class_count: int,
    *,
    available_hz: Sequence[int] = (8, 60),
    strategy: str = "compact_harmonic_aware",
    interference_harmonics: Sequence[int] = (2, 3),
    spacing_hz: int | None = None,
    start_hz: int | None = None,
) -> tuple[int, ...]:
    """Select integer stimulus classes without silently introducing spacing bias.

    ``compact_harmonic_aware`` evaluates every consecutive window and chooses
    the one with the fewest exact harmonic class collisions. Ties prefer the
    lower-frequency window. ``low_contiguous`` is useful as a controlled
    ablation, while ``legacy_spread`` reproduces the earlier 8--39-style design.
    """
    if len(available_hz) != 2:
        raise ValueError("available_hz must contain [minimum, maximum]")
    low, high = (int(value) for value in available_hz)
    if class_count < 2 or low < 1 or high < low or class_count > high - low + 1:
        raise ValueError("class_count does not fit the available integer frequencies")

    if strategy == "fixed_spacing_harmonic_aware":
        if spacing_hz is None or int(spacing_hz) < 1:
            raise ValueError("fixed_spacing_harmonic_aware requires positive spacing_hz")
        spacing = int(spacing_hz)
        span = spacing * (class_count - 1)
        if low + span > high:
            raise ValueError("class_count and spacing do not fit the available frequencies")
        if start_hz is not None:
            start = int(start_hz)
            values = tuple(start + spacing * index for index in range(class_count))
            if values[0] < low or values[-1] > high:
                raise ValueError("fixed start, class_count and spacing exceed available frequencies")
            return values
        candidates = (
            tuple(start + spacing * index for index in range(class_count))
            for start in range(low, high - span + 1)
        )
        return min(
            candidates,
            key=lambda values: (len(harmonic_collisions(values, interference_harmonics)), values[0]),
        )
    if strategy == "low_contiguous":
        return tuple(range(low, low + class_count))
    if strategy == "legacy_spread":
        import numpy as np

        values = np.rint(np.linspace(low, high, class_count)).astype(int)
        if len(set(values.tolist())) != class_count:
            raise ValueError("legacy spacing produced duplicate frequencies")
        return tuple(int(value) for value in values)
    if strategy != "compact_harmonic_aware":
        raise ValueError(f"unknown class-selection strategy: {strategy}")

    candidates = (
        tuple(range(start, start + class_count))
        for start in range(low, high - class_count + 2)
    )
    return min(
        candidates,
        key=lambda values: (len(harmonic_collisions(values, interference_harmonics)), values[0]),
    )
