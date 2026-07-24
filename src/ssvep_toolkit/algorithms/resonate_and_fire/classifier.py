from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .model import ResonateAndFireParameters
from .simulator import simulate_bank


@dataclass
class OscillatorBankClassifier:
    frequencies_hz: tuple[float, ...]
    sampling_rate_hz: float
    parameters: ResonateAndFireParameters
    channel_scale_: Any | None = None
    spread_hz: tuple[float, ...] = (0.0,)
    harmonics: tuple[int, ...] = (1,)
    harmonic_weights: tuple[float, ...] = (1.0,)
    score_center_: Any | None = None
    score_scale_: Any | None = None
    score_direction_: Any | None = None
    class_templates_: Any | None = None

    def __init__(self, frequencies_hz: Sequence[float], sampling_rate_hz: float, parameters: ResonateAndFireParameters,
                 *, spread_hz: Sequence[float] = (0.0,), harmonics: Sequence[int] = (1,),
                 harmonic_weights: Sequence[float] | None = None):
        self.frequencies_hz = tuple(float(x) for x in frequencies_hz)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.parameters = parameters
        self.spread_hz = tuple(float(x) for x in spread_hz)
        self.harmonics = tuple(int(x) for x in harmonics)
        if not self.spread_hz or not self.harmonics or any(x < 1 for x in self.harmonics):
            raise ValueError("spread_hz and positive harmonics cannot be empty")
        if harmonic_weights is None:
            harmonic_weights = tuple(1.0/h for h in self.harmonics)
        self.harmonic_weights = tuple(float(x) for x in harmonic_weights)
        if len(self.harmonic_weights) != len(self.harmonics):
            raise ValueError("harmonic_weights must match harmonics")
        self.channel_scale_ = None
        self.score_center_ = None; self.score_scale_ = None; self.score_direction_ = None; self.class_templates_ = None

    @property
    def neuron_frequencies_hz(self) -> tuple[float, ...]:
        return tuple(h*f + offset for f in self.frequencies_hz for h in self.harmonics for offset in self.spread_hz)

    def fit_scaler(self, signals: Any) -> "OscillatorBankClassifier":
        import numpy as np

        values = np.asarray(signals, dtype=float)
        scale = np.std(values, axis=(0, 2), ddof=1)
        self.channel_scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, signals: Any) -> Any:
        import numpy as np

        if self.channel_scale_ is None:
            raise RuntimeError("fit_scaler must be called first")
        values = np.asarray(signals, dtype=float)
        centered = values - values.mean(axis=-1, keepdims=True)
        return centered / self.channel_scale_[None, :, None]

    def scores(self, signals: Any, duration_samples: Sequence[int]) -> Any:
        import numpy as np
        grouped = self.neuron_scores(signals, duration_samples).mean(axis=-1)
        weights = np.asarray(self.harmonic_weights, dtype=float)
        return np.sum(grouped * weights, axis=-1) / weights.sum()

    def neuron_scores(self, signals: Any, duration_samples: Sequence[int]) -> Any:
        """Return unpooled counts as ``duration, trial, target, harmonic, neuron``.

        Keeping the spread-neuron axis lets downstream experiments compare a
        single resonator with a local oscillator bank without discarding the
        response pattern through premature averaging.
        """
        spikes = simulate_bank(
            self.transform(signals), self.neuron_frequencies_hz,
            self.sampling_rate_hz, self.parameters, duration_samples,
        )
        shape = spikes.shape[:-1] + (len(self.frequencies_hz), len(self.harmonics), len(self.spread_hz))
        return spikes.reshape(shape)

    def decision_scores(self, signals: Any, duration_samples: Sequence[int]) -> Any:
        """Return spike rates calibrated against training-fold non-target responses."""
        import numpy as np
        stops = np.asarray(duration_samples, dtype=float)
        rates = self.scores(signals, duration_samples) / (stops[:, None, None] / self.sampling_rate_hz)
        if self.score_center_ is None:
            return rates
        center=self.score_center_; scale=self.score_scale_; templates=self.class_templates_
        if center.ndim == 1:
            center=center[None,:]; scale=scale[None,:]; templates=templates[None,:,:]
        if center.shape[0] == 1 and rates.shape[0] != 1:
            center=np.broadcast_to(center,(rates.shape[0],center.shape[1])); scale=np.broadcast_to(scale,center.shape)
            templates=np.broadcast_to(templates,(rates.shape[0],)+templates.shape[1:])
        if center.shape[0] != rates.shape[0]:
            raise ValueError("calibration durations do not match requested durations")
        features = (rates - center[:, None, :]) / scale[:, None, :]
        if self.class_templates_ is not None:
            return -np.mean((features[:, :, None, :] - templates[:, None, :, :])**2, axis=-1)
        return self.score_direction_[None, None, :] * features

    def fit_calibration(self, signals: Any, labels: Any, duration_samples: Any) -> "OscillatorBankClassifier":
        """Fit class-specific null response distributions using training data only."""
        import numpy as np
        labels = np.asarray(labels, dtype=int)
        stops=np.atleast_1d(np.asarray(duration_samples,dtype=int))
        rates = self.scores(signals, stops) / (stops[:,None,None]/self.sampling_rate_hz)
        # All oscillator responses jointly form the feature vector.  Class
        # templates retain useful harmonic-overlap patterns that argmax spike
        # counts discards (e.g. 10 Hz's second harmonic overlaps 20 Hz).
        self.score_center_ = rates.mean(axis=1)
        self.score_scale_ = np.maximum(rates.std(axis=1,ddof=1),1e-6)
        standardized=(rates-self.score_center_[:,None,:])/self.score_scale_[:,None,:]
        self.class_templates_=np.stack([standardized[:,labels==index].mean(axis=1) for index in range(len(self.frequencies_hz))],axis=1)
        centers=[]; scales=[]; directions=[]
        for class_index in range(len(self.frequencies_hz)):
            audit_rates=rates[0]; null = audit_rates[labels != class_index, class_index]
            if null.size < 2: null = audit_rates[:, class_index]
            target = audit_rates[labels == class_index, class_index]
            null_mean=float(np.mean(null)); target_mean=float(np.mean(target))
            centers.append((null_mean+target_mean)/2); scales.append(float(np.std(null,ddof=1)))
            # Some reset regimes encode resonance as suppressed rather than
            # increased firing; learn that orientation without test leakage.
            directions.append(1.0 if target_mean >= null_mean else -1.0)
        # Retain orientation metadata for auditability, though template
        # distances are used whenever calibration has all classes.
        self.score_direction_ = np.asarray(directions)
        return self

    def predict(self, signals: Any, duration_samples: Sequence[int]) -> Any:
        import numpy as np

        return np.argmax(self.decision_scores(signals, duration_samples), axis=-1)
