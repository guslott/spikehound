from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from shared.models import Chunk


@dataclass(frozen=True)
class ChannelFilterSettings:
    """Filter configuration applied to a single channel."""

    ac_couple: bool = False
    ac_cutoff_hz: float = 1.0
    notch_enabled: bool = False
    notch_freq_hz: float = 60.0
    notch_q: float = 30.0
    lowpass_hz: Optional[float] = None
    lowpass_order: int = 4
    highpass_hz: Optional[float] = None
    highpass_order: int = 2

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def any_enabled(self) -> bool:
        return (
            self.ac_couple
            or self.notch_enabled
            or self.lowpass_hz is not None
            or self.highpass_hz is not None
        )

    def validate(self, sample_rate: float) -> None:
        nyquist = sample_rate / 2.0
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.ac_couple and not (0 < self.ac_cutoff_hz < nyquist):
            raise ValueError("ac_cutoff_hz must be between 0 and Nyquist")
        if self.notch_enabled:
            if not (0 < self.notch_freq_hz < nyquist):
                raise ValueError("notch_freq_hz must be between 0 and Nyquist")
            if self.notch_q <= 0:
                raise ValueError("notch_q must be positive")
        if self.lowpass_hz is not None:
            if not (0 < self.lowpass_hz < nyquist):
                raise ValueError("lowpass_hz must be between 0 and Nyquist")
            if self.lowpass_order <= 0:
                raise ValueError("lowpass_order must be positive")
        if self.highpass_hz is not None:
            if not (0 < self.highpass_hz < nyquist):
                raise ValueError("highpass_hz must be between 0 and Nyquist")
            if self.highpass_order <= 0:
                raise ValueError("highpass_order must be positive")


@dataclass
class FilterSettings:
    """Per-channel filter configuration for the dispatcher."""

    default: ChannelFilterSettings = field(default_factory=ChannelFilterSettings)
    overrides: Dict[str, ChannelFilterSettings] = field(default_factory=dict)

    def for_channel(self, channel_name: str) -> ChannelFilterSettings:
        return self.overrides.get(channel_name, self.default)

    def any_enabled(self) -> bool:
        if self.default.any_enabled():
            return True
        return any(cfg.any_enabled() for cfg in self.overrides.values())

    def validate(self, sample_rate: float) -> None:
        self.default.validate(sample_rate)
        for cfg in self.overrides.values():
            cfg.validate(sample_rate)

    def as_dict(self) -> Dict[str, object]:
        return {
            "default": self.default.to_dict(),
            "overrides": {name: cfg.to_dict() for name, cfg in self.overrides.items()},
        }


class _BaseFilter:
    """Interface for stateful per-channel filters."""

    def apply(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self, n_channels: int) -> None:
        raise NotImplementedError

    def prime(self, initial_values: np.ndarray) -> None:
        """Set initial conditions based on the first sample of each channel."""
        # Default: nothing to do
        return


class _IIRFilter(_BaseFilter):
    def __init__(self, b: np.ndarray, a: np.ndarray, n_channels: int) -> None:
        self._b = np.asarray(b, dtype=np.float64)
        self._a = np.asarray(a, dtype=np.float64)
        self._n_states = max(len(self._a), len(self._b)) - 1
        self._zi = np.zeros((n_channels, self._n_states), dtype=np.float64)
        if self._n_states > 0:
            try:
                self._zi_template = signal.lfilter_zi(self._b, self._a)
            except Exception:  # pragma: no cover - defensive
                self._zi_template = np.zeros(self._n_states, dtype=np.float64)
        else:  # pragma: no cover - zero-order filter, uncommon
            self._zi_template = np.zeros(0, dtype=np.float64)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim != 2:
            raise ValueError("samples must be 2D (channels, frames)")
        if samples.shape[0] != self._zi.shape[0]:
            raise ValueError("channel count changed without reset")

        out = np.empty_like(samples, dtype=np.float32)
        for idx in range(samples.shape[0]):
            row = samples[idx].astype(np.float64, copy=False)
            filtered, zf = signal.lfilter(self._b, self._a, row, zi=self._zi[idx])
            self._zi[idx] = zf
            out[idx] = filtered.astype(np.float32, copy=False)
        return out

    def reset(self, n_channels: int) -> None:
        self._zi = np.zeros((n_channels, self._n_states), dtype=np.float64)

    def prime(self, initial_values: np.ndarray) -> None:
        if self._n_states == 0 or self._zi_template.size == 0:
            return
        if initial_values.shape[0] != self._zi.shape[0]:
            raise ValueError("initial_values length must match channel count")
        for idx, value in enumerate(initial_values):
            self._zi[idx] = self._zi_template * float(value)


class _ACCouplingFilter(_IIRFilter):
    def __init__(self, sample_rate: float, cutoff_hz: float, n_channels: int) -> None:
        norm = cutoff_hz / (sample_rate / 2.0)
        b, a = signal.butter(1, norm, btype="highpass")
        super().__init__(b, a, n_channels)


class _NotchFilter(_IIRFilter):
    def __init__(self, sample_rate: float, freq_hz: float, q: float, n_channels: int) -> None:
        norm = freq_hz / (sample_rate / 2.0)
        b, a = signal.iirnotch(norm, q)
        super().__init__(b, a, n_channels)


class _ButterworthFilter(_IIRFilter):
    def __init__(
        self,
        sample_rate: float,
        cutoff_hz: float,
        n_channels: int,
        *,
        order: int,
        btype: str,
    ) -> None:
        norm = cutoff_hz / (sample_rate / 2.0)
        b, a = signal.butter(order, norm, btype=btype)
        super().__init__(b, a, n_channels)


class SignalConditioner:
    """Applies configured IIR filters while preserving per-channel state."""

    def __init__(self, settings: Optional[FilterSettings] = None) -> None:
        self._settings = settings or FilterSettings()
        self._sample_rate: Optional[float] = None
        self._channel_names: Optional[Sequence[str]] = None
        self._channel_specs: Optional[Tuple[ChannelFilterSettings, ...]] = None
        self._channel_filters: List[List[_BaseFilter]] = []

    @property
    def settings(self) -> FilterSettings:
        return self._settings

    def update_settings(self, settings: FilterSettings) -> None:
        self._settings = settings
        # Force rebuild on next chunk
        self._sample_rate = None
        self._channel_filters = []
        self._channel_specs = None

    def describe(self) -> Dict[str, object]:
        return self._settings.as_dict()

    def _ensure_filters(self, chunk: Chunk) -> bool:
        sample_rate = 1.0 / chunk.dt
        channel_names = chunk.channel_names
        channel_specs = tuple(self._settings.for_channel(name) for name in channel_names)
        if (
            self._channel_filters
            and self._sample_rate == sample_rate
            and self._channel_names == channel_names
            and self._channel_specs == channel_specs
        ):
            return False

        self._settings.validate(sample_rate)
        self._sample_rate = sample_rate
        self._channel_names = channel_names
        self._channel_specs = channel_specs

        channel_filters: List[List[_BaseFilter]] = []
        for spec in channel_specs:
            chain: List[_BaseFilter] = []
            if spec.ac_couple:
                chain.append(_ACCouplingFilter(sample_rate, spec.ac_cutoff_hz, 1))
            if spec.notch_enabled:
                chain.append(_NotchFilter(sample_rate, spec.notch_freq_hz, spec.notch_q, 1))
            if spec.highpass_hz is not None:
                chain.append(
                    _ButterworthFilter(
                        sample_rate,
                        spec.highpass_hz,
                        1,
                        order=spec.highpass_order,
                        btype="highpass",
                    )
                )
            if spec.lowpass_hz is not None:
                chain.append(
                    _ButterworthFilter(
                        sample_rate,
                        spec.lowpass_hz,
                        1,
                        order=spec.lowpass_order,
                        btype="lowpass",
                    )
                )
            channel_filters.append(chain)

        self._channel_filters = channel_filters
        return True

    def process(self, chunk: Chunk) -> np.ndarray:
        samples = np.array(chunk.samples, dtype=np.float32, copy=True)
        if not self._settings.any_enabled():
            return samples

        rebuilt = self._ensure_filters(chunk)
        if not self._channel_filters:
            return samples

        if rebuilt and samples.size:
            initial = samples[:, 0]
            for idx, chain in enumerate(self._channel_filters):
                if not chain:
                    continue
                priming = np.asarray([initial[idx]], dtype=np.float32)
                for filt in chain:
                    filt.prime(priming)

        filtered = samples.copy()
        for idx, chain in enumerate(self._channel_filters):
            if not chain:
                continue
            row = filtered[idx : idx + 1]
            for filt in chain:
                row = filt.apply(row)
            filtered[idx, :] = row[0]
        return filtered

    def reset(self) -> None:
        if self._channel_filters:
            for chain in self._channel_filters:
                for filt in chain:
                    filt.reset(1)
