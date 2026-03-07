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
        if sample_rate <= 0 or not np.isfinite(sample_rate):
            raise ValueError("sample_rate must be positive and finite")
        nyquist = sample_rate / 2.0
        if self.ac_couple and not (0 < self.ac_cutoff_hz < nyquist and np.isfinite(self.ac_cutoff_hz)):
            raise ValueError("ac_cutoff_hz must be between 0 and Nyquist (finite)")
        if self.notch_enabled:
            if not (0 < self.notch_freq_hz < nyquist and np.isfinite(self.notch_freq_hz)):
                raise ValueError("notch_freq_hz must be between 0 and Nyquist (finite)")
            if self.notch_q <= 0 or not np.isfinite(self.notch_q):
                raise ValueError("notch_q must be positive and finite")
        if self.lowpass_hz is not None:
            if not (0 < self.lowpass_hz < nyquist and np.isfinite(self.lowpass_hz)):
                raise ValueError("lowpass_hz must be between 0 and Nyquist (finite)")
            if self.lowpass_order <= 0:
                raise ValueError("lowpass_order must be positive")
        if self.highpass_hz is not None:
            if not (0 < self.highpass_hz < nyquist and np.isfinite(self.highpass_hz)):
                raise ValueError("highpass_hz must be between 0 and Nyquist (finite)")
            if self.highpass_order <= 0:
                raise ValueError("highpass_order must be positive")

    def describe(self) -> str:
        parts = []
        if self.ac_couple:
            parts.append(f"AC({self.ac_cutoff_hz}Hz)")
        if self.notch_enabled:
            parts.append(f"Notch({self.notch_freq_hz}Hz)")
        if self.highpass_hz is not None:
            parts.append(f"HP({self.highpass_hz}Hz)")
        if self.lowpass_hz is not None:
            parts.append(f"LP({self.lowpass_hz}Hz)")
        return ", ".join(parts) if parts else "None"


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
        return  # Default: nothing to do


class _IIRFilter(_BaseFilter):
    """SOS-based IIR filter that processes ALL channels simultaneously.

    Using Second-Order Sections (SOS) rather than direct-form b/a coefficients
    gives better numerical stability for higher-order filters.

    scipy.signal.sosfilt processes the full (n_channels, n_frames) array in
    a single C-level loop — eliminating the Python-level per-channel iteration
    and the float32→float64→float32 cast that happened once per channel.

    zi layout expected by sosfilt with axis=1:
        (n_channels, n_sections, 2)
    """

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        self._sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        self._n_sections = self._sos.shape[0]
        self._n_channels = n_channels
        # scipy sosfilt with axis=1 on input (n_channels, n_frames) requires
        # zi.shape == (n_sections, n_channels, 2)  — NOT (n_channels, n_sections, 2).
        # The rule is: zi = (*batch_dims_of_x_excluding_axis, n_sections, 2)
        # → batch_dim = x.shape[0] = n_channels → (n_sections, n_channels, 2).
        self._zi = np.zeros((self._n_sections, n_channels, 2), dtype=np.float64)
        # Template for priming: shape (n_sections, 2) — one IC per section
        self._zi_template = signal.sosfilt_zi(self._sos)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Apply filter to all channels at once.

        Args:
            samples: float32 array of shape (n_channels, n_frames)
        Returns:
            Filtered float32 array of same shape.
        """
        if samples.ndim != 2 or samples.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected ({self._n_channels}, n_frames), got {samples.shape}"
            )
        # sosfilt operates in float64 for precision; cast the whole array once
        # rather than once per channel in a Python loop.
        out, self._zi = signal.sosfilt(
            self._sos,
            samples.astype(np.float64, copy=False),
            axis=1,
            zi=self._zi,
        )
        return out.astype(np.float32, copy=False)

    def reset(self, n_channels: int) -> None:
        self._n_channels = n_channels
        self._zi = np.zeros((self._n_sections, n_channels, 2), dtype=np.float64)

    def prime(self, initial_values: np.ndarray) -> None:
        """Seed filter state so the first output matches the DC level of the signal.

        Args:
            initial_values: 1-D array of shape (n_channels,) — the first sample
                            of each channel used to compute step-response ICs.
        """
        n = min(self._n_channels, len(initial_values))
        for i in range(n):
            # zi[:, i, :] is the section×delay slice for channel i
            self._zi[:, i, :] = self._zi_template * float(initial_values[i])


class _ACCouplingFilter(_IIRFilter):
    def __init__(self, sample_rate: float, cutoff_hz: float, n_channels: int) -> None:
        norm = cutoff_hz / (sample_rate / 2.0)
        sos = signal.butter(1, norm, btype="highpass", output="sos")
        super().__init__(sos, n_channels)


class _NotchFilter(_IIRFilter):
    def __init__(self, sample_rate: float, freq_hz: float, q: float, n_channels: int) -> None:
        norm = freq_hz / (sample_rate / 2.0)
        b, a = signal.iirnotch(norm, q)
        sos = signal.tf2sos(b, a)
        super().__init__(sos, n_channels)


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
        sos = signal.butter(order, norm, btype=btype, output="sos")
        super().__init__(sos, n_channels)


class SignalConditioner:
    """Applies configured IIR filters while preserving per-channel state.

    Filter strategy
    ---------------
    *Uniform mode* (all channels share the same FilterSettings):
        A single set of _IIRFilter objects is created with n_channels=N.
        ``sosfilt`` processes the full (N, frames) array in one C-level call —
        no Python loop, no per-channel cast overhead.

    *Non-uniform mode* (per-channel overrides active):
        Falls back to one n_channels=1 filter per channel.  Still uses sosfilt
        rather than lfilter so the zi shape is consistent.
    """

    def __init__(self, settings: Optional[FilterSettings] = None) -> None:
        self._settings = settings or FilterSettings()
        self._sample_rate: Optional[float] = None
        self._channel_names: Optional[Sequence[str]] = None
        self._channel_specs: Optional[Tuple[ChannelFilterSettings, ...]] = None
        # Non-uniform path: one chain per channel
        self._channel_filters: List[List[_BaseFilter]] = []
        # Uniform path: single chain shared by all channels
        self._uniform_chain: List[_BaseFilter] = []
        self._uniform_mode: bool = False

    @property
    def settings(self) -> FilterSettings:
        return self._settings

    def update_settings(self, settings: FilterSettings) -> None:
        self._settings = settings
        # Force rebuild on next chunk
        self._sample_rate = None
        self._channel_filters = []
        self._uniform_chain = []
        self._uniform_mode = False
        self._channel_specs = None

    def describe(self) -> Dict[str, str]:
        if not self._channel_names or not self._channel_specs:
            return {}
        return {name: spec.describe() for name, spec in zip(self._channel_names, self._channel_specs)}

    def _ensure_filters(self, chunk: Chunk) -> bool:
        sample_rate = 1.0 / chunk.dt
        channel_names = chunk.channel_names
        channel_specs = tuple(self._settings.for_channel(name) for name in channel_names)

        # Check if anything has changed; if not, reuse existing filter state
        already_built = bool(self._channel_filters) or bool(self._uniform_chain)
        if (
            already_built
            and self._sample_rate == sample_rate
            and self._channel_names == channel_names
            and self._channel_specs == channel_specs
        ):
            return False

        self._settings.validate(sample_rate)
        self._sample_rate = sample_rate
        self._channel_names = channel_names
        self._channel_specs = channel_specs

        n_ch = len(channel_specs)

        def _build_chain(spec: ChannelFilterSettings, n_channels: int) -> List[_BaseFilter]:
            chain: List[_BaseFilter] = []
            if spec.ac_couple:
                chain.append(_ACCouplingFilter(sample_rate, spec.ac_cutoff_hz, n_channels))
            if spec.notch_enabled:
                chain.append(_NotchFilter(sample_rate, spec.notch_freq_hz, spec.notch_q, n_channels))
            if spec.highpass_hz is not None:
                chain.append(
                    _ButterworthFilter(
                        sample_rate, spec.highpass_hz, n_channels,
                        order=spec.highpass_order, btype="highpass",
                    )
                )
            if spec.lowpass_hz is not None:
                chain.append(
                    _ButterworthFilter(
                        sample_rate, spec.lowpass_hz, n_channels,
                        order=spec.lowpass_order, btype="lowpass",
                    )
                )
            return chain

        # Uniform mode: all channels share the same spec (the overwhelmingly common case).
        # Build ONE multi-channel filter chain so sosfilt processes all channels in one call.
        if n_ch > 0 and len(set(channel_specs)) == 1:
            self._uniform_chain = _build_chain(channel_specs[0], n_ch)
            self._channel_filters = []
            self._uniform_mode = True
        else:
            # Non-uniform: per-channel specs — one n_channels=1 chain per channel
            self._channel_filters = [_build_chain(spec, 1) for spec in channel_specs]
            self._uniform_chain = []
            self._uniform_mode = False

        return True

    def process(self, chunk: Chunk) -> np.ndarray:
        # np.ascontiguousarray only allocates when the source is non-contiguous or
        # the dtype differs — avoids the unconditional copy=True that existed before.
        samples = np.ascontiguousarray(chunk.samples, dtype=np.float32)

        if not self._settings.any_enabled():
            return samples

        rebuilt = self._ensure_filters(chunk)
        has_filters = bool(self._uniform_chain) or bool(self._channel_filters)
        if not has_filters:
            return samples

        # Prime filter initial conditions on the first chunk after a rebuild so
        # the filter starts at the DC level of the signal (avoids transient artifacts).
        if rebuilt and samples.size:
            initial = samples[:, 0]  # (n_channels,)
            if self._uniform_mode:
                for filt in self._uniform_chain:
                    filt.prime(initial)
            else:
                for idx, chain in enumerate(self._channel_filters):
                    if chain and idx < len(initial):
                        priming = initial[idx : idx + 1]
                        for filt in chain:
                            filt.prime(priming)

        if self._uniform_mode:
            # All channels share the same filter spec — process the full
            # (n_channels, n_frames) array in a single sosfilt call (no Python loop).
            row: np.ndarray = samples
            for filt in self._uniform_chain:
                row = filt.apply(row)
            return row
        else:
            # Per-channel filters — build output without a full-array pre-copy.
            filtered = np.empty_like(samples)
            for idx, chain in enumerate(self._channel_filters):
                if not chain:
                    filtered[idx] = samples[idx]
                    continue
                row = samples[idx : idx + 1]
                for filt in chain:
                    row = filt.apply(row)
                filtered[idx] = row[0]
            return filtered

    def reset(self) -> None:
        if self._uniform_mode:
            for filt in self._uniform_chain:
                filt.reset(filt._n_channels)
        else:
            for chain in self._channel_filters:
                for filt in chain:
                    filt.reset(1)
