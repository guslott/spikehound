#!/usr/bin/env python3
"""
bench_conditioner.py — Timing benchmark for SignalConditioner.

Run directly (no pytest required):
    python test/perf/bench_conditioner.py

Reports mean ± std ± p95 in µs for each combination of filter spec,
channel count, and chunk size that matters for the monitor audio path
(64–256 frames) and the main acquisition pipeline (512–1024 frames).

Use this to catch performance regressions before/after conditioning changes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.conditioning import ChannelFilterSettings, FilterSettings, SignalConditioner
from shared.models import Chunk

SAMPLE_RATE = 44_100
N_REPS = 500
WARMUP = 20

SPECS = [
    (
        "bypass",
        FilterSettings(),
    ),
    (
        "notch 60 Hz",
        FilterSettings(
            default=ChannelFilterSettings(
                notch_enabled=True, notch_freq_hz=60.0, notch_q=30.0
            )
        ),
    ),
    (
        "AC + notch + LP-300 Hz order-4",
        FilterSettings(
            default=ChannelFilterSettings(
                ac_couple=True,
                ac_cutoff_hz=1.0,
                notch_enabled=True,
                notch_freq_hz=60.0,
                notch_q=30.0,
                lowpass_hz=300.0,
                lowpass_order=4,
            )
        ),
    ),
]


def _make_chunk(n_channels: int, n_frames: int, rng: np.random.Generator) -> Chunk:
    data = rng.standard_normal((n_channels, n_frames)).astype(np.float32)
    names = tuple(f"ch{i}" for i in range(n_channels))
    return Chunk(
        samples=data,
        start_time=0.0,
        dt=1.0 / SAMPLE_RATE,
        seq=0,
        channel_names=names,
        units="V",
    )


def _bench(label: str, cond: SignalConditioner, chunk: Chunk) -> None:
    for _ in range(WARMUP):
        cond.process(chunk)
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        cond.process(chunk)
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e6  # µs
    print(
        f"  {label:<55s}  "
        f"mean={arr.mean():7.1f} µs  "
        f"std={arr.std():5.1f} µs  "
        f"p95={np.percentile(arr, 95):7.1f} µs"
    )


def main() -> None:
    rng = np.random.default_rng(42)
    print("=" * 80)
    print("SignalConditioner — performance benchmark")
    print(f"  sample_rate={SAMPLE_RATE} Hz   reps={N_REPS}   warmup={WARMUP}")
    print("=" * 80)

    for n_ch in (1, 2, 8):
        for n_frames in (64, 128, 256, 512, 1024):
            print(f"\n  channels={n_ch}  frames={n_frames}")
            chunk = _make_chunk(n_ch, n_frames, rng)
            for spec_label, filt_settings in SPECS:
                cond = SignalConditioner(filt_settings)
                label = f"ch={n_ch} fr={n_frames} | {spec_label}"
                _bench(label, cond, chunk)


if __name__ == "__main__":
    main()
