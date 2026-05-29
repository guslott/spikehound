"""
audio/duplex_monitor_prototype.py — EXPERIMENTAL, STANDALONE proof-of-concept.

Roadmap item 6.5 ("PortAudio / full-duplex same-device monitor") from
20260528_listen_latency_analysis.md, implemented WITHOUT a new dependency:
miniaudio (already required) ships a ``DuplexStream`` that joins a capture and a
playback device into a single low-latency callback.

This module is deliberately NOT imported anywhere in the application.  It is a
hardware-validation prototype: run it directly, listen to the monitored input
coming out of the speakers, and (optionally) measure the true round-trip with a
loopback as described in §10 of the analysis doc.  Promote it into a real DAQ
source only after it has been validated on real audio hardware.

Why this is separate from the shipping path
--------------------------------------------
A full-duplex monitor must OWN the sound-card capture (you cannot open the
card's capture twice), so integrating it would mean a new "duplex sound-card
source" that produces the pipeline's input AND the monitor output from one
stream.  That is an architectural fork that only helps the same-device case
(sound-card-in -> same-sound-card-out); it does nothing for the common
SpikerBox/amplifier-in -> laptop-speakers-out scenario, where the existing
MonitorAudioBridge path is the right tool.  Keeping this as an isolated
prototype avoids touching the verified shipping path with code that cannot be
exercised in CI.

Latency
-------
A duplex stream's structural latency floor is roughly ``2 x buffersize_msec``
(one capture period + one playback period) plus the hardware ADC/DAC, because a
sample cannot be played before it has been captured.  At ``buffersize_msec=5``
that is ~10 ms + DAC, versus ~20-28 ms for the separate-device bridge path.
The win is real but only for same-device monitoring.

Usage
-----
    python -m audio.duplex_monitor_prototype --seconds 5 --channel 0 --gain 0.3
    # or: python audio/duplex_monitor_prototype.py --seconds 5

Pass ``--buffer-ms 5`` to probe the low-latency floor (higher glitch risk).
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import numpy as np

try:
    import miniaudio
except ImportError as exc:  # pragma: no cover - prototype only
    miniaudio = None
    _IMPORT_ERR = exc

logger = logging.getLogger(__name__)


class DuplexMonitor:
    """Full-duplex monitor: routes one captured input channel to the output.

    The capture and playback run in a single miniaudio ``DuplexStream`` so the
    monitor incurs only the duplex buffer latency, not a separate capture
    device buffer + emitter batching + output ring + playback device buffer.

    Args:
        sample_rate: Stream sample rate (Hz). Capture and playback share it.
        capture_channels: Number of input channels delivered by the device.
        listen_channel: Index (within the captured channels) to monitor.
        gain: Output gain applied to the monitored channel (hard-clipped at ±1.5).
        playback_channels: Output channel count (mono replicated across them).
        buffer_msec: Per-direction device buffer. Smaller = lower latency, more XRUNs.
        capture_device_id / playback_device_id: miniaudio device ids, or None for
            the system defaults. Pass matching ids for true same-device duplex.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        capture_channels: int = 2,
        listen_channel: int = 0,
        gain: float = 0.3,
        playback_channels: int = 1,
        buffer_msec: int = 10,
        capture_device_id=None,
        playback_device_id=None,
    ) -> None:
        if miniaudio is None:  # pragma: no cover - prototype only
            raise RuntimeError(f"miniaudio is not available: {_IMPORT_ERR!r}")
        self.sample_rate = int(sample_rate)
        self.capture_channels = max(1, int(capture_channels))
        self.listen_channel = int(listen_channel)
        self.gain = float(gain)
        self.playback_channels = max(1, int(playback_channels))
        self.buffer_msec = max(2, int(buffer_msec))
        self._capture_device_id = capture_device_id
        self._playback_device_id = playback_device_id
        self._stream: Optional["miniaudio.DuplexStream"] = None

    def _monitor_generator(self):
        """Duplex callback: receive captured frames, yield playback frames.

        A duplex callback can only play what it has already captured, so there
        is an inherent one-period delay (the structural duplex latency floor).
        """
        n_cap = self.capture_channels
        n_play = self.playback_channels
        listen = self.listen_channel
        # Prime: play silence on the first callback, receive the first capture.
        captured = yield b""
        while True:
            if not captured:
                captured = yield b""
                continue
            arr = np.frombuffer(captured, dtype=np.float32)
            frames = arr.size // n_cap
            if frames <= 0 or listen >= n_cap:
                # Nothing usable — emit matching silence to keep the stream fed.
                silence = np.zeros(max(frames, 0) * n_play, dtype=np.float32)
                captured = yield silence.tobytes()
                continue
            block = arr.reshape(frames, n_cap)
            mono = block[:, listen] * self.gain
            np.clip(mono, -1.5, 1.5, out=mono)
            if n_play == 1:
                out = np.ascontiguousarray(mono, dtype=np.float32)
            else:
                out = np.repeat(mono[:, None], n_play, axis=1).astype(np.float32)
            captured = yield out.tobytes()

    def start(self) -> None:
        if self._stream is not None:
            return
        fmt = miniaudio.SampleFormat.FLOAT32
        self._stream = miniaudio.DuplexStream(
            playback_format=fmt,
            playback_channels=self.playback_channels,
            capture_format=fmt,
            capture_channels=self.capture_channels,
            sample_rate=self.sample_rate,
            buffersize_msec=self.buffer_msec,
            playback_device_id=self._playback_device_id,
            capture_device_id=self._capture_device_id,
        )
        gen = self._monitor_generator()
        next(gen)  # prime the generator
        self._stream.start(gen)
        logger.info(
            "DuplexMonitor started: %d Hz, %d->%d ch, listen=%d, buffer=%d ms "
            "(structural floor ~%d ms + DAC)",
            self.sample_rate, self.capture_channels, self.playback_channels,
            self.listen_channel, self.buffer_msec, 2 * self.buffer_msec,
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None


def _main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Standalone full-duplex monitor prototype (6.5).")
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to run.")
    parser.add_argument("--rate", type=int, default=44_100, help="Sample rate (Hz).")
    parser.add_argument("--capture-channels", type=int, default=2, help="Input channel count.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to monitor.")
    parser.add_argument("--gain", type=float, default=0.3, help="Output gain.")
    parser.add_argument("--buffer-ms", type=int, default=10, help="Per-direction device buffer (ms).")
    args = parser.parse_args()

    if miniaudio is None:  # pragma: no cover - prototype only
        print(f"miniaudio not available: {_IMPORT_ERR!r}")
        return 1

    monitor = DuplexMonitor(
        sample_rate=args.rate,
        capture_channels=args.capture_channels,
        listen_channel=args.channel,
        gain=args.gain,
        buffer_msec=args.buffer_ms,
    )
    print(
        f"Running duplex monitor for {args.seconds:.1f}s "
        f"(rate={args.rate}, listen ch={args.channel}, buffer={args.buffer_ms} ms). "
        f"You should hear the input on the output. Ctrl-C to stop."
    )
    monitor.start()
    try:
        time.sleep(max(0.0, args.seconds))
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
