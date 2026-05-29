"""Device-lifecycle regression tests for SoundCardSource (finding 6.6c).

A capture device created on the OS audio stream must be ``close()``d on every
teardown path — a failed ``start()`` and a ``stop()`` issued while the device is
not running — or the underlying stream leaks.
"""
from __future__ import annotations

import types

import daq.soundcard_source as scs
from daq.soundcard_source import SoundCardSource


class _FakeCaptureDevice:
    def __init__(self, *, start_raises: bool = False) -> None:
        self.running = False
        self.stopped = False
        self.closed = False
        self._start_raises = start_raises

    def start(self, gen) -> None:
        if self._start_raises:
            raise RuntimeError("simulated miniaudio start failure")
        self.running = True

    def stop(self) -> None:
        self.stopped = True
        self.running = False

    def close(self) -> None:
        self.closed = True


def _wire_miniaudio(monkeypatch, device):
    fake = types.SimpleNamespace(
        CaptureDevice=lambda **kwargs: device,
        SampleFormat=types.SimpleNamespace(FLOAT32=object()),
    )
    monkeypatch.setattr(scs, "miniaudio", fake)


def test_start_failure_closes_device(monkeypatch):
    device = _FakeCaptureDevice(start_raises=True)
    _wire_miniaudio(monkeypatch, device)

    source = SoundCardSource()
    source._n_in = 1
    source._miniaudio_device_id = object()
    source.config = types.SimpleNamespace(sample_rate=48_000, chunk_size=256)

    # _start_impl swallows the start error (existing contract) but must not leak
    # the device it already created.
    source._start_impl()

    assert device.closed, "device must be close()d when start() fails"
    assert source._device is None


def test_stop_closes_device_even_when_not_running(monkeypatch):
    device = _FakeCaptureDevice()
    device.running = False  # created but never reached the running state

    source = SoundCardSource()
    source._device = device

    source._stop_impl()

    assert device.closed, "device must be close()d on stop even when not running"
    assert source._device is None


def test_stop_stops_and_closes_running_device(monkeypatch):
    device = _FakeCaptureDevice()
    device.running = True  # normal running teardown still works

    source = SoundCardSource()
    source._device = device

    source._stop_impl()

    assert device.stopped, "a running device should be stop()ped"
    assert device.closed, "a running device should be close()d"
    assert source._device is None
