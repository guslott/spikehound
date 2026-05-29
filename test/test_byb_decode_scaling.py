"""BYB sample-decode scaling tests (finding 6.2).

Verifies that the 7+7 wire reconstruction and the bit-depth normalization use a
consistent width, so decoded amplitudes stay within [-1, 1) for every profile
bit depth — including when a sample carries stray high bits.
"""
from __future__ import annotations

import numpy as np
import pytest

from daq.backyard_brains import _BYBDecoder


def _pack_sample(value: int) -> bytes:
    """Pack a single sample as the standard right-aligned 7+7 per-sample frame."""
    high = ((value >> 7) & 0x7F) | 0x80  # flag set on the high byte
    low = value & 0x7F                    # flag clear on the low byte
    return bytes([high, low])


def _decode_one(bits: int, value: int) -> float:
    dec = _BYBDecoder(bits=bits, candidate_widths=[1])
    # Bypass stream-width inference: drive the per-sample decode path directly.
    dec._stream_width = 1
    dec._frame_mode = "per_sample"
    dec._raw.extend(_pack_sample(value))
    out = dec._decode_available_frames()
    assert out is not None and out.size == 1
    return float(np.asarray(out).reshape(-1)[0])


def test_decode_10bit_scaling_is_consistent() -> None:
    assert _decode_one(10, 0) == pytest.approx(-1.0)
    assert _decode_one(10, 512) == pytest.approx(0.0)
    assert _decode_one(10, 1023) == pytest.approx((1023 - 512) / 512)
    # Every in-range 10-bit value normalizes within [-1, 1).
    for v in range(0, 1024, 17):
        assert -1.0 <= _decode_one(10, v) < 1.0


def test_decode_14bit_scaling_is_consistent() -> None:
    assert _decode_one(14, 0) == pytest.approx(-1.0)
    assert _decode_one(14, 8192) == pytest.approx(0.0)
    assert _decode_one(14, 16383) == pytest.approx((16383 - 8192) / 8192)
    for v in range(0, 16384, 311):
        assert -1.0 <= _decode_one(14, v) < 1.0


def test_decode_masks_stray_high_bits_to_profile_width() -> None:
    """Finding 6.2: if a sample arrives with more significant bits than the
    profile's bit depth, the width-consistency mask keeps the normalized
    amplitude bounded instead of producing |amp| >> 1.

    A full 14-bit value (16383) on a 10-bit profile reconstructs to 16383; the
    mask (0x3FF) keeps only the low 10 bits (1023). Without the mask this would
    normalize to (16383 - 512) / 512 ≈ 31.0 — far outside [-1, 1).
    """
    out = _decode_one(10, 16383)
    assert out == pytest.approx((1023 - 512) / 512)
    assert -1.0 <= out < 1.0
