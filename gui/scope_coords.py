"""Coordinate mapping between channel volts and the scope's normalized Y axis.

Single source of truth for the affine transform shared by the threshold line,
the voltage axis ticks, and the scope pop-out. The scope renders every trace in
a fixed normalized ``[0, 1]`` viewport using::

    y_norm = volts / (2 * span) + offset
    volts  = (y_norm - offset) * (2 * span)

where ``span`` is the channel's half-range in volts (``vertical_span_v``) and
``offset`` is its screen position (``screen_offset``).

Both helpers accept scalars or NumPy arrays. ``TraceRenderer`` implements the
same forward transform in-place on pre-allocated buffers for the hot render
path; keep that math in sync with this module.
"""
from __future__ import annotations


def _safe_span(span: float) -> float:
    """Clamp the span away from zero so the transform stays invertible."""
    return max(float(span), 1e-9)


def volts_to_norm(volts, span, offset):
    """Map a channel voltage to its normalized [0, 1] screen position."""
    return volts / (2.0 * _safe_span(span)) + offset


def norm_to_volts(norm, span, offset):
    """Inverse of :func:`volts_to_norm`: screen position back to volts."""
    return (norm - offset) * (2.0 * _safe_span(span))
