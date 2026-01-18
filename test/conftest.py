from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the spikehound package is importable
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


# -----------------------------------------------------------------------------
# Pytest markers
# -----------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast, deterministic unit tests")
    config.addinivalue_line("markers", "property: Hypothesis property-based tests")
    config.addinivalue_line("markers", "integration: Multi-component tests")
    config.addinivalue_line("markers", "slow: Long-running tests (>30s)")
    config.addinivalue_line("markers", "hil: Hardware-in-the-loop tests")


# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_rate():
    """Standard sample rate for tests (Hz)."""
    return 20_000.0


@pytest.fixture
def sample_rate_low():
    """Low sample rate for filter tests (Hz)."""
    return 1_000.0


@pytest.fixture
def dt(sample_rate):
    """Time step corresponding to sample_rate."""
    return 1.0 / sample_rate


# Re-export fixtures from fixtures subpackage for convenient access
from test.fixtures.signal_generators import (
    make_sine,
    make_spike_train,
    make_triphasic_spike,
    make_dc_with_drift,
    make_mains_hum,
    add_gaussian_noise,
)
from test.fixtures.controlled_device import ControlledDevice
from test.fixtures.reference_models import ReferenceRingBuffer
