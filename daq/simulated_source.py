# daq/simulated_source.py
"""
Simulated Physiology Source with Physiologically Accurate Neural Units.

This module generates realistic multi-unit neural recordings with:
- Triphasic extracellular spike waveforms
- Multiple unit types (sensory, motor, interneuron)
- Refractory period enforcement
- Conduction velocity simulation
- Post-synaptic potential (PSP) generation
"""
import logging
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from .base_device import BaseDevice, Chunk, DeviceInfo, ChannelInfo, Capabilities, ActualConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Type Presets - Invertebrate neurophysiology parameters
# =============================================================================
# These parameters are calibrated for invertebrate preparations (cricket, cockroach,
# earthworm, crayfish, etc.) which have slower, broader action potentials due to
# unmyelinated or partially myelinated axons.

UNIT_TYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    'sensory_afferent': {
        'description': 'Sensory afferents (e.g., cricket cercal wind sensors, mechanoreceptors)',
        'rate_hz': (2.0, 6.0),            # Baseline firing rate range (Hz)
        'spike_width_ms': (1.5, 2.5),     # Broader spikes typical of invertebrates
        'amplitude_v': (0.15, 0.40),      # Moderate amplitude
        'velocity_m_s': (3.0, 8.0),       # Slow unmyelinated fibers
        'refractory_ms': (2.0, 3.5),      # Longer refractory period
        'distal_ratio': (0.6, 0.9),       # Some attenuation
        'psp_gain': (0.015, 0.030),       # Moderate synaptic strength
        'syn_delay_ms': (2.0, 4.0),       # Synaptic delay
    },
    'giant_fiber': {
        'description': 'Giant fiber neurons (e.g., cockroach giant interneurons, squid giant axon)',
        'rate_hz': (1.0, 3.0),            # Lower baseline rate
        'spike_width_ms': (2.0, 4.0),     # Very broad spikes (giant axons)
        'amplitude_v': (0.6, 1.5),        # Large amplitude (large diameter)
        'velocity_m_s': (8.0, 15.0),      # Faster due to large diameter, but still invertebrate
        'refractory_ms': (2.5, 4.0),      # Longer refractory
        'distal_ratio': (0.7, 1.0),       # Good propagation
        'psp_gain': (0.030, 0.050),       # Strong synaptic drive
        'syn_delay_ms': (1.5, 3.0),       # Faster synaptic response
    },
    'motor_neuron': {
        'description': 'Motor neurons (e.g., leg motor neurons, body wall innervation)',
        'rate_hz': (3.0, 10.0),           # Variable rate
        'spike_width_ms': (1.8, 3.0),     # Broad spikes
        'amplitude_v': (0.25, 0.60),      # Medium-large amplitude
        'velocity_m_s': (2.0, 6.0),       # Slow, often unmyelinated
        'refractory_ms': (2.0, 3.0),      # Standard invertebrate refractory
        'distal_ratio': (0.5, 0.8),       # Some attenuation
        'psp_gain': (0.020, 0.040),       # Moderate-strong synapses
        'syn_delay_ms': (2.5, 5.0),       # Variable delay
    },
}

# Default unit distribution for 6-unit simulation
DEFAULT_UNIT_DISTRIBUTION = [
    'sensory_afferent',   # Unit 0 - wind/touch sensor
    'sensory_afferent',   # Unit 1 - wind/touch sensor
    'giant_fiber',        # Unit 2 - escape response giant fiber
    'giant_fiber',        # Unit 3 - escape response giant fiber
    'motor_neuron',       # Unit 4 - leg/body motor
    'motor_neuron',       # Unit 5 - leg/body motor
]


def generate_triphasic_template(spike_len: int, width_factor: float = 1.0) -> np.ndarray:
    """
    Generate a realistic triphasic extracellular action potential template.
    
    Extracellular recordings show a characteristic triphasic waveform due to
    the spatial derivative of the intracellular action potential as it
    propagates past the electrode:
    
    1. Initial negative deflection (approaching depolarization)
    2. Positive peak (passing depolarization wavefront)
    3. After-hyperpolarization dip (repolarization/undershoot)
    
    Parameters
    ----------
    spike_len : int
        Number of samples for the spike template.
    width_factor : float
        Scaling factor for spike width (1.0 = normal, >1 = broader).
        
    Returns
    -------
    np.ndarray
        Normalized triphasic spike template with zero endpoints.
    """
    # Time axis spans slightly asymmetric range for realistic shape
    t = np.linspace(-1.2, 2.0, spike_len)
    
    # Scale time by width factor (broader spikes = slower kinetics)
    t_scaled = t / max(0.5, width_factor)
    
    # Primary negative phase: Gaussian derivative-like shape
    # This represents the initial negative deflection as depolarization approaches
    primary_neg = -np.exp(-t_scaled**2 / 0.15)
    
    # Positive phase: Delayed positive overshoot
    # This is the main "spike" as the action potential passes under the electrode
    positive_phase = 1.2 * np.exp(-(t_scaled - 0.4)**2 / 0.12)
    
    # After-hyperpolarization: Smaller negative tail
    # Represents the undershoot/repolarization phase
    ahp = -0.35 * np.exp(-(t_scaled - 1.0)**2 / 0.25)
    
    # Combine phases
    template = primary_neg + positive_phase + ahp
    
    # Remove DC offset
    template -= np.mean(template)
    
    # Smooth taper to zero at edges using half-cosine window
    taper_len = max(2, spike_len // 8)
    taper_in = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
    taper_out = 0.5 * (1 + np.cos(np.pi * np.arange(taper_len) / taper_len))
    template[:taper_len] *= taper_in
    template[-taper_len:] *= taper_out
    
    # Force exact zero at endpoints before DC correction
    # (will be re-forced after normalization)
    template[0] = 0.0
    template[-1] = 0.0
    
    # DC correction: subtract mean of interior samples to preserve zero endpoints
    if len(template) > 2:
        interior_mean = np.mean(template[1:-1])
        template[1:-1] -= interior_mean
    
    # Normalize to unit peak
    peak = np.max(np.abs(template))
    if peak > 1e-12:
        template /= peak
    
    # Final guarantee: force exact zero at endpoints after all processing
    template[0] = 0.0
    template[-1] = 0.0
    
    return template.astype(np.float32)


@dataclass
class ActivePSP:
    """Represents an active PSP that spans multiple chunks."""
    start_sample: int       # The absolute sample index (global counter) where this PSP begins.
    template: np.ndarray    # The pre-calculated waveform array for this PSP (alpha function).
    gain: float             # Scaling factor for the PSP amplitude (simulating synaptic strength).
    unit_index: int         # ID of the presynaptic unit that triggered this event (for debugging).

class SimulatedPhysiologySource(BaseDevice):
    """
    Simulated Physiology Source
    ===========================

    Theory of Operations
    --------------------
    This source simulates a multi-unit recording from a nerve bundle. It generates synthetic
    neural data across three distinct channels, representing different recording modalities:

    1.  **Extracellular Proximal (Ch 0)**:
        -   Represents a recording electrode placed close to the nerve bundle.
        -   Captures large-amplitude spikes from multiple units.
        -   Signal = Sum of unit spikes + Broadband Noise + Line Hum.

    2.  **Extracellular Distal (Ch 1)**:
        -   Represents a second electrode placed further down the nerve (e.g., 20mm away).
        -   Captures the same spikes as the proximal channel but with a conduction delay.
        -   The delay is determined by the conduction velocity of each unit.
        -   Signal = Sum of delayed unit spikes + Broadband Noise.

    3.  **Intracellular (Ch 2)**:
        -   Represents a patch-clamp or intracellular recording from a post-synaptic muscle fiber.
        -   Displays Post-Synaptic Potentials (PSPs) triggered by the spikes in the nerve bundle.
        -   Each unit has a specific synaptic delay and PSP gain.
        -   Signal = Sum of PSPs (alpha functions) + Baseline (-70mV) + Low-level Noise.

    Simulation Logic
    ----------------
    -   **Initialization**: Randomly generates N units, assigning each a unique spike template,
        firing rate, amplitude, conduction velocity, and synaptic properties.
    -   **Loop**:
        1.  **Shift Buffers**: Moves the rolling wave buffers for extracellular signals.
        2.  **Generate Events**: Probabilistically triggers spikes for each unit based on its rate.
            -   Spikes are added to the extracellular wave buffers.
            -   PSPs are instantiated as `ActivePSP` objects with a start time and template.
        3.  **Render Channels**:
            -   Extracellular channels sum the wave buffers (with appropriate delays for Distal).
            -   Intracellular channel sums the contributions of all `ActivePSP` objects that overlap
                with the current data chunk.
        4.  **Cleanup**: Removes completed PSPs and updates the global sample counter.
    """

    @classmethod
    def device_class_name(cls) -> str:
        return "Simulated"

    # Fixed seed for repeatable simulations - change this to get different but still repeatable patterns
    RANDOM_SEED = 42
    
    def __init__(self, queue_maxsize: int = 4) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._noise_level = 0.040  # Increased: ~10mV broadband noise
        self._distance_m = 0.02  # Prox→Dist electrode spacing for delay calc
        self._units = []
        self._worker: threading.Thread | None = None
        # Event-based PSP tracking (replaces rolling buffers)
        self._global_sample_counter = 0  # Tracks absolute sample position
        self._active_psps: list[ActivePSP] = []  # Currently active PSPs
        # Wave buffers still used for extracellular signals (simpler, don't cross chunks as much)
        self._unit_wave_buffers = []
        self._buf_len = 0
        self._psp_template = np.zeros(1)
        self._psp_len = 1
        self._buffer_margin = 0
        self._default_line_hum_amp = 0.05  # Increased: ~5mV 60Hz line hum
        self._line_hum_amp = self._default_line_hum_amp
        self._line_hum_freq = 60.0
        self._line_hum_phase = 0.0
        self._line_hum_omega = 0.0
        self._chunk_buffer: np.ndarray | None = None
        # Seeded RNG for repeatable spike timing and noise patterns
        self._rng = np.random.default_rng(self.RANDOM_SEED)

    # ---- Discovery ------------------------------------------------------------
    @classmethod
    def list_available_devices(cls) -> list[DeviceInfo]:
        return [DeviceInfo(id="sim0", name="Simulated Physiology (virtual)")]

    def get_capabilities(self, device_id: str) -> Capabilities:
        return Capabilities(max_channels_in=3, sample_rates=[20_000], dtype="float32")

    def list_available_channels(self, device_id: str):
        return [
            ChannelInfo(id=0, name='Extracellular Proximal', units='V'),
            ChannelInfo(id=1, name='Extracellular Distal', units='V'),
            ChannelInfo(id=2, name='Intracellular', units='V'),
        ]

    def _initialize_units(self, sample_rate: int, num_units: int) -> None:
        """
        Initialize simulated neural units with FIXED, REPEATABLE parameters.

        This method creates a deterministic population of 6 neural units designed for:
        - Reliable, repeatable testing of spike sorting and conduction velocity measurements
        - Student laboratory exercises with known ground-truth answers
        - Easy separation of units based on amplitude, spike width, and timing
        - **Filter compatibility**: All spike widths ≥2.0ms to survive 1kHz low-pass filtering
        
        GROUND TRUTH VALUES (at 20kHz sample rate, 20mm electrode spacing):
        =====================================================================
        Unit | Type           | Prox Amp | Post-1kHz | Width  | Velocity | Cond Delay | Rate
        -----|----------------|----------|-----------|--------|----------|------------|------
          0  | Small Sensory  | 0.12 V   | ~0.10 V   | 2.0 ms |  4.0 m/s |   5.0 ms   | 3 Hz
          1  | Med Sensory    | 0.22 V   | ~0.21 V   | 2.2 ms |  6.0 m/s |   3.33 ms  | 4 Hz
          2  | Giant Fiber A  | 0.45 V   | ~0.46 V   | 2.8 ms | 12.0 m/s |   1.67 ms  | 2 Hz
          3  | Giant Fiber B  | 0.70 V   | ~0.72 V   | 3.2 ms | 10.0 m/s |   2.0 ms   | 1.5 Hz
          4  | Motor A        | 0.32 V   | ~0.31 V   | 2.4 ms |  5.0 m/s |   4.0 ms   | 5 Hz
          5  | Motor B        | 0.55 V   | ~0.55 V   | 2.6 ms |  3.0 m/s |   6.67 ms  | 3.5 Hz
        
        Post-1kHz amplitude gaps: ~100mV between adjacent clusters (well separated)
        Conduction Delay = electrode_spacing (20mm) / velocity
        
        Noise level is 40mV (0.040 V), so all units are above 2.5x noise floor,
        making them detectable with MAD-based threshold detection even after filtering.
        """
        # PSP kernel (alpha function) - 20ms duration
        psp_len = int(0.020 * sample_rate)
        t_psp = np.linspace(0, 5, psp_len)
        self._psp_template = t_psp * np.exp(-t_psp)
        self._psp_template /= np.max(self._psp_template)
        
        # Subtract the endpoint value to shift the baseline so it ends at zero
        endpoint_value = self._psp_template[-1]
        self._psp_template -= endpoint_value
        
        # Trim the beginning where values are <= 0 (below the shifted baseline)
        positive_indices = np.where(self._psp_template > 0)[0]
        if positive_indices.size > 0:
            start_idx = positive_indices[0]
            self._psp_template = self._psp_template[start_idx:]
        
        # Ensure exact zero at start and end
        self._psp_template[0] = 0.0
        self._psp_template[-1] = 0.0
        
        self._psp_template = self._psp_template.astype(np.float32, copy=False)
        self._psp_len = len(self._psp_template)

        # =============================================================================
        # HARDCODED UNIT DEFINITIONS - Designed for repeatability and separability
        # =============================================================================
        # Each unit has distinct characteristics to ensure easy classification:
        # - Amplitudes span from just above noise floor (~2.5x) to large (~17.5x)
        # - Spike widths vary from narrow (1.2ms) to broad (3.0ms)
        # - Conduction velocities produce clearly different delays (1.67ms to 6.67ms)
        # - Different distal/proximal ratios simulate electrode placement variation
        #
        # All values are deterministic - no random number generation.
        # =============================================================================
        
        HARDCODED_UNITS = [
            {
                # Unit 0: Small Sensory Afferent - smallest amplitude, slow conduction
                # FILTER-COMPATIBLE: 2.0ms width survives 1kHz LP with ~15% loss
                'unit_type': 'small_sensory',
                'rate_hz': 3.0,
                'spike_width_ms': 2.0,        # Widened from 1.2ms for filter compatibility
                'amp_prox': 0.12,             # 3x noise floor, ~0.10V after filtering
                'amp_dist_ratio': 0.80,       # 80% of proximal → 0.10V distal
                'velocity': 4.0,              # 4 m/s → 5.0ms conduction delay
                'refractory_ms': 2.5,
                'syn_delay_ms': 3.0,
                'psp_gain': 0.015,
            },
            {
                # Unit 1: Medium Sensory Afferent - moderate size
                # FILTER-COMPATIBLE: 2.2ms width survives 1kHz LP with ~7% loss
                'unit_type': 'medium_sensory',
                'rate_hz': 4.0,
                'spike_width_ms': 2.2,        # Widened from 1.5ms for filter compatibility
                'amp_prox': 0.22,             # 5.5x noise floor, ~0.21V after filtering
                'amp_dist_ratio': 0.82,       # 82% of proximal → 0.18V distal
                'velocity': 6.0,              # 6 m/s → 3.33ms conduction delay
                'refractory_ms': 2.8,
                'syn_delay_ms': 3.5,
                'psp_gain': 0.020,
            },
            {
                # Unit 2: Giant Fiber A - large amplitude, broad spike, fast conduction
                # FILTER-COMPATIBLE: 2.8ms width survives 1kHz LP with ~2% loss
                'unit_type': 'giant_fiber_a',
                'rate_hz': 2.0,
                'spike_width_ms': 2.8,        # Slightly widened from 2.5ms
                'amp_prox': 0.45,             # 11.25x noise floor, ~0.46V after filtering
                'amp_dist_ratio': 0.89,       # 89% of proximal → 0.40V distal
                'velocity': 12.0,             # 12 m/s → 1.67ms conduction delay (fastest)
                'refractory_ms': 3.0,
                'syn_delay_ms': 2.0,
                'psp_gain': 0.035,
            },
            {
                # Unit 3: Giant Fiber B - largest amplitude, broadest spike
                # FILTER-COMPATIBLE: 3.2ms width survives 1kHz LP with ~2% loss
                'unit_type': 'giant_fiber_b',
                'rate_hz': 1.5,
                'spike_width_ms': 3.2,        # Slightly widened from 3.0ms
                'amp_prox': 0.70,             # 17.5x noise floor (largest), ~0.72V after filtering
                'amp_dist_ratio': 0.80,       # 80% of proximal → 0.56V distal
                'velocity': 10.0,             # 10 m/s → 2.0ms conduction delay
                'refractory_ms': 3.5,
                'syn_delay_ms': 2.5,
                'psp_gain': 0.045,
            },
            {
                # Unit 4: Motor Neuron A - medium amplitude
                # FILTER-COMPATIBLE: 2.4ms width survives 1kHz LP with ~3% loss
                'unit_type': 'motor_a',
                'rate_hz': 5.0,
                'spike_width_ms': 2.4,        # Widened from 2.0ms
                'amp_prox': 0.32,             # 8x noise floor, ~0.31V after filtering
                'amp_dist_ratio': 0.70,       # 70% of proximal → 0.22V distal
                'velocity': 5.0,              # 5 m/s → 4.0ms conduction delay
                'refractory_ms': 2.5,
                'syn_delay_ms': 4.0,
                'psp_gain': 0.025,
            },
            {
                # Unit 5: Motor Neuron B - medium-large amplitude, slowest conduction
                # FILTER-COMPATIBLE: 2.6ms width survives 1kHz LP with ~0% loss
                'unit_type': 'motor_b',
                'rate_hz': 3.5,
                'spike_width_ms': 2.6,        # Widened from 2.2ms
                'amp_prox': 0.55,             # 13.75x noise floor, ~0.55V after filtering
                'amp_dist_ratio': 0.60,       # 60% of proximal → 0.33V distal (most attenuation)
                'velocity': 3.0,              # 3 m/s → 6.67ms conduction delay (slowest)
                'refractory_ms': 2.8,
                'syn_delay_ms': 5.0,
                'psp_gain': 0.030,
            },
        ]
        
        self._units.clear()
        
        # Use only the number of units requested (default 6)
        units_to_create = min(num_units, len(HARDCODED_UNITS))
        
        for i in range(units_to_create):
            params = HARDCODED_UNITS[i]
            
            # Calculate spike template length from width
            spike_width_ms = params['spike_width_ms']
            spike_width_s = spike_width_ms / 1000.0
            spike_len = max(16, int(spike_width_s * sample_rate))
            
            # Generate triphasic spike template with fixed width
            width_factor = spike_width_ms / 1.0  # Normalize to 1.0ms reference
            template = generate_triphasic_template(spike_len, width_factor)
            
            # Convert time-based parameters to samples
            refractory_samples = int(round(params['refractory_ms'] / 1000.0 * sample_rate))
            syn_delay_samples = int(round(params['syn_delay_ms'] / 1000.0 * sample_rate))
            
            self._units.append({
                'unit_type': params['unit_type'],
                'template': template,
                'templ_len': len(template),
                'rate_hz': params['rate_hz'],
                'amp_prox': params['amp_prox'],
                'amp_dist_ratio': params['amp_dist_ratio'],
                'velocity': params['velocity'],
                'refractory_samples': refractory_samples,
                'last_spike_sample': -100000,  # Initialize far in the past
                'syn_delay_samples': syn_delay_samples,
                'psp_gain': params['psp_gain'],
            })
            
            # Calculate expected conduction delay for logging
            cond_delay_ms = (self._distance_m / params['velocity']) * 1000.0
            
            logger.info(
                "Unit %d [%s]: prox=%.2fV, dist=%.2fV, width=%.1fms, "
                "vel=%.1fm/s, cond_delay=%.2fms, syn_delay=%.1fms, rate=%.1fHz",
                i, params['unit_type'], 
                params['amp_prox'], 
                params['amp_prox'] * params['amp_dist_ratio'],
                spike_width_ms, 
                params['velocity'],
                cond_delay_ms,
                params['syn_delay_ms'],
                params['rate_hz']
            )

        # Calculate buffer size for extracellular wave buffers
        # PSPs now use event-based tracking, so buffers only need to handle
        # spike templates and conduction delays for extracellular signals
        max_template = max(u['templ_len'] for u in self._units) if self._units else 0
        distance_m = self._distance_m
        max_ec_delay = 0
        for u in self._units:
            delay_s = distance_m / max(1e-6, u['velocity'])
            max_ec_delay = max(max_ec_delay, int(round(delay_s * sample_rate)))
        
        chunk_size = self.config.chunk_size if self.config is not None else 0
        # Buffer margin for wave buffers: just need to handle spike templates and conduction delays
        self._buffer_margin = max(max_template, max_ec_delay, chunk_size)
        
        # Initialize wave buffers for extracellular signals 
        buf_len = chunk_size + self._buffer_margin if self.config else 0
        self._unit_wave_buffers = [np.zeros(buf_len, dtype=np.float32) for _ in self._units]
        self._buf_len = buf_len
        
        # Reset PSP tracking
        self._active_psps.clear()
        self._global_sample_counter = 0
    # ------------- BaseSource overrides -------------

    def _open_impl(self, device_id: str) -> None:
        # Nothing to open for the simulator
        pass

    def _close_impl(self) -> None:
        # Nothing to close; worker should already be stopped
        pass

    def _configure_impl(self, sample_rate: int, channels, chunk_size: int, **options) -> ActualConfig:
        # Validate sample rate
        supported = [10000, 20000]
        if int(sample_rate) not in supported:
            sample_rate = 20000
        # Generate random units
        # Multiple units with variable amplitudes for realistic simulation
        n_units = int(options.get('num_units', 6))
        self._line_hum_amp = float(options.get('line_hum_amp', self._default_line_hum_amp))
        self._line_hum_freq = float(options.get('line_hum_freq', 60.0))
        self._line_hum_phase = 0.0
        self._line_hum_omega = 0.0 if sample_rate <= 0 else 2.0 * np.pi * (self._line_hum_freq / sample_rate)
        # Build channel list (full device list already cached on open)
        selected = [c for c in self._available_channels if c.id in set(channels)]
        # Initialize unit templates and buffers with the new sample rate
        # Temporarily create a dummy config to size buffers
        self.config = ActualConfig(sample_rate=sample_rate, channels=selected, chunk_size=chunk_size)
        self._initialize_units(sample_rate=sample_rate, num_units=n_units)
        return self.config

    def _start_impl(self) -> None:
        assert self.config is not None
        self._global_sample_counter = 0
        # Reset per-unit spike timing to avoid stale refractory state after restart
        for u in self._units:
            u['last_spike_sample'] = -100000
        # Clear any lingering PSPs from previous run
        self._active_psps.clear()
        # Reset RNG to ensure repeatable spike patterns on each start
        self._rng = np.random.default_rng(self.RANDOM_SEED)
        if self._worker and self._worker.is_alive():
            return

        def _loop():
            chunk_size = self.config.chunk_size
            sr = self.config.sample_rate
            chunk_duration = chunk_size / sr
            
            wave_buffers = self._unit_wave_buffers
            buf_len = self._buf_len
            # Main Simulation Loop
            # --------------------
            # Generates data in blocks of `chunk_size`.
            # Maintains real-time timing by sleeping until the next deadline.
            next_deadline = time.perf_counter() + chunk_duration
            counter = 0

            while not self.stop_event.is_set():
                loop_start = time.perf_counter()
                active_infos = self.get_active_channels()
                active_ids = [ch.id for ch in active_infos]
                if not active_ids:
                    time.sleep(0.01)
                    continue

                # Resolve active channel ids to types in order
                id_to_type = {0: 'extracellular_prox', 1: 'extracellular_dist', 2: 'intracellular'}

                if (
                    self._chunk_buffer is None
                    or self._chunk_buffer.shape[0] != chunk_size
                    or self._chunk_buffer.shape[1] != len(active_ids)
                ):
                    self._chunk_buffer = np.zeros((chunk_size, len(active_ids)), dtype=np.float32)
                else:
                    self._chunk_buffer.fill(0.0)
                data_chunk = self._chunk_buffer

                # ============================================================
                # STEP 1: SHIFT WAVE BUFFERS (extracellular only)
                # ============================================================
                # Wave buffers still use rolling buffer for extracellular signals
                if chunk_size < buf_len:
                    shift = buf_len - chunk_size
                    for ui in range(len(wave_buffers)):
                        wave = wave_buffers[ui]
                        wave[:shift] = wave[chunk_size:]
                        wave[shift:] = 0.0
                else:
                    for ui in range(len(wave_buffers)):
                        wave_buffers[ui][:] = 0.0

                # ============================================================
                # STEP 2: GENERATE NEW SPIKES AND PSP EVENTS
                # ============================================================
                # Generate spikes into wave buffers with refractory period enforcement.
                # For PSPs: create ActivePSP events instead of writing to buffers.
                chunk_start_sample = self._global_sample_counter
                
                # Calculate maximum conduction delay across all units
                max_delay = 0
                for u in self._units:
                    delay_s = self._distance_m / max(1e-6, u['velocity'])
                    delay_samples = int(round(delay_s * sr))
                    max_delay = max(max_delay, delay_samples)

                for ui, u in enumerate(self._units):
                    p_spike = u['rate_hz'] / sr
                    events = self._rng.random(chunk_size) < p_spike
                    candidate_offs = np.where(events)[0]
                    templ = u['template']
                    templ_len = u['templ_len']
                    wave_buf = wave_buffers[ui]
                    psp_gain = u['psp_gain']
                    refractory_samples = u['refractory_samples']

                    # Calculate conduction delay for this unit
                    delay_s = self._distance_m / max(1e-6, u['velocity'])
                    conduction_delay_samples = int(round(delay_s * sr))

                    # Filter candidate spikes by refractory period
                    # last_spike_sample is the global sample index of the last spike
                    for off in candidate_offs:
                        global_sample = chunk_start_sample + off

                        # Check refractory period: skip if too soon after last spike
                        if global_sample - u['last_spike_sample'] < refractory_samples:
                            continue  # Still in refractory period, skip this spike

                        # Record this spike time for future refractory checks
                        u['last_spike_sample'] = global_sample

                        # Insert spike with offset to ensure both channels can read it
                        # Adding max_delay ensures proximal (reading from delay:delay+chunk) can see it
                        insert_pos = off + max_delay
                        amp = u['amp_prox']
                        scaled = templ * amp
                        end = insert_pos + templ_len
                        if end > buf_len:
                            end = buf_len
                        if insert_pos < buf_len:
                            wave_buf[insert_pos:end] += scaled[: end - insert_pos]

                        # Create PSP event (will be rendered in step 3)
                        # CRITICAL TIMING CORRECTION:
                        # The buffer insertion adds max_delay offset, but proximal reads from wave[delay:]
                        # This means proximal spike appears in output at: chunk_start + off
                        # Distal reads from wave[0:], so distal spike appears at: chunk_start + off + max_delay
                        # We need to calculate when the distal spike actually appears in the output stream
                        # and add synaptic delay from there.
                        #
                        # Actual output timing:
                        # - Proximal in output: chunk_start_sample + off (because it reads from offset position)
                        # - Distal in output: chunk_start_sample + off + max_delay (because it reads from start)
                        # - PSP should appear AFTER distal by synaptic delay
                        #
                        # Therefore: PSP_start = (chunk_start + off + max_delay) + synaptic_delay
                        psp_delay_samples = u['syn_delay_samples']
                        # Skip template[0] which is forced to 0.0 for baseline correction
                        new_psp = ActivePSP(
                            start_sample=global_sample + max_delay + psp_delay_samples + 1,
                            template=self._psp_template[1:],  # Skip the forced-zero first sample
                            gain=psp_gain,
                            unit_index=ui,
                        )
                        self._active_psps.append(new_psp)


                # CRITICAL: Preserve the order of active_ids as provided by the UI.
                # The dispatcher and UI expect data columns to match the user-added order,
                # NOT sorted by channel ID. Sorting here caused data mismatches when
                # channels were added out of order (e.g., adding ch0, ch2, ch1).

                
                # ============================================================
                # STEP 3: RENDER CHANNELS
                # ============================================================
                chunk_end_sample = chunk_start_sample + chunk_size
                
                for col, cid in enumerate(active_ids):
                    ch_type = id_to_type.get(cid, 'extracellular_prox')
                    if ch_type == 'extracellular_prox':
                        # PROXIMAL: Signal arrives FIRST (from CNS toward muscle)
                        # Read from the delayed portion of the buffer - this makes spikes
                        # appear at earlier output sample positions (first in time)
                        sig = np.zeros(chunk_size, dtype=np.float32)
                        for ui, u in enumerate(self._units):
                            wave = wave_buffers[ui]
                            # Apply conduction delay offset so proximal appears first
                            delay_s = self._distance_m / max(1e-6, u['velocity'])
                            delay = int(round(delay_s * sr))
                            start = delay
                            end = start + chunk_size
                            if end > buf_len:
                                end = buf_len
                            seg = wave[start:end]
                            if seg.shape[0] < chunk_size:
                                padded = np.zeros(chunk_size, dtype=np.float32)
                                padded[: seg.shape[0]] = seg
                                seg = padded
                            sig += seg  # Wave buffer already contains amplitude-scaled spikes
                        sig += self._rng.normal(0.0, self._noise_level, size=chunk_size).astype(np.float32)
                        data_chunk[:, col] = sig
                    elif ch_type == 'extracellular_dist':
                        # DISTAL: Signal arrives AFTER proximal (delayed by conduction time)
                        # Read from the immediate portion of the buffer - spikes appear at
                        # later output sample positions relative to proximal
                        sig = np.zeros(chunk_size, dtype=np.float32)
                        for ui, u in enumerate(self._units):
                            wave = wave_buffers[ui]
                            # No offset - read immediate buffer position
                            seg = wave[:chunk_size]
                            if seg.shape[0] < chunk_size:
                                padded = np.zeros(chunk_size, dtype=np.float32)
                                padded[: seg.shape[0]] = seg
                                seg = padded
                            sig += seg * u['amp_dist_ratio']
                        sig += self._rng.normal(0.0, self._noise_level, size=chunk_size).astype(np.float32)
                        data_chunk[:, col] = sig
                    else:  # intracellular - render PSPs
                        # STEP 3: INTRACELLULAR PSPs (event-based, can span multiple chunks)
                        ic_signal = np.zeros(chunk_size, dtype=np.float32)
                        
                        for psp in self._active_psps:
                            psp_start_global = psp.start_sample
                            psp_end_global = psp.start_sample + len(psp.template)
                            
                            # Check if PSP overlaps with current chunk
                            if psp_end_global <= chunk_start_sample or psp_start_global >= chunk_end_sample:
                                continue  # No overlap
                            
                            # Calculate where in the template to start reading
                            template_start_idx = max(0, chunk_start_sample - psp_start_global)
                            # Calculate where in the chunk to start writing
                            chunk_start_idx = max(0, psp_start_global - chunk_start_sample)
                            # Calculate how many samples to copy
                            overlap_len = min(chunk_end_sample - psp_start_global, len(psp.template)) - template_start_idx
                            
                            # DIAGNOSTIC: Log if overlap_len is problematic
                            if overlap_len <= 0:
                                logger.warning(
                                    "PSP rendering bug - overlap_len <= 0, PSP will be skipped",
                                    extra={
                                        "chunk": counter,
                                        "psp_start": psp_start_global,
                                        "psp_end": psp_end_global,
                                        "psp_len": len(psp.template),
                                        "chunk_start": chunk_start_sample,
                                        "chunk_end": chunk_end_sample,
                                        "template_start_idx": template_start_idx,
                                        "chunk_start_idx": chunk_start_idx,
                                        "overlap_len": overlap_len,
                                    }
                                )
                            
                            
                            if overlap_len > 0:
                                # Add the PSP slice to the intracellular signal
                                template_slice = psp.template[template_start_idx:template_start_idx + overlap_len]
                                ic_signal[chunk_start_idx:chunk_start_idx + overlap_len] += template_slice * psp.gain
                        
                        # Apply baseline and noise to the intracellular signal
                        ic_signal = -0.070 + ic_signal + self._rng.normal(0.0, self._noise_level * 0.1, size=chunk_size).astype(np.float32)
                        data_chunk[:, col] = ic_signal

                # Add line hum if enabled
                if self._line_hum_amp != 0.0 and self._line_hum_omega != 0.0:
                    idx = np.arange(chunk_size, dtype=np.float64)
                    hum = (self._line_hum_amp * np.sin(self._line_hum_phase + self._line_hum_omega * idx)).astype(np.float32)
                    data_chunk += hum[:, None]
                    self._line_hum_phase = (self._line_hum_phase + self._line_hum_omega * chunk_size) % (2.0 * np.pi)

                chunk_meta = {"active_channel_ids": active_ids}
                
                # Emit the chunk
                self.emit_array(data_chunk, mono_time=time.monotonic())

                # ============================================================
                # STEP 4: CLEANUP
                # ============================================================
                # Remove PSPs that have completely finished
                self._active_psps = [psp for psp in self._active_psps if psp.start_sample + len(psp.template) > chunk_end_sample]
                
                # Update global sample counter
                self._global_sample_counter = chunk_end_sample
                
                # if counter % 20 == 0:
                #     with open("source_debug.txt", "a") as f:
                #         f.write(f"Source: sample={self._global_sample_counter}, active_psps={len(self._active_psps)}\n")

                next_deadline += chunk_duration
                time.sleep(max(0.0, next_deadline - time.perf_counter()))
                counter += 1

        self._worker = threading.Thread(target=_loop, name="SimPhys-Worker", daemon=True)
        self._worker.start()

    def _stop_impl(self) -> None:
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
