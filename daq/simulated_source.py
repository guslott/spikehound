# daq/simulated_source.py
import numpy as np
import time
import queue
from .base_source import DataAcquisitionSource, Chunk, DeviceInfo

class SimulatedPhysiologySource(DataAcquisitionSource):
    """
    Simulates a multi-unit nerve bundle recorded on:
      - Extracellular Proximal (Ex-Prox)
      - Extracellular Distal (Ex-Dist)
      - Intracellular PSP (IC)

    Only hard-coded input: the number of units in the bundle.
    All other per-unit parameters (amplitudes, velocities/delays, firing rates,
    synaptic delays, PSP gains) are randomly sampled once at initialization and
    remain fixed throughout the session.
    """

    def __init__(self, sample_rate: int, chunk_size: int, num_units: int = 5, device: str | None = None):
        super().__init__(sample_rate, chunk_size)
        self.num_units = num_units
        self._noise_level = 0.05

        # Channel definitions
        self._channel_params = {
            'Extracellular Proximal': {'type': 'extracellular_prox'},
            'Extracellular Distal':   {'type': 'extracellular_dist', 'distance_m': 0.02},
            'Intracellular':          {'type': 'intracellular'},
        }

        # Will be filled in by _initialize_units()
        self._units = []  # list of dicts with per-unit fixed params

        self._initialize_units()

    # ---- Discovery ------------------------------------------------------------
    @classmethod
    def list_available_devices(cls):
        """Return a single virtual device for the simulator."""
        return [DeviceInfo(id="sim0", name="Simulated Physiology (virtual)")]

    def _initialize_units(self):
        """Sample per-unit parameters and initialize rolling buffers.

        - Units get a spike template, firing rate, base amplitude at Prox,
          distal amplitude ratio, conduction velocity, synaptic delay, and PSP gain.
        - Buffers store generated waveforms and unit impulses (soma events).
        """
        # PSP kernel (alpha function)
        psp_len = int(0.020 * self.sample_rate)
        t_psp = np.linspace(0, 5, psp_len)
        self._psp_template = t_psp * np.exp(-t_psp)
        self._psp_template /= max(1e-12, np.max(self._psp_template))
        self._psp_len = len(self._psp_template)

        self._units.clear()
        rng = np.random.default_rng()  # independent RNG per session

        for i in range(self.num_units):
            spike_duration_s = 0.0012 + rng.random() * 0.0018
            spike_len = max(8, int(spike_duration_s * self.sample_rate))
            t_spike = np.linspace(-1, 1, spike_len)
            template = (1 - t_spike**2) * np.exp(-t_spike**2 / 0.5)
            template = -template
            template -= np.mean(template)
            template /= max(1e-12, np.max(np.abs(template)))

            rate_hz = 5 + rng.random() * 20
            base_amp_prox = 0.2 + rng.random() * 1.0
            distal_ratio = 0.5 + rng.random() * 0.7  # 0.5–1.2 relative to prox
            velocity_m_per_s = 10.0 + rng.random() * 50.0
            syn_delay_s = 0.002 + rng.random() * 0.004  # 2–6 ms
            psp_gain = 0.005 + rng.random() * 0.02

            self._units.append({
                'template': template,
                'templ_len': len(template),
                'rate_hz': rate_hz,
                'amp_prox': base_amp_prox,
                'amp_dist_ratio': distal_ratio,
                'velocity': velocity_m_per_s,
                'syn_delay_samples': int(round(syn_delay_s * self.sample_rate)),
                'psp_gain': psp_gain,
            })

        # Precompute max margins for buffers
        max_template = max(u['templ_len'] for u in self._units) if self._units else 0
        distance_m = self._channel_params['Extracellular Distal']['distance_m']
        max_ec_delay = 0
        for u in self._units:
            delay_s = distance_m / max(1e-6, u['velocity'])
            max_ec_delay = max(max_ec_delay, int(round(delay_s * self.sample_rate)))
        max_syn = max((u['syn_delay_samples'] for u in self._units), default=0)
        self._buffer_margin = max(max_template, max_ec_delay, max_syn + self._psp_len)

        buf_len = self.chunk_size + self._buffer_margin
        self._unit_wave_buffers = [np.zeros(buf_len) for _ in self._units]
        self._unit_imp_buffers = [np.zeros(buf_len) for _ in self._units]
        self._buf_len = buf_len

    def list_available_channels(self) -> list:
        """Returns the list of channels this simulated device can produce."""
        return list(self._channel_params.keys())

    def run(self):
        """Generate data with fixed per-unit parameters and emit Chunk objects."""
        chunk_duration = self.chunk_size / self.sample_rate

        wave_buffers = self._unit_wave_buffers
        imp_buffers = self._unit_imp_buffers
        buf_len = self._buf_len

        next_deadline = time.perf_counter()

        while self.is_running():
            loop_start = time.perf_counter()

            with self._channel_lock:
                active_ch_names = self.active_channels[:]

            if not active_ch_names:
                next_deadline += chunk_duration
                time.sleep(max(0.0, next_deadline - time.perf_counter()))
                continue

            # Roll buffers and clear new segment
            for i in range(len(wave_buffers)):
                wave_buffers[i] = np.roll(wave_buffers[i], -self.chunk_size)
                imp_buffers[i] = np.roll(imp_buffers[i], -self.chunk_size)
                wave_buffers[i][-self.chunk_size:] = 0.0
                imp_buffers[i][-self.chunk_size:] = 0.0

            start_idx = buf_len - self.chunk_size
            # Generate new events per unit
            for ui, u in enumerate(self._units):
                p_spike = u['rate_hz'] / self.sample_rate
                events = np.random.rand(self.chunk_size) < p_spike
                offs = np.where(events)[0]
                if offs.size == 0:
                    continue
                templ = u['template']
                for off in offs:
                    i0 = start_idx + off
                    if i0 + u['templ_len'] < buf_len:
                        # Optional small trial-to-trial jitter in amplitude
                        amp = u['amp_prox'] * (0.95 + 0.10 * np.random.rand())
                        wave_buffers[ui][i0 : i0 + u['templ_len']] += templ * amp
                        imp_buffers[ui][i0] += 1.0

            # Build output chunk
            num_cols = len(active_ch_names)
            data_chunk = np.zeros((self.chunk_size, num_cols), dtype=float)

            for col, ch_name in enumerate(active_ch_names):
                ch_type = self._channel_params[ch_name]['type']
                if ch_type == 'extracellular_prox':
                    sig = np.zeros(self.chunk_size)
                    for ui, _ in enumerate(self._units):
                        seg = wave_buffers[ui][-self.chunk_size :]
                        sig += seg
                    data_chunk[:, col] = sig + np.random.randn(self.chunk_size) * self._noise_level

                elif ch_type == 'extracellular_dist':
                    sig = np.zeros(self.chunk_size)
                    distance_m = self._channel_params['Extracellular Distal']['distance_m']
                    for ui, u in enumerate(self._units):
                        delay_s = distance_m / max(1e-6, u['velocity'])
                        delay = int(round(delay_s * self.sample_rate))
                        seg = wave_buffers[ui][delay : delay + self.chunk_size]
                        sig += seg * u['amp_dist_ratio']
                    data_chunk[:, col] = sig + np.random.randn(self.chunk_size) * self._noise_level

                elif ch_type == 'intracellular':
                    # Sum per-unit PSPs with their own synaptic delays
                    sig = np.zeros(self.chunk_size)
                    for ui, u in enumerate(self._units):
                        base = np.zeros(self.chunk_size + u['syn_delay_samples'] + self._psp_len)
                        base[u['syn_delay_samples'] : u['syn_delay_samples'] + self.chunk_size] = imp_buffers[ui][-self.chunk_size :]
                        psp = np.convolve(base, self._psp_template, mode='full')[: self.chunk_size]
                        sig += psp * u['psp_gain']
                    # resting potential around -70mV visual offset
                    data_chunk[:, col] = -0.070 + sig + np.random.randn(self.chunk_size) * (self._noise_level * 0.1)

            # Emit chunk (drop-oldest if full to avoid blocking)
            start_sample, seq = self._next_chunk_meta()
            try:
                self.data_queue.put_nowait(Chunk(start_sample=start_sample, mono_time=loop_start, seq=seq, data=data_chunk))
            except queue.Full:
                try:
                    _ = self.data_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.data_queue.put_nowait(Chunk(start_sample=start_sample, mono_time=loop_start, seq=seq, data=data_chunk))
                except Exception:
                    pass

            next_deadline += chunk_duration
            time.sleep(max(0.0, next_deadline - time.perf_counter()))
