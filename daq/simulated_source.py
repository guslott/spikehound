# daq/simulated_source.py
import numpy as np
import time
import threading
from .base_source import BaseSource, Chunk, DeviceInfo, ChannelInfo, Capabilities, ActualConfig

class SimulatedPhysiologySource(BaseSource):
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

    @classmethod
    def device_class_name(cls) -> str:
        return "Simulated"

    def __init__(self, queue_maxsize: int = 64) -> None:
        super().__init__(queue_maxsize=queue_maxsize)
        self._noise_level = 0.05
        self._distance_m = 0.02  # Prox→Dist electrode spacing for delay calc
        self._units = []
        self._worker: threading.Thread | None = None
        # Buffers initialized in _configure_impl
        self._unit_wave_buffers = []
        self._unit_imp_buffers = []
        self._buf_len = 0
        self._psp_template = np.zeros(1)
        self._psp_len = 1
        self._buffer_margin = 0

    # ---- Discovery ------------------------------------------------------------
    @classmethod
    def list_available_devices(cls) -> list[DeviceInfo]:
        return [DeviceInfo(id="sim0", name="Simulated Physiology (virtual)")]

    def get_capabilities(self, device_id: str) -> Capabilities:
        return Capabilities(max_channels_in=3, sample_rates=None, dtype="float32")

    def list_available_channels(self, device_id: str):
        return [
            ChannelInfo(id=0, name='Extracellular Proximal', units='V'),
            ChannelInfo(id=1, name='Extracellular Distal', units='V'),
            ChannelInfo(id=2, name='Intracellular', units='V'),
        ]

    def _initialize_units(self, sample_rate: int, num_units: int):
        """Sample per-unit parameters and initialize rolling buffers.

        - Units get a spike template, firing rate, base amplitude at Prox,
          distal amplitude ratio, conduction velocity, synaptic delay, and PSP gain.
        - Buffers store generated waveforms and unit impulses (soma events).
        """
        # PSP kernel (alpha function)
        psp_len = int(0.020 * sample_rate)
        t_psp = np.linspace(0, 5, psp_len)
        self._psp_template = t_psp * np.exp(-t_psp)
        self._psp_template /= np.max(self._psp_template)
        self._psp_len = len(self._psp_template)

        self._units.clear()
        rng = np.random.default_rng()  # independent RNG per session

        for i in range(num_units):
            spike_duration_s = 0.0012 + rng.random() * 0.0018
            spike_len = max(8, int(spike_duration_s * sample_rate))
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
                'syn_delay_samples': int(round(syn_delay_s * sample_rate)),
                'psp_gain': psp_gain,
            })

        # Precompute max margins for buffers
        max_template = max(u['templ_len'] for u in self._units) if self._units else 0
        distance_m = self._distance_m
        max_ec_delay = 0
        for u in self._units:
            delay_s = distance_m / max(1e-6, u['velocity'])
            max_ec_delay = max(max_ec_delay, int(round(delay_s * sample_rate)))
        max_syn = max((u['syn_delay_samples'] for u in self._units), default=0)
        self._buffer_margin = max(max_template, max_ec_delay, max_syn + self._psp_len)

        buf_len = self.config.chunk_size + self._buffer_margin if self.config else 0
        self._unit_wave_buffers = [np.zeros(buf_len) for _ in self._units]
        self._unit_imp_buffers = [np.zeros(buf_len) for _ in self._units]
        self._buf_len = buf_len
    # ------------- BaseSource overrides -------------

    def _open_impl(self, device_id: str) -> None:
        # Nothing to open for the simulator
        pass

    def _close_impl(self) -> None:
        # Nothing to close; worker should already be stopped
        pass

    def _configure_impl(self, sample_rate: int, channels, chunk_size: int, **options) -> ActualConfig:
        num_units = int(options.get('num_units', 5))
        # Build channel list (full device list already cached on open)
        selected = [c for c in self._available_channels if c.id in set(channels)]
        # Initialize unit templates and buffers with the new sample rate
        # Temporarily create a dummy config to size buffers
        self.config = ActualConfig(sample_rate=sample_rate, channels=selected, chunk_size=chunk_size)
        self._initialize_units(sample_rate=sample_rate, num_units=num_units)
        return self.config

    def _start_impl(self) -> None:
        assert self.config is not None
        if self._worker and self._worker.is_alive():
            return

        def _loop():
            chunk_size = self.config.chunk_size
            sr = self.config.sample_rate
            chunk_duration = chunk_size / sr
            wave_buffers = self._unit_wave_buffers
            imp_buffers = self._unit_imp_buffers
            buf_len = self._buf_len
            next_deadline = time.perf_counter()

            # Resolve active channel ids to types in order
            id_to_type = {0: 'extracellular_prox', 1: 'extracellular_dist', 2: 'intracellular'}
            active_ids = [ch.id for ch in self.get_active_channels()]

            while not self.stop_event.is_set():
                loop_start = time.perf_counter()

                # Roll buffers and clear the new segment
                for i in range(len(wave_buffers)):
                    wave_buffers[i] = np.roll(wave_buffers[i], -chunk_size)
                    imp_buffers[i] = np.roll(imp_buffers[i], -chunk_size)
                    wave_buffers[i][-chunk_size:] = 0.0
                    imp_buffers[i][-chunk_size:] = 0.0

                start_idx = buf_len - chunk_size
                # Generate new events per unit
                for ui, u in enumerate(self._units):
                    p_spike = u['rate_hz'] / sr
                    events = np.random.rand(chunk_size) < p_spike
                    offs = np.where(events)[0]
                    templ = u['template']
                    for off in offs:
                        i0 = start_idx + off
                        if i0 + u['templ_len'] < buf_len:
                            amp = u['amp_prox'] * (0.95 + 0.10 * np.random.rand())
                            wave_buffers[ui][i0 : i0 + u['templ_len']] += templ * amp
                            imp_buffers[ui][i0] += 1.0

                # Build output in the selected order
                data_chunk = np.zeros((chunk_size, len(active_ids)), dtype=np.float32)

                for col, cid in enumerate(active_ids):
                    ch_type = id_to_type.get(cid, 'extracellular_prox')
                    if ch_type == 'extracellular_prox':
                        sig = np.zeros(chunk_size)
                        for ui, _ in enumerate(self._units):
                            seg = wave_buffers[ui][-chunk_size:]
                            sig += seg
                        data_chunk[:, col] = sig + np.random.randn(chunk_size) * self._noise_level
                    elif ch_type == 'extracellular_dist':
                        sig = np.zeros(chunk_size)
                        for ui, u in enumerate(self._units):
                            delay_s = self._distance_m / max(1e-6, u['velocity'])
                            delay = int(round(delay_s * sr))
                            seg = wave_buffers[ui][delay : delay + chunk_size]
                            sig += seg * u['amp_dist_ratio']
                        data_chunk[:, col] = sig + np.random.randn(chunk_size) * self._noise_level
                    else:  # intracellular
                        sig = np.zeros(chunk_size)
                        for ui, u in enumerate(self._units):
                            base = np.zeros(chunk_size + u['syn_delay_samples'] + self._psp_len)
                            base[u['syn_delay_samples'] : u['syn_delay_samples'] + chunk_size] = imp_buffers[ui][-chunk_size:]
                            psp = np.convolve(base, self._psp_template, mode='full')[:chunk_size]
                            sig += psp * u['psp_gain']
                        data_chunk[:, col] = -0.070 + sig + np.random.randn(chunk_size) * (self._noise_level * 0.1)

                self.emit_array(data_chunk, mono_time=loop_start)

                next_deadline += chunk_duration
                time.sleep(max(0.0, next_deadline - time.perf_counter()))

        self._worker = threading.Thread(target=_loop, name="SimPhys-Worker", daemon=True)
        self._worker.start()

    def _stop_impl(self) -> None:
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
