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
        self._unit_psp_buffers = []
        self._buf_len = 0
        self._psp_template = np.zeros(1)
        self._psp_len = 1
        self._buffer_margin = 0
        self._default_line_hum_amp = 0.05
        self._line_hum_amp = self._default_line_hum_amp
        self._line_hum_freq = 60.0
        self._line_hum_phase = 0.0
        self._line_hum_omega = 0.0
        self._chunk_buffer: np.ndarray | None = None

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
        # Extend tail to ensure the waveform decays smoothly to zero so it does not
        # step down at chunk boundaries.
        tail = float(self._psp_template[-1])
        while tail > 1e-4:
            tail *= 0.5
            self._psp_template = np.append(self._psp_template, tail)
        self._psp_template = np.append(self._psp_template, 0.0).astype(np.float32, copy=False)
        self._psp_len = len(self._psp_template)

        self._units.clear()
        rng = np.random.default_rng()  # independent RNG per session

        classes = [
            {"amp": (0.1, 0.2), "width": (0.0006, 0.0010)},
            {"amp": (0.3, 0.5), "width": (0.0010, 0.0018)},
            {"amp": (0.6, 0.9), "width": (0.0015, 0.0025)},
            {"amp": (1.0, 1.5), "width": (0.0025, 0.0048)},
        ]
        for i in range(num_units):
            cls = classes[i % len(classes)]
            spike_duration_s = cls["width"][0] + rng.random() * (cls["width"][1] - cls["width"][0])
            spike_len = max(8, int(spike_duration_s * sample_rate))
            t_spike = np.linspace(-1, 1, spike_len)
            template = (1 - t_spike**2) * np.exp(-t_spike**2 / 0.5)
            template = -template
            # Remove DC component while keeping endpoints pinned at zero.
            template -= np.mean(template)
            edge_val = template[0]
            template -= edge_val
            basis = 1.0 - (t_spike / t_spike[-1])**2
            basis_mean = np.mean(basis)
            if abs(basis_mean) > 1e-12:
                template -= (template.mean() / basis_mean) * basis
            peak = np.max(np.abs(template))
            if peak > 1e-12:
                template /= peak
            # Truncate tail once magnitude falls below 1% of the peak.
            significant = np.where(np.abs(template) >= 0.01)[0]
            if significant.size:
                end_idx = significant[-1]
                trimmed = template[: end_idx + 1]
                tail = []
                tail_value = trimmed[-1]
                while abs(tail_value) > 0.01:
                    tail_value *= 0.5
                    tail.append(tail_value)
                tail.append(0.0)
                template = np.concatenate((trimmed, np.asarray(tail, dtype=np.float64)))
            else:
                template = template[:1]
            template = template.astype(np.float32, copy=False)
            template -= np.mean(template)
            if template.size > 1:
                edge_start = template[0]
                edge_end = template[-1]
                ramp = np.linspace(edge_start, edge_end, template.size, dtype=np.float32)
                template -= ramp
            template -= np.mean(template)
            template[0] = 0.0
            template[-1] = 0.0
            template -= np.mean(template)
            template[0] = 0.0
            template[-1] = 0.0
            peak = np.max(np.abs(template))
            if peak > 1e-12:
                template /= peak

            rate_hz = 2.0 + rng.random()
            amp_min, amp_max = cls["amp"]
            base_amp_prox = amp_min + rng.random() * (amp_max - amp_min)
            # Keep spikes comfortably above the noise floor.
            base_amp_prox = max(base_amp_prox, self._noise_level * 6.0 + 0.05)
            distal_ratio = 0.5 + rng.random() * 0.7  # 0.5–1.2 relative to prox
            distal_ratio = max(0.6, distal_ratio)
            velocity_m_per_s = 10.0 + rng.random() * 50.0
            syn_delay_s = 0.002 + rng.random() * 0.004  # 2–6 ms
            psp_gain = 0.02 + rng.random() * 0.03

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
        self._unit_wave_buffers = [np.zeros(buf_len, dtype=np.float32) for _ in self._units]
        self._unit_psp_buffers = [np.zeros(buf_len, dtype=np.float32) for _ in self._units]
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
        self._line_hum_amp = float(options.get('line_hum_amp', self._default_line_hum_amp))
        self._line_hum_freq = float(options.get('line_hum_freq', 60.0))
        self._line_hum_phase = 0.0
        self._line_hum_omega = 0.0 if sample_rate <= 0 else 2.0 * np.pi * (self._line_hum_freq / sample_rate)
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
            psp_buffers = self._unit_psp_buffers
            buf_len = self._buf_len
            next_deadline = time.perf_counter()

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

                # Generate new events per unit
                for ui, u in enumerate(self._units):
                    p_spike = u['rate_hz'] / sr
                    events = np.random.rand(chunk_size) < p_spike
                    offs = np.where(events)[0]
                    templ = u['template']
                    templ_len = u['templ_len']
                    wave_buf = wave_buffers[ui]
                    psp_buf = psp_buffers[ui]
                    psp_gain = u['psp_gain']
                    syn_delay = u['syn_delay_samples']
                    for off in offs:
                        amp = u['amp_prox'] * (0.95 + 0.10 * np.random.rand())
                        amp = max(amp, self._noise_level * 5.0 + 0.05)
                        scaled = templ * amp
                        end = off + templ_len
                        if end > buf_len:
                            end = buf_len
                        wave_buf[off:end] += scaled[: end - off]
                        psp_start = off + syn_delay
                        psp_end = psp_start + self._psp_len
                        if psp_end > buf_len:
                            psp_end = buf_len
                        span = psp_end - psp_start
                        if span > 0:
                            psp_buf[psp_start:psp_end] += self._psp_template[:span] * psp_gain

                for col, cid in enumerate(active_ids):
                    ch_type = id_to_type.get(cid, 'extracellular_prox')
                    if ch_type == 'extracellular_prox':
                        sig = np.zeros(chunk_size, dtype=np.float32)
                        for ui, _ in enumerate(self._units):
                            seg = wave_buffers[ui][:chunk_size]
                            sig += seg
                        sig += np.random.normal(0.0, self._noise_level, size=chunk_size).astype(np.float32)
                        data_chunk[:, col] = sig
                    elif ch_type == 'extracellular_dist':
                        sig = np.zeros(chunk_size, dtype=np.float32)
                        for ui, u in enumerate(self._units):
                            delay_s = self._distance_m / max(1e-6, u['velocity'])
                            delay = int(round(delay_s * sr))
                            start = delay
                            end = start + chunk_size
                            if end > buf_len:
                                end = buf_len
                            seg = wave_buffers[ui][start:end]
                            if seg.shape[0] < chunk_size:
                                padded = np.zeros(chunk_size, dtype=np.float32)
                                padded[: seg.shape[0]] = seg
                                seg = padded
                            sig += seg * u['amp_dist_ratio']
                        sig += np.random.normal(0.0, self._noise_level, size=chunk_size).astype(np.float32)
                        data_chunk[:, col] = sig
                    else:  # intracellular
                        sig = np.zeros(chunk_size, dtype=np.float32)
                        for ui, u in enumerate(self._units):
                            sig += psp_buffers[ui][:chunk_size]
                        sig = -0.070 + sig + np.random.normal(0.0, self._noise_level * 0.1, size=chunk_size).astype(np.float32)
                        data_chunk[:, col] = sig

                if self._line_hum_amp != 0.0 and self._line_hum_omega != 0.0:
                    idx = np.arange(chunk_size, dtype=np.float64)
                    hum = (self._line_hum_amp * np.sin(self._line_hum_phase + self._line_hum_omega * idx)).astype(np.float32)
                    data_chunk += hum[:, None]
                    self._line_hum_phase = (self._line_hum_phase + self._line_hum_omega * chunk_size) % (2.0 * np.pi)

                chunk_meta = {"active_channel_ids": list(active_ids)}
                self.emit_array(data_chunk, mono_time=loop_start, meta=chunk_meta)

                # Advance buffers for the next chunk
                if chunk_size < buf_len:
                    shift = buf_len - chunk_size
                    for ui in range(len(wave_buffers)):
                        wave = wave_buffers[ui]
                        wave[:shift] = wave[chunk_size:]
                        wave[shift:] = 0.0
                        psp = psp_buffers[ui]
                        psp[:shift] = psp[chunk_size:]
                        psp[shift:] = 0.0
                else:
                    for ui in range(len(wave_buffers)):
                        wave_buffers[ui][:] = 0.0
                        psp_buffers[ui][:] = 0.0

                next_deadline += chunk_duration
                time.sleep(max(0.0, next_deadline - time.perf_counter()))

        self._worker = threading.Thread(target=_loop, name="SimPhys-Worker", daemon=True)
        self._worker.start()

    def _stop_impl(self) -> None:
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
