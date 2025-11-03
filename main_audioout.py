import threading
import queue
import time
import sounddevice as sd

from audio.player import AudioPlayer, AudioConfig

# Adjust this import path if your sim source lives elsewhere:
from daq.simulated_source import SimulatedPhysiologySource
from core.dispatcher import Dispatcher
from core.models import EndOfStream

IN_SAMPLE_RATE = 20_000
CHUNK_SIZE = 400
SELECTED_CHANNEL = 1  # try 0 or 1 if you don't hear anything
LINE_HUM_FREQ = 440.0
LINE_HUM_AMP = 0.8
LINE_HUM_DURATION = 2.0
SPIKE_PLAY_DURATION = 10.0

class AudioOutput:
    """Single-channel audio output helper around AudioPlayer."""

    def __init__(
        self,
        *,
        input_samplerate: int,
        output_samplerate: int = 44_100,
        device: int | str | None = None,
        volume: float = 0.9,
        queue_maxsize: int = 64,
        blocksize: int = 512,
        ring_seconds: float = 0.75,
        selected_channel: int | None = 0,
    ) -> None:
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        cfg = AudioConfig(
            out_samplerate=output_samplerate,
            out_channels=1,
            device=device,
            gain=volume,
            blocksize=blocksize,
            ring_seconds=ring_seconds,
        )
        self._player = AudioPlayer(
            audio_queue=self._queue,
            input_samplerate=input_samplerate,
            config=cfg,
            selected_channel=selected_channel,
        )

    def start(self) -> None:
        self._player.start()

    def stop(self) -> None:
        self._player.stop()

    def join(self, timeout: float | None = None) -> None:
        self._player.join(timeout=timeout)

    def submit(self, chunk) -> None:
        """Submit a chunk without blocking, dropping oldest on overflow."""
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

def start_audio_feed(source_queue: "queue.Queue", audio_out: AudioOutput, name: str) -> tuple[threading.Thread, threading.Event]:
    """Thread that forwards dispatcher audio queue chunks into the audio output."""
    stop_event = threading.Event()

    def _pump():
        while True:
            try:
                item = source_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue
            if item is EndOfStream:
                break
            audio_out.submit(item)
        # Clear any remaining items so queue is ready for the next phase
        while True:
            try:
                _ = source_queue.get_nowait()
            except queue.Empty:
                break

    thread = threading.Thread(target=_pump, name=name, daemon=True)
    thread.start()
    return thread, stop_event

def push_end_of_stream(q: "queue.Queue") -> None:
    """Ensure an EndOfStream sentinel lands in the queue, dropping oldest if needed."""
    placed = False
    while not placed:
        try:
            q.put_nowait(EndOfStream)
            placed = True
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                break

def main():
    print(sd.query_devices(kind='output'))
    print("Starting audio player…")
    audio_out = AudioOutput(
        input_samplerate=IN_SAMPLE_RATE,
        output_samplerate=44_100,
        device=None,
        volume=0.9,
        queue_maxsize=64,
        blocksize=512,
        ring_seconds=0.75,
        selected_channel=SELECTED_CHANNEL,
    )
    audio_out.start()

    def run_sim_phase(*, line_hum_amp: float, line_hum_freq: float, duration: float, label: str, audio: AudioOutput) -> None:
        print(f"{label}…")
        src = SimulatedPhysiologySource()
        dev = src.list_available_devices()[0]
        src.open(dev.id)
        chans = src.list_available_channels(dev.id)
        channel_ids = [c.id for c in chans]
        src.configure(sample_rate=IN_SAMPLE_RATE,
                      channels=channel_ids,
                      chunk_size=CHUNK_SIZE,
                      num_units=6,
                      line_hum_amp=line_hum_amp,
                      line_hum_freq=line_hum_freq)

        visualization_q: "queue.Queue" = queue.Queue(maxsize=8)
        analysis_q: "queue.Queue" = queue.Queue(maxsize=8)
        audio_q: "queue.Queue" = queue.Queue(maxsize=64)
        logging_q: "queue.Queue" = queue.Queue(maxsize=32)

        disp = Dispatcher(
            raw_queue=src.data_queue,
            visualization_queue=visualization_q,
            analysis_queue=analysis_q,
            audio_queue=audio_q,
            logging_queue=logging_q,
            filter_settings=None,
            poll_timeout=0.05,
        )
        channel_names = [c.name for c in chans]
        disp.set_channel_layout(channel_ids, channel_names)
        disp.set_active_channels(channel_ids)

        feed_name = f"audio-feed-{label.replace(' ', '-').lower()}"
        pump_thread, pump_stop = start_audio_feed(audio_q, audio, feed_name)

        disp.start()
        src.start()

        time.sleep(duration)

        src.stop()
        disp.stop()
        pump_stop.set()
        pump_thread.join(timeout=1.0)
        if pump_thread.is_alive():
            push_end_of_stream(audio_q)
            pump_thread.join(timeout=1.0)
        src.close()

    run_sim_phase(line_hum_amp=LINE_HUM_AMP,
                  line_hum_freq=LINE_HUM_FREQ,
                  duration=LINE_HUM_DURATION,
                  label="Phase 1: line hum 440 Hz",
                  audio=audio_out)

    audio_out.drain()
    print("Switching to spike-only playback…")

    run_sim_phase(line_hum_amp=0.0,
                  line_hum_freq=LINE_HUM_FREQ,
                  duration=SPIKE_PLAY_DURATION,
                  label="Phase 2: spikes only",
                  audio=audio_out)

    print("Stopping…")
    audio_out.stop()
    audio_out.join(timeout=2.0)
    print("Done.")

if __name__ == "__main__":
    main()
