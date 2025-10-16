import time, queue, threading
from daq.simulated_source import SimulatedPhysiologySource
from daq.base_source import Chunk
from audio.player import AudioPlayer, AudioConfig

IN_SR = 20_000
CHUNK = 256
SELECTED = 0  # try 0 or 1 (extracellular), 2 is intracellular

audio_q: "queue.Queue" = queue.Queue(maxsize=64)

def dispatcher(src):
    while src.running:
        try:
            ch: Chunk = src.data_queue.get(timeout=0.05)
            audio_q.put_nowait(ch)   # pass-through (pretend 'filtered')
        except queue.Empty:
            pass

def main():
    print("Setting up simulated source…")
    src = SimulatedPhysiologySource()
    dev = src.list_available_devices()[0]
    src.open(dev.id)
    src.configure(sample_rate=IN_SR, channels=[0,1,2], chunk_size=CHUNK, num_units=6)
    src.start()

    print("Starting dispatcher…")
    t = threading.Thread(target=dispatcher, args=(src,), daemon=True)
    t.start()

    print("Starting audio player…")
    player = AudioPlayer(
        audio_queue=audio_q,
        input_samplerate=IN_SR,
        config=AudioConfig(out_samplerate=44_100, out_channels=1, gain=0.4, blocksize=512),
        selected_channel=SELECTED,
    )
    player.start()

    print("Playing spikes for 10 seconds…")
    time.sleep(10)

    print("Stopping…")
    player.stop(); player.join(timeout=2.0)
    src.stop(); src.close()
    print("Done.")

if __name__ == "__main__":
    main()

