from __future__ import annotations
import argparse
import numpy as np
import h5py
import sounddevice as sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to an HDF5 log (e.g., logs/test_sine.h5)")
    ap.add_argument("--block", type=int, default=2048, help="Frames per block for playback")
    ap.add_argument("--device", type=str, default=None, help="Sound device name or index (optional)")
    args = ap.parse_args()

    with h5py.File(args.infile, "r") as f:
        sr = int(float(f.attrs["sample_rate"]))
        samples = f["/samples"]
        N, C = samples.shape

        print(f"Playing {N} frames at {sr} Hz with {C} channel(s)…")

        stream_args = dict(samplerate=sr, channels=C, dtype="float32")
        if args.device:
            stream_args["device"] = args.device

        with sd.OutputStream(**stream_args) as stream:
            idx = 0
            while idx < N:
                n = min(args.block, N - idx)
                block = samples[idx:idx + n, :].astype(np.float32)
                stream.write(block)
                idx += n


if __name__ == "__main__":
    main()
