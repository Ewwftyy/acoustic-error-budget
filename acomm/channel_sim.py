"""Simulated room channel (used by --sim to test the pipeline without speaker/mic)."""
import numpy as np


def room(x, fs, rng, noise_rms=0.002, ppm=5.0, drive=1.6, n_echo=6, max_delay_ms=6.0):
    # speaker nonlinearity (soft clipping)
    x = np.tanh(drive * x) / drive
    # multipath: direct path + decaying echoes
    h = np.zeros(int(max_delay_ms * 1e-3 * fs) + 1)
    h[0] = 1.0
    for _ in range(n_echo):
        d = rng.integers(1, len(h))
        h[d] += rng.normal(0, 0.35) * np.exp(-d / len(h) * 2)
    y = np.convolve(x, h)[: len(x)]
    # sample-clock offset between playback and recording
    t = np.arange(len(y)) * (1 + ppm * 1e-6)
    y = np.interp(t, np.arange(len(y)), y, right=0.0)
    # random latency + background noise
    lat = int(rng.integers(int(0.03 * fs), int(0.15 * fs)))
    y = np.concatenate([np.zeros(lat), y])[: len(x)]
    return (y + rng.normal(0, noise_rms, len(y))).astype(np.float32)
