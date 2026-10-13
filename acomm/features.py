"""Per-symbol features for the neural receiver (NRX).

For every data symbol of every detected frame we keep:
  win   : (2*EQ_HALF+1,) complex  T/2-spaced received samples around the symbol
          (after downconversion, matched filter, frame sync and CFO correction),
          normalized by the one-tap preamble channel estimate h1
  taps  : (2*EQ_HALF+1,) complex  the classical LS equalizer taps of that frame (channel summary)
  ls    : complex                 classical receiver output for that symbol (LS + LMS + PLL)
  tx    : int                     transmitted symbol index (ground truth)
plus per-frame bookkeeping (capture name, label, modulation, symbol rate, TX level, repeat).
"""
from pathlib import Path

import numpy as np

from . import io_sigmf, modem as M
from . import receiver as R


def _frame_features(fr, n0, sps, L):
    x, hstep, w = fr["x"], fr["hstep"], fr["w"]
    p = M.preamble()
    h1 = np.mean(x[n0 + np.arange(L) * sps] * np.conj(p))
    wins = []
    for i in range(M.N_DATA):
        u = R._regressor(x, n0 + (L + i) * sps, hstep)
        if u is None:
            return None
        wins.append(u)
    wins = np.array(wins) / h1
    taps = w * np.conj(h1) / abs(h1)  # undo the same phase so taps and windows share a frame of reference
    taps = taps / (np.linalg.norm(taps) + 1e-12)
    return wins.astype(np.complex64), taps.astype(np.complex64)


def extract(y, meta):
    """Return a list of per-frame dicts for one capture (y = real audio, meta = SigMF acomm fields)."""
    M.set_preamble_len(meta.get("preamble_len", 127))
    fs, fc, rs = meta["fs"], meta["fc"], meta["rs"]
    sps = fs // rs
    L = M.PREAMBLE_LEN
    mf = R.downconvert(np.asarray(y, float), fs, fc, sps)
    pk, _ = R.detect_frames(mf, sps, meta["n_frames"])
    frame_len = (L + M.N_DATA) * sps
    gap = int(meta.get("gap_s", 0.3) * fs)
    out = []
    for j, n0 in enumerate(pk):
        f_idx = j if len(pk) == meta["n_frames"] else int(round((n0 - pk[0]) / (frame_len + gap + sps * 2 * M.SPAN)))
        seed = meta["base_seed"] + f_idx
        fr = R.receive_frame(mf, n0, sps, meta["mod"], seed)
        if fr is None:
            continue
        ff = _frame_features(fr, n0, sps, L)
        if ff is None:
            continue
        wins, taps = ff
        _, tx_idx = M.decide(fr["tx_sym"], meta["mod"])
        out.append(dict(win=wins, taps=taps, ls=fr["eq"].astype(np.complex64), tx=tx_idx.astype(np.int16)))
    return out


def load_capture(meta_path, cache_dir=None):
    """Features for one SigMF capture, cached as .npz (extraction is the slow part)."""
    meta_path = Path(meta_path)
    y, meta = io_sigmf.load(meta_path)
    info = dict(name=meta_path.stem.replace(".sigmf", ""), label=meta["label"], mod=meta["mod"], rs=int(meta["rs"]),
                level_db=float(meta["level_db"]),
                repeat=int(meta_path.stem.split("_r")[1].split("_")[0]) if "_r" in meta_path.stem else 0)
    cache = Path(cache_dir) / f"{info['label']}__{info['name']}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        d = np.load(cache)
        frames = [dict(win=d["win"][k], taps=d["taps"][k], ls=d["ls"][k], tx=d["tx"][k]) for k in range(len(d["tx"]))]
        return info, frames
    frames = extract(y, meta)
    if cache is not None and frames:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, win=np.stack([f["win"] for f in frames]), taps=np.stack([f["taps"] for f in frames]),
                            ls=np.stack([f["ls"] for f in frames]), tx=np.stack([f["tx"] for f in frames]))
    return info, frames


def simulate_capture(mod, rs, level_db, rng, n_frames=5):
    """Generate one capture through the simulated room channel and extract its features (sim-to-real)."""
    from .channel_sim import room
    seed = int(rng.integers(0, 2 ** 31 - 1))
    x = M.build_capture(mod, 10 ** (level_db / 20), rs, n_frames, seed)
    noise = 10 ** rng.uniform(-3.3, -2.0)  # random SNR range
    y = room(x, M.FS, rng, noise_rms=noise, ppm=rng.uniform(-60, 60), drive=rng.uniform(0.5, 2.0),
             n_echo=int(rng.integers(2, 10)), max_delay_ms=rng.uniform(2, 40))
    meta = dict(fs=M.FS, fc=M.FC, rs=rs, n_frames=n_frames, gap_s=0.3, mod=mod, base_seed=seed)
    info = dict(name=f"sim_{seed}", label="sim", mod=mod, rs=rs, level_db=level_db, repeat=0)
    return info, extract(y, meta)
