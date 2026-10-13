"""Receiver chain: downconvert -> matched filter -> frame sync -> CFO/one-tap -> FSE + PLL."""
import numpy as np
from scipy.signal import fftconvolve, find_peaks
from . import modem as M

EQ_HALF = 48  # equalizer reaches 48 half-symbols (24 symbols) each side


def downconvert(y, fs, fc, sps):
    n = np.arange(len(y))
    bb = 2.0 * y * np.exp(-1j * 2 * np.pi * fc * n / fs)
    return fftconvolve(bb, M.rrc(sps), mode="same")


def detect_frames(mf, sps, n_frames, thresh=0.35):
    p = M.preamble()
    L = len(p)
    comb = np.zeros((L - 1) * sps + 1, complex)
    comb[::sps] = p
    corr = fftconvolve(mf, np.conj(comb[::-1]), mode="valid")
    ecomb = np.zeros_like(comb.real)
    ecomb[::sps] = 1
    energy = fftconvolve(np.abs(mf) ** 2, ecomb, mode="valid")
    metric = np.abs(corr) / np.sqrt(np.maximum(energy, 1e-20) * L)
    frame_len = (L + M.N_DATA) * sps
    pk, prop = find_peaks(metric, height=thresh, distance=int(0.8 * frame_len))
    if len(pk) > n_frames:
        keep = np.argsort(prop["peak_heights"])[::-1][:n_frames]
        pk = np.sort(pk[keep])
    return pk, metric


def _regressor(x, center, h, half=None):
    half = EQ_HALF if half is None else half
    idx = center + (np.arange(2 * half + 1) - half) * h
    if idx[0] < 0 or idx[-1] >= len(x):
        return None
    return x[idx]


def receive_frame(mf, n0, sps, mod, seed, mu_train=0.01, mu_dd=0.003, kp=0.01, ki=1e-4):
    p = M.preamble()
    L, N = len(p), M.N_DATA
    k = np.arange(L + N)
    sidx = n0 + k * sps
    if sidx[-1] + EQ_HALF * sps // 2 >= len(mf) or n0 - EQ_HALF * sps // 2 < 0:
        return None
    raw = mf[sidx]

    # CFO from preamble halves + one-tap channel
    z = raw[:L] * np.conj(p)
    hL = L // 2
    w_cfo = np.angle(np.sum(z[hL:2 * hL] * np.conj(z[:hL]))) / hL
    rot = np.exp(-1j * w_cfo * k)
    h1 = np.mean(raw[:L] * rot[:L] * np.conj(p))
    coarse = raw * rot / h1

    # Fractionally spaced (T/2) equalizer: least-squares fit on the preamble,
    # refining the CFO from the residual phase slope of the equalized preamble.
    n = np.arange(len(mf))
    hstep = sps // 2
    ntap = 2 * EQ_HALF + 1
    for _ in range(3):
        x = mf * np.exp(-1j * w_cfo * (n - n0) / sps)
        U = np.array([_regressor(x, n0 + i * sps, hstep) for i in range(L)])
        G = U.conj().T @ U
        w = np.linalg.solve(G + 1e-3 * np.trace(G).real / ntap * np.eye(ntap), U.conj().T @ p).conj()
        ph = np.unwrap(np.angle((U @ w.conj()) * np.conj(p)))
        w_cfo += np.polyfit(np.arange(L), ph, 1)[0]
    x = mf * np.exp(-1j * w_cfo * (n - n0) / sps)
    U = np.array([_regressor(x, n0 + i * sps, hstep) for i in range(L)])
    G = U.conj().T @ U
    w = np.linalg.solve(G + 1e-3 * np.trace(G).real / ntap * np.eye(ntap), U.conj().T @ p).conj()
    theta, integ = 0.0, 0.0

    def step(u, d_known, mu):
        nonlocal w, theta, integ
        y = np.vdot(w, u) * np.exp(-1j * theta)
        d = d_known if d_known is not None else M.decide(np.array([y]), mod)[0][0]
        e = d - y
        w = w + mu * u * np.conj(e * np.exp(1j * theta)) / (np.vdot(u, u).real + 1e-12)
        ph = np.angle(y * np.conj(d))
        integ += ki * ph
        theta += kp * ph + integ
        return y

    rot = np.exp(-1j * w_cfo * k)
    h1 = np.mean(raw[:L] * rot[:L] * np.conj(p))
    coarse = raw * rot / h1

    for i in range(L):  # short LMS polish on the preamble, then decision-directed
        step(U[i], p[i], mu_train)
    eq = np.empty(N, complex)
    for i in range(N):
        eq[i] = step(_regressor(x, n0 + (L + i) * sps, hstep), None, mu_dd)

    tx_bits = M.payload_bits(mod, seed)
    tx_sym, _ = M.bits_to_symbols(tx_bits, mod)
    return dict(raw=raw[L:], coarse=coarse[L:], eq=eq, tx_sym=tx_sym, tx_bits=tx_bits,
                w=w, x=x, hstep=hstep, cfo_hz_per_sym=w_cfo)


def noise_through_equalizer(x, w, start, stop, hstep):
    """Output power of the final equalizer applied to a signal-free gap."""
    out = []
    for c in range(start, stop, hstep):
        u = _regressor(x, c, hstep)
        if u is not None:
            out.append(np.vdot(w, u))
    out = np.array(out)
    return np.mean(np.abs(out) ** 2) if len(out) > 8 else np.nan


def receive_capture(y, meta):
    M.set_preamble_len(meta.get("preamble_len", 127))
    fs, fc, rs = meta["fs"], meta["fc"], meta["rs"]
    sps = fs // rs
    mf = downconvert(y.astype(float), fs, fc, sps)
    pk, _ = detect_frames(mf, sps, meta["n_frames"])
    frames = []
    frame_len = (M.PREAMBLE_LEN + M.N_DATA) * sps
    gap = int(meta["gap_s"] * fs)
    for j, n0 in enumerate(pk):
        # which transmitted frame is this? nearest nominal position relative to the first peak
        f_idx = j if len(pk) == meta["n_frames"] else int(round((n0 - pk[0]) / (frame_len + gap + sps * 2 * M.SPAN)))
        r = receive_frame(mf, n0, sps, meta["mod"], meta["base_seed"] + f_idx)
        if r is None:
            continue
        # Background noise: the lead-in silence before the first frame (nothing has been played yet).
        # The inter-frame gaps are NOT used for noise: they still contain the decaying room
        # reverberation of the previous frame (measured -26..-36 dB at 80-250 ms), which is
        # signal energy, not noise. That gap energy is kept separately as "gap_out".
        l0 = int(0.02 * fs)
        l1 = pk[0] - M.SPAN * sps - int(0.03 * fs)
        r["noise_out"] = noise_through_equalizer(r["x"], r["w"], l0, l1, r["hstep"])
        g0 = n0 + frame_len + int(0.08 * fs)
        g1 = n0 + frame_len + gap - int(0.05 * fs)
        r["gap_out"] = noise_through_equalizer(r["x"], r["w"], g0, g1, r["hstep"])
        del r["x"]
        frames.append(r)
    return frames
