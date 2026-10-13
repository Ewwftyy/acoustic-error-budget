"""Transmitter side: constellations, pulse shaping, frame and capture waveforms."""
import numpy as np

FS = 48000          # audio sample rate (Hz)
FC = 3000.0         # audio carrier (Hz)
BETA = 0.35         # RRC roll-off
SPAN = 8            # RRC span (symbols each side)
PREAMBLE_LEN = 127  # m-sequence length
N_DATA = 600        # data symbols per frame
MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]


def gray(n):
    return n ^ (n >> 1)


def bits_per_symbol(mod):
    return {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6}[mod]


def constellation(mod):
    """Gray-mapped table: table[symbol_index] -> complex point, unit average power."""
    k = bits_per_symbol(mod)
    M = 2 ** k
    table = np.zeros(M, complex)
    if mod == "BPSK":
        table[:] = [-1, 1]
    elif mod.endswith("PSK"):
        off = np.pi / 4 if M == 4 else 0.0
        for i in range(M):
            table[gray(i)] = np.exp(1j * (2 * np.pi * i / M + off))
    else:
        m = int(np.sqrt(M))
        h = k // 2
        lv = 2 * np.arange(m) - (m - 1)
        for ii in range(m):
            for qq in range(m):
                table[(gray(ii) << h) | gray(qq)] = lv[ii] + 1j * lv[qq]
    return table / np.sqrt(np.mean(np.abs(table) ** 2))


def bits_to_symbols(bits, mod):
    k = bits_per_symbol(mod)
    b = bits.reshape(-1, k)
    idx = b.dot(1 << np.arange(k - 1, -1, -1))
    return constellation(mod)[idx], idx


def symbols_to_bits(idx, mod):
    k = bits_per_symbol(mod)
    return ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def decide(y, mod):
    table = constellation(mod)
    idx = np.argmin(np.abs(y[:, None] - table[None, :]), axis=1)
    return table[idx], idx


def payload_bits(mod, seed, n_sym=N_DATA):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_sym * bits_per_symbol(mod), dtype=np.uint8)


# primitive polynomials (exponents of the taps) for maximal-length sequences of length 2^n - 1
_MSEQ_TAPS = {7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7), 11: (11, 9)}


def set_preamble_len(L):
    """Select the preamble (m-sequence) length: 127 (default), 255, 511, 1023 or 2047 symbols.
    Every capture stores its preamble_len in the SigMF metadata."""
    global PREAMBLE_LEN
    L = int(L)
    if L + 1 not in [2 ** n for n in _MSEQ_TAPS]:
        raise ValueError(f"preamble length must be one of {[2 ** n - 1 for n in _MSEQ_TAPS]}")
    PREAMBLE_LEN = L


def preamble():
    """Maximal-length (m-)sequence of PREAMBLE_LEN symbols mapped to BPSK (127: x^7 + x^6 + 1)."""
    n = int(np.log2(PREAMBLE_LEN + 1))
    taps = _MSEQ_TAPS[n]
    reg = [1] * n
    out = []
    for _ in range(PREAMBLE_LEN):
        out.append(reg[-1])
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
    return 1.0 - 2.0 * np.array(out, float) + 0j


def rrc(sps, beta=BETA, span=SPAN):
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / \
                   (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def frame_symbols(mod, seed):
    data, idx = bits_to_symbols(payload_bits(mod, seed), mod)
    return np.concatenate([preamble(), data])


def build_capture(mod, level, rs, n_frames, base_seed, fs=FS, fc=FC,
                  lead_s=0.4, gap_s=0.3, tail_s=0.6):
    """Real passband audio for one capture: silence | frame | gap | frame ... | tail."""
    sps = fs // rs
    h = rrc(sps)
    parts = [np.zeros(int(lead_s * fs))]
    for f in range(n_frames):
        sym = frame_symbols(mod, base_seed + f)
        up = np.zeros(len(sym) * sps, complex)
        up[::sps] = sym
        bb = np.convolve(up, h)
        n = np.arange(len(bb))
        pb = np.real(bb * np.exp(1j * 2 * np.pi * fc * n / fs))
        pb *= 0.3 / np.sqrt(np.mean(pb ** 2))  # fixed RMS for every modulation
        parts.append(pb)
        parts.append(np.zeros(int(gap_s * fs)))
    parts.append(np.zeros(int(tail_s * fs)))
    x = np.concatenate(parts) * level
    return np.clip(x, -1, 1).astype(np.float32)

