"""Room geometry -> impulse response -> predicted error floor.

1. Impulse response (IR). From a recorded capture and the known transmitted audio, the
   acoustic channel is estimated as H(f) = S_xy(f) / S_xx(f) over the occupied band and turned
   into an impulse response. Use the highest symbol rate available (widest band = finest delay
   resolution: 2000 baud -> ~0.4 ms).

2. Predicted MER for any symbol rate. The IR is turned into the symbol-rate channel the receiver
   sees (RRC * IR * RRC, sampled at T/2). The ideal (MMSE) T/2 equalizer with the same number of
   taps as acomm/receiver.py is computed, using the noise measured in the silent parts of the
   recording. Its output SINR is the predicted MER. Two curves:
       noise + ISI : what the receiver should reach in that room
       ISI only    : the floor set by the room alone (no noise)

3. Comparison with what the receiver measured (metrics.csv from analyze.py).

Usage
  python geometry.py --labels canonical refl_none refl_17cm refl_34cm absorb_walls \
                     --results results_paper --expect refl_17cm=1.0 refl_34cm=2.0
  (--expect label=ms draws the delay you predicted from the tape measure on the IR plot)

Outputs in <results>/geometry/: ir_<label>.png, ir_overlay.png, floor_vs_baud.png, geometry.csv
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import csd, fftconvolve, welch

from acomm import io_sigmf, modem as M
from acomm import receiver as R

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]
C_SOUND = 343.0
IR_PRE_MS, IR_POST_MS = 3.0, 500.0  # long enough for the full reverberation tail (RT60 ~ 0.3 s)


# ----------------------------------------------------------------------------- IR estimation
def tx_audio(meta):
    M.set_preamble_len(meta.get("preamble_len", 127))
    lead = meta.get("lead_s", 0.4)
    return M.build_capture(meta["mod"], 10 ** (meta["level_db"] / 20), meta["rs"], meta["n_frames"],
                           meta["base_seed"], fc=meta["fc"], lead_s=lead, gap_s=meta.get("gap_s", 0.3)).astype(float)


def estimate_ir(y, meta, nper=32768):
    """Passband IR (real, 48 kHz) aligned so the strongest path is at t = 0."""
    fs = meta["fs"]
    x = tx_audio(meta)
    c = fftconvolve(y, x[::-1])
    lag = int(np.argmax(np.abs(c))) - (len(x) - 1)
    if lag < 0:
        return None
    ya = y[lag:lag + len(x)]
    if len(ya) < len(x):
        x = x[:len(ya)]
    f, Pxy = csd(x, ya, fs=fs, nperseg=nper)
    _, Pxx = welch(x, fs=fs, nperseg=nper)
    H = Pxy / (Pxx + 1e-30)
    band = Pxx > Pxx.max() * 10 ** (-25 / 10)
    win = np.zeros_like(Pxx)
    win[band] = 1.0
    # taper band edges to limit ringing
    from scipy.ndimage import uniform_filter1d
    win = uniform_filter1d(win, 9)
    h = np.fft.irfft(H * win, n=nper)
    pk = int(np.argmax(np.abs(h[: nper // 2])))
    h = np.roll(h, -pk + int(IR_PRE_MS * 1e-3 * fs))[: int((IR_PRE_MS + IR_POST_MS) * 1e-3 * fs)]
    lo, hi = f[band].min(), f[band].max()
    return h, (lo, hi), lag


def to_baseband(h_pb, fs, fc):
    n = np.arange(len(h_pb))
    return h_pb * np.exp(-1j * 2 * np.pi * fc * n / fs)


def envelope_db(h_pb, fs, fc, smooth_ms=0.25):
    hb = np.abs(to_baseband(h_pb, fs, fc)) ** 2
    k = max(1, int(smooth_ms * 1e-3 * fs))
    e = np.convolve(hb, np.ones(k) / k, mode="same")
    return 10 * np.log10(e / e.max() + 1e-9)


# ----------------------------------------------------------------------------- noise
def noise_samples(y, meta, lag=None):
    """Background noise: the lead-in silence before anything is played (gaps and tail contain
    the decaying reverberation of earlier frames, so they are not used)."""
    fs = meta["fs"]
    from scipy.signal import butter, sosfiltfilt
    a = int(0.02 * fs)
    # the whole silent lead-in: recording start -> (playback latency + lead_s) - 50 ms
    end_s = (lag / fs if lag is not None else 0.0) + meta.get("lead_s", 0.4) - 0.05
    b = max(a + int(0.1 * fs), int(end_s * fs))
    sos = butter(6, [1000, 5000], btype="bandpass", fs=fs, output="sos")  # drop room rumble far outside the band
    return sosfiltfilt(sos, y)[a:b]


def non_lti_floor_db(y, meta, nper=32768):
    """How well ANY linear time-invariant channel explains the recording: estimate the channel on
    one half of the capture, predict the other half, and measure the in-band residual
    (cross-validated, both ways). Whatever is left is time variation, nonlinear distortion or
    device processing - an error floor that no LTI equalizer can remove."""
    from scipy.signal import butter, sosfiltfilt
    fs = meta["fs"]
    x = tx_audio(meta)
    c = fftconvolve(y, x[::-1])
    lag = int(np.argmax(np.abs(c))) - (len(x) - 1)
    ya = y[lag:lag + len(x)]
    x = x[:len(ya)]
    half = len(x) // 2
    out = []
    for tr, te in [((0, half), (half, len(x))), ((half, len(x)), (0, half))]:
        f, Pxy = csd(x[tr[0]:tr[1]], ya[tr[0]:tr[1]], fs=fs, nperseg=nper)
        _, Pxx = welch(x[tr[0]:tr[1]], fs=fs, nperseg=nper)
        band = Pxx > Pxx.max() * 10 ** -2.5
        H = np.where(band, Pxy / (Pxx + 1e-30), 0)
        h = np.fft.irfft(H, n=nper)
        h2 = np.concatenate([h[nper // 2:], h[:nper // 2]])
        yh = fftconvolve(x, h2)[nper // 2: nper // 2 + len(x)]
        sos = butter(8, [f[band].min(), f[band].max()], btype="bandpass", fs=fs, output="sos")
        e = sosfiltfilt(sos, (ya - yh)[te[0]:te[1]])
        sg = sosfiltfilt(sos, ya[te[0]:te[1]])
        out.append(10 * np.log10(np.mean(sg ** 2) / np.mean(e ** 2)))
    return float(np.mean(out))


def symbol_scale(meta):
    """Amplitude of one unit symbol in the transmitted complex baseband (exact, per capture)."""
    M.set_preamble_len(meta.get("preamble_len", 127))
    sps = meta["fs"] // meta["rs"]
    h = M.rrc(sps)
    cs = []
    for f in range(meta["n_frames"]):
        sym = M.frame_symbols(meta["mod"], meta["base_seed"] + f)
        up = np.zeros(len(sym) * sps, complex); up[::sps] = sym
        bb = np.convolve(up, h)
        n = np.arange(len(bb))
        pb = np.real(bb * np.exp(1j * 2 * np.pi * meta["fc"] * n / meta["fs"]))
        cs.append(0.3 / np.sqrt(np.mean(pb ** 2)))
    return float(np.mean(cs)) * 10 ** (meta["level_db"] / 20)


# ----------------------------------------------------------------------------- prediction
def predict_mer(h_pb, noise_pb, rs, fs, fc, scale, half=None):
    """MMSE T/2 equalizer SINR for symbol rate rs. Returns (mer_noise_isi_db, mer_isi_only_db)."""
    half = R.EQ_HALF if half is None else half
    sps = fs // rs
    hs = sps // 2
    g_rrc = M.rrc(sps)
    hb = to_baseband(h_pb, fs, fc)
    g = fftconvolve(fftconvolve(g_rrc, hb), g_rrc) * scale
    d0 = int(np.argmax(np.abs(g[::1])))
    taps = (np.arange(2 * half + 1) - half) * hs
    m_lo = int(np.floor((taps[0] - (len(g) - d0)) / sps)) - 1
    m_hi = int(np.ceil((taps[-1] + d0) / sps)) + 1
    ms = np.arange(m_lo, m_hi + 1)
    idx = d0 + taps[:, None] - ms[None, :] * sps
    G = np.where((idx >= 0) & (idx < len(g)), g[np.clip(idx, 0, len(g) - 1)], 0)
    g0 = G[:, ms == 0][:, 0]
    Gi = G[:, ms != 0]
    # noise through the same matched filter, at T/2 spacing -> Toeplitz covariance
    nb = fftconvolve(2 * noise_pb * np.exp(-1j * 2 * np.pi * fc * np.arange(len(noise_pb)) / fs), g_rrc, mode="same")
    nb = nb[len(g_rrc): -len(g_rrc)]
    L = 2 * half + 1
    # Background noise is treated as white within the signal band: its power after the matched
    # filter is measured from the lead-in silence, and its T/2 correlation is the raised-cosine
    # shape (RRC autocorrelation), which keeps the covariance positive definite.
    L = 2 * half + 1
    sigma2 = np.mean(np.abs(nb) ** 2)
    rc = np.correlate(g_rrc, g_rrc, mode="full")
    c0 = len(g_rrc) - 1
    r = np.array([rc[c0 + k * hs] if c0 + k * hs < len(rc) else 0.0 for k in range(L)]) * sigma2 / rc[c0]
    Rn = toeplitz(r, np.conj(r))
    Risi = Gi @ Gi.conj().T
    out = []
    # second case: same room, background noise 30 dB lower -> the floor set by the room alone
    for Rd in (Risi + Rn, Risi + 1e-3 * Rn):
        sinr = np.real(g0.conj() @ np.linalg.solve(Rd, g0))
        out.append(10 * np.log10(1 + max(sinr, 0)))  # MMSE output MSE = 1/(1+SINR), comparable to measured MER
    return out[0], out[1]


# ----------------------------------------------------------------------------- main
def best_ir_capture(files):
    """Highest symbol rate, loudest level, QPSK preferred."""
    import json
    metas = []
    for f in files:
        g = json.load(open(f))["global"]
        metas.append((g["acomm:rs"], g["acomm:level_db"], g["acomm:mod"] == "QPSK", f))
    metas.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    top_rs, top_lv = metas[0][0], metas[0][1]
    return [t[3] for t in metas if t[0] == top_rs and t[1] == top_lv][:6]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--results", default="results_paper", help="folder with metrics.csv from analyze.py")
    ap.add_argument("--expect", nargs="*", default=[], help="label=delay_ms predicted from geometry")
    ap.add_argument("--rates", nargs="+", type=int, default=[250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000])
    ap.add_argument("--ir-xlim", type=float, default=40.0, help="x-axis limit of IR plots (ms)")
    ap.add_argument("--level", type=float, default=0.0, help="TX level (dBFS) used for the floor comparison")
    args = ap.parse_args()

    out = Path(args.results) / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    expect = {}
    for e in args.expect:
        k, v = e.split("=")
        expect.setdefault(k, []).append(float(v))
    metrics = None
    mp = Path(args.results) / "metrics.csv"
    if mp.exists():
        metrics = pd.read_csv(mp)

    rows, irs = [], {}
    for li, lab in enumerate(args.labels):
        files = sorted((Path(args.data) / lab).glob("*.sigmf-meta"))
        if not files:
            print(f"{lab}: no captures, skipped"); continue
        hs, bands, noise, metas, nls = [], [], [], [], []
        for f in best_ir_capture(files):
            y, meta = io_sigmf.load(f)
            r = estimate_ir(y.astype(float), meta)
            if r is None:
                continue
            hs.append(r[0]); bands.append(r[1]); noise.append(noise_samples(y.astype(float), meta, r[2])); metas.append(meta)

        if not hs:
            print(f"{lab}: IR estimation failed"); continue
        L = min(len(h) for h in hs)
        # average IRs coherently after aligning on the strongest path (already aligned)
        h = np.mean([x[:L] for x in hs], axis=0)
        meta = metas[0]
        fs, fc = meta["fs"], meta["fc"]
        noise_pb = np.concatenate(noise)
        import json as _json
        by_rate = {}
        for f in files:
            g_ = _json.load(open(f))["global"]
            if g_["acomm:level_db"] == args.level:
                by_rate.setdefault(int(g_["acomm:rs"]), []).append(f)
        irs[lab] = (h, fs, fc, bands[0], meta["rs"])
        env = envelope_db(h, fs, fc)
        t = (np.arange(len(h)) / fs - IR_PRE_MS * 1e-3) * 1e3
        p = 10 ** (env / 10); p = p / p.sum()
        mu = np.sum(t * p); spread = np.sqrt(np.sum((t - mu) ** 2 * p))

        # per-label IR figure
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.plot(t, env, color=PAL[li % len(PAL)], lw=1.2)
        for d in expect.get(lab, []):
            ax.axvline(d, color="k", ls="--", lw=1)
            ax.text(d, -2, f" predicted {d:g} ms", fontsize=8, va="top")
        ax.set_xlim(-IR_PRE_MS, args.ir_xlim); ax.set_ylim(-50, 1)
        ax.set_xlabel("Delay after direct path (ms)"); ax.set_ylabel("Energy (dB, smoothed 0.25 ms)")
        ax.set_title(f"Measured impulse response — {lab}  (band {bands[0][0]:.0f}–{bands[0][1]:.0f} Hz, "
                     f"from {meta['rs']} baud, RMS delay spread {spread:.1f} ms)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"ir_{lab}.png"); plt.close(fig)

        # MER predictions for each rate
        scale = symbol_scale(meta) * 10 ** ((args.level - meta["level_db"]) / 20)
        for rs in args.rates:
            if M.FS % rs or (M.FS // rs) % 2:
                continue
            occupied = rs * (1 + M.BETA) / 2
            valid = (fc - occupied >= bands[0][0] - 50) and (fc + occupied <= bands[0][1] + 50)
            pn, pi = predict_mer(h, noise_pb, rs, fs, fc, scale)
            nl_db = np.nan
            if rs in by_rate:
                vals = []
                for f in by_rate[rs][:3]:
                    yy, mm = io_sigmf.load(f)
                    vals.append(non_lti_floor_db(yy.astype(float), mm))
                nl_db = float(np.mean(vals))
            tot = -10 * np.log10(10 ** (-pn / 10) + 10 ** (-nl_db / 10)) if np.isfinite(nl_db) else np.nan
            row = dict(label=lab, rs=rs, pred_total_db=tot, pred_mer_db=pn, pred_isi_floor_db=pi,
                       non_lti_floor_db=nl_db, ir_band_valid=valid,
                       rms_delay_spread_ms=spread, ir_from_rs=meta["rs"])
            if metrics is not None:
                d = metrics[(metrics["label"] == lab) & (metrics["rs"] == rs) & (metrics["level_db"] == args.level)]
                if len(d):
                    row.update(meas_mer_db=d["mer_db"].mean(), meas_mer_std=d["mer_db"].std(), n_meas=len(d))
            rows.append(row)
        print(f"{lab}: IR from {len(hs)} captures at {meta['rs']} baud, RMS delay spread {spread:.1f} ms")

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.round(3).to_csv(out / "geometry.csv", index=False)

    # overlay of IRs
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, (lab, (h, fs, fc, band, rs)) in enumerate(irs.items()):
        t = (np.arange(len(h)) / fs - IR_PRE_MS * 1e-3) * 1e3
        ax.plot(t, envelope_db(h, fs, fc), color=PAL[i % len(PAL)], lw=1.1, label=lab)
    ax.set_xlim(-IR_PRE_MS, args.ir_xlim); ax.set_ylim(-50, 1)
    ax.set_xlabel("Delay after direct path (ms)"); ax.set_ylabel("Energy (dB)")
    ax.set_title("Measured impulse responses", fontweight="bold"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ir_overlay.png"); plt.close(fig)

    # predicted vs measured floor
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, lab in enumerate(df["label"].unique()):
        d = df[df["label"] == lab].sort_values("rs")
        col = PAL[i % len(PAL)]
        dt = d[d.pred_total_db.notna()]
        ax.plot(dt.rs, dt.pred_total_db, "^-", color=col, lw=1.8, ms=5, label=f"{lab}: predicted (noise + ISI + measured non-LTI)")
        ax.plot(d.rs, d.pred_mer_db, "--", color=col, lw=1.0, label=f"{lab}: predicted, LTI channel only")
        ax.plot(d.rs, d.pred_isi_floor_db, ":", color=col, lw=1.0, label=f"{lab}: room-only floor (noise -30 dB)")
        if "meas_mer_db" in d and d["meas_mer_db"].notna().any():
            ax.errorbar(d.rs, d.meas_mer_db, yerr=d.meas_mer_std, fmt="o", color=col, ms=5, capsize=3,
                        label=f"{lab}: measured")
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log"); ax.set_xticks([250, 500, 1000, 2000]); ax.set_xticklabels(["250", "500", "1000", "2000"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Symbol rate (baud)"); ax.set_ylabel(f"MER (dB) at TX {args.level:g} dBFS")
    ax.set_title("Error floor predicted from the measured impulse response vs measured", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "floor_vs_baud.png"); plt.close(fig)
    print(df.round(2).to_string(index=False))
    print(f"\nFigures -> {out}")


if __name__ == "__main__":
    main()
