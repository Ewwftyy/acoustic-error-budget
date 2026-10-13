"""Validation of the error-budget decomposition on synthetic data with known ground truth.

Part A (symbol level): inject each impairment alone at several strengths, and all together,
  and compare the power the decomposition attributes to it with the injected power.
Part B (waveform level, full receiver): send frames through the simulated room channel and
  check (1) the noise estimate from the silent gaps against the noise-only MER, and
  (2) that adding an echo moves error into "ISI" while the noise share stays put.

No hardware needed. Outputs in results_validation/: validation_symbol_level.png,
validation_waveform.png, validation.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acomm import modem as M
from acomm.analysis import COMPONENTS, decompose, capture_metrics
from acomm.receiver import receive_capture

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
NAMES = dict(rotation_gain="Rotation / gain", iq_imbalance="I/Q imbalance", dc_offset="DC offset",
             compression="Compression", phase_jitter="Phase jitter", amplitude_jitter="Amplitude jitter",
             noise="Noise", isi_distortion="ISI / distortion")


def estimate(y, s, mod, noise_power):
    parts, _ = decompose(y, s, mod)
    iso = parts.pop("isotropic")
    parts["noise"] = min(noise_power, iso)
    parts["isi_distortion"] = iso - parts["noise"]
    return parts


def impair(s, rng, kind, k):
    """Return (y, injected_component_power)."""
    n = len(s)
    if kind == "rotation_gain":
        a = (1 + k) * np.exp(1j * k)
        return a * s, abs(a - 1) ** 2 * np.mean(abs(s) ** 2)
    if kind == "iq_imbalance":
        b = k * np.exp(0.7j)
        return s + b * np.conj(s), abs(b) ** 2 * np.mean(abs(s) ** 2)
    if kind == "dc_offset":
        c = k * np.exp(1.1j)
        return s + c, abs(c) ** 2
    if kind == "compression":
        e = -k * (abs(s) ** 2 - np.mean(abs(s) ** 2)) * s
        return s + e, np.mean(abs(e) ** 2)
    if kind == "phase_jitter":
        e = s * (np.exp(1j * rng.normal(0, k, n)) - 1)
        return s + e, np.mean(abs(e) ** 2)
    if kind == "amplitude_jitter":
        e = s * rng.normal(0, k, n)
        return s + e, np.mean(abs(e) ** 2)
    if kind == "noise":
        e = (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) * k / np.sqrt(2)
        return s + e, np.mean(abs(e) ** 2)
    if kind == "isi_distortion":
        h = np.array([0, k, 0.6 * k * 1j, -0.3 * k])  # residual post-cursor ISI
        e = np.convolve(s, h)[:n]
        return s + e, np.mean(abs(e) ** 2)
    raise ValueError(kind)


def part_a(rng, mod="16QAM", n=40000):
    s = M.constellation(mod)[rng.integers(0, 2 ** M.bits_per_symbol(mod), n)]
    strengths = dict(rotation_gain=[0.01, 0.03, 0.1, 0.2], iq_imbalance=[0.01, 0.03, 0.1, 0.2],
                     dc_offset=[0.01, 0.03, 0.1, 0.2], compression=[0.01, 0.03, 0.08, 0.15],
                     phase_jitter=[0.01, 0.03, 0.08, 0.15], amplitude_jitter=[0.01, 0.03, 0.08, 0.15],
                     noise=[0.01, 0.03, 0.1, 0.3], isi_distortion=[0.01, 0.03, 0.1, 0.25])
    rows = []
    # each impairment alone, with a small noise floor that the method is told about
    for kind, ks in strengths.items():
        for k in ks:
            y, true = impair(s, rng, kind, k)
            floor = 1e-4
            y = y + (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) * np.sqrt(floor / 2)
            est = estimate(y, s, mod, floor if kind != "noise" else true + floor)
            rows.append(dict(test="alone", component=kind, strength=k, true=true + (floor if kind == "noise" else 0),
                             est=est[kind], leak=sum(v for c, v in est.items() if c not in (kind, "noise"))))
    # all impairments together
    for trial in range(5):
        y, trues = s.copy(), {}
        for kind in ["isi_distortion", "compression", "phase_jitter", "amplitude_jitter", "rotation_gain",
                     "iq_imbalance", "dc_offset", "noise"]:
            k = rng.choice(strengths[kind][:3])
            e_y, t = impair(s, rng, kind, k)
            y = y + (e_y - s)
            trues[kind] = t
        est = estimate(y, s, mod, trues["noise"])
        for kind in trues:
            rows.append(dict(test="combined", component=kind, strength=np.nan, true=trues[kind], est=est[kind], leak=np.nan))
    return pd.DataFrame(rows)


def part_b(rng):
    from acomm.channel_sim import room
    rows = []
    base = dict(fs=M.FS, fc=M.FC, n_frames=3, gap_s=0.3)

    def run(mod, rs, noise, echo_amp, echo_ms):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        x = M.build_capture(mod, 1.0, rs, 3, seed)
        y = room(x, M.FS, np.random.default_rng(1), noise_rms=noise, ppm=0, drive=1e-3, n_echo=0)
        if echo_amp:
            d = int(echo_ms * 1e-3 * M.FS)
            y = y + echo_amp * np.concatenate([np.zeros(d), y[:-d]])
        meta = dict(base, rs=rs, mod=mod, base_seed=seed)
        return capture_metrics(receive_capture(y, meta), mod)

    for noise in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
        m = run("QPSK", 1000, noise, 0, 0)
        rows.append(dict(test="noise_sweep", noise_rms=noise, echo=0, snr_est=m["snr_db"], mer=m["mer_db"],
                         frac_noise=m["frac_noise"], frac_isi=m["frac_isi_distortion"],
                         pow_noise=m["pow_noise"], pow_isi=m["pow_isi_distortion"]))
    for ems in [1.0, 30.0]:  # 1 ms: inside the equalizer span; 30 ms: beyond it (+/-24 symbols at 1000 baud)
        for amp in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]:
            m = run("QPSK", 1000, 0.03, amp, ems)
            rows.append(dict(test="echo_sweep", echo_ms=ems, noise_rms=0.03, echo=amp, snr_est=m["snr_db"],
                             mer=m["mer_db"], frac_noise=m["frac_noise"], frac_isi=m["frac_isi_distortion"],
                             pow_noise=m["pow_noise"], pow_isi=m["pow_isi_distortion"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    a = part_a(rng)
    b = part_b(rng)
    pd.concat([a.assign(part="A"), b.assign(part="B")]).to_csv(out / "validation.csv", index=False)

    # Figure A: estimated vs injected power, log-log
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, comp in zip(axes.flat, COMPONENTS):
        d = a[a.component == comp]
        al, co = d[d.test == "alone"], d[d.test == "combined"]
        lo = min(d.true.min(), d.est.clip(lower=1e-7).min()) / 2
        hi = max(d.true.max(), d.est.max()) * 2
        ax.loglog([lo, hi], [lo, hi], "k--", lw=1)
        ax.loglog(al.true, al.est.clip(lower=1e-7), "o", color="#1f77b4", ms=6, label="injected alone")
        ax.loglog(co.true, co.est.clip(lower=1e-7), "s", color="#d62728", ms=5, mfc="none", label="all combined")
        ax.set_title(NAMES[comp]); ax.set_xlabel("Injected power"); ax.set_ylabel("Estimated power")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Error-budget decomposition: estimated vs injected power (16QAM, 40k symbols; dashed = perfect)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "validation_symbol_level.png"); plt.close(fig)

    # Figure B: waveform level
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    n = b[b.test == "noise_sweep"]
    ax = axes[0]
    lim = [min(n.snr_est.min(), n.mer.min()) - 2, max(n.snr_est.max(), n.mer.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, label="MER = SNR")
    ax.plot(n.snr_est, n.mer, "o-", color="#1f77b4", label="noise-only channel")
    for _, r in n.iterrows():
        ax.annotate(f"{100 * r.frac_noise:.0f}% noise", (r.snr_est, r.mer), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_xlabel("SNR estimated from silent gaps (dB)"); ax.set_ylabel("Measured MER (dB)")
    ax.set_title("Noise estimator: a noise-only link should sit on the line"); ax.legend(fontsize=8)
    ax = axes[1]
    for ems, ls in [(1.0, "--"), (30.0, "-")]:
        e = b[(b.test == "echo_sweep") & (b.echo_ms == ems)]
        ax.plot(e.echo, 10 * np.log10(e.pow_noise), "o" + ls, color="#55a868", ms=4, label=f"Noise, echo at {ems:g} ms")
        ax.plot(e.echo, 10 * np.log10(e.pow_isi.clip(lower=1e-6)), "s" + ls, color="#ccb974", ms=4, label=f"ISI, echo at {ems:g} ms")
    aa = np.linspace(0.03, 0.45, 50)
    ax.plot(aa, 20 * np.log10(aa), "k:", lw=1, label="injected echo power a$^2$")
    ax.set_xlabel("Echo amplitude (QPSK, 1000 baud)"); ax.set_ylabel("Estimated power (dB rel. symbol)")
    ax.set_title("Echo beyond the equalizer span (30 ms) shows up as ISI;\nnoise estimate is unchanged", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle("Error-budget decomposition through the full receiver (simulated room)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "validation_waveform.png"); plt.close(fig)

    al = a[a.test == "alone"]
    ok = al[al.true > 1e-3]
    err_db = 10 * np.log10(ok.est.clip(lower=1e-9) / ok.true)
    print("Part A, each impairment alone (true power > 1e-3):")
    print(ok.assign(err_db=err_db).groupby("component")["err_db"].agg(["mean", "max"]).round(2).to_string())
    co = a[a.test == "combined"]
    cerr = 10 * np.log10(co.est.clip(lower=1e-9) / co.true)
    print("\nPart A, all combined: mean |error| per component (dB)")
    print(co.assign(err_db=cerr.abs()).groupby("component")["err_db"].mean().round(2).to_string())
    print("\nPart B:"); print(b.round(3).to_string(index=False))
    print(f"\nFigures -> {out}")


if __name__ == "__main__":
    main()
