"""Compare recording placements (labels) side by side.

Outputs (in --out, default results2/compare):
  mer_vs_level.png       MER vs TX level, one panel per symbol rate, one line per placement
  ser_vs_snr.png         QPSK / 16QAM SER vs noise-floor SNR per placement
  error_budget.png       error-power shares per placement at the two loudest TX levels
  impulse_response.png   room impulse response (energy vs delay) per placement
  summary.csv            per placement x symbol rate: MER ceiling, SNR, ISI share, RMS delay spread
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acomm import io_sigmf, modem as M
from acomm.analysis import COMPONENTS
from acomm.plots import COMP_COLORS, COMP_NAMES
from acomm.receiver import downconvert, detect_frames

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def impulse_response(meta_path, max_ms=80):
    """Symbol-spaced channel estimate from a whole known frame (least squares)."""
    y, meta = io_sigmf.load(meta_path)
    rs, fs = meta["rs"], meta["fs"]
    sps = fs // rs
    mf = downconvert(y.astype(float), fs, meta["fc"], sps)
    pk, _ = detect_frames(mf, sps, meta["n_frames"])
    if len(pk) == 0:
        return None
    taps = np.arange(-3, int(max_ms * 1e-3 * rs) + 1)
    acc = np.zeros(len(taps))
    used = 0
    for j, n0 in enumerate(pk[: meta["n_frames"]]):
        sym = M.frame_symbols(meta["mod"], meta["base_seed"] + j)
        L = len(sym)
        if n0 + (L - 1) * sps >= len(mf):
            continue
        X = np.zeros((L, len(taps)), complex)
        for c, d in enumerate(taps):
            if d >= 0:
                X[d:, c] = sym[: L - d]
            else:
                X[:d, c] = sym[-d:]
        r = mf[n0 + np.arange(L) * sps]
        h = np.linalg.lstsq(X, r, rcond=None)[0]
        acc += np.abs(h) ** 2
        used += 1
    if not used:
        return None
    p = acc / acc.sum()
    return taps / rs * 1e3, p  # delay in ms, normalized power


def delay_spread_ms(t, p):
    mu = np.sum(t * p)
    return float(np.sqrt(np.sum((t - mu) ** 2 * p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results2")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", nargs="*", default=["sim_room", "lab_0.5m"])
    args = ap.parse_args()
    out = Path(args.out or Path(args.results) / "compare")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.results) / "metrics.csv")
    df = df[~df["label"].isin(args.exclude)]
    labels = sorted(df["label"].unique())
    col = {lab: PAL[i % len(PAL)] for i, lab in enumerate(labels)}
    rates = sorted(df["rs"].unique())

    # 1) MER vs TX level
    fig, axes = plt.subplots(1, len(rates), figsize=(4.6 * len(rates), 4.2), sharey=True, squeeze=False)
    for ax, rs in zip(axes[0], rates):
        for lab in labels:
            d = df[(df["label"] == lab) & (df["rs"] == rs)].groupby("level_db")["mer_db"].agg(["mean", "std"])
            if len(d):
                ax.errorbar(d.index, d["mean"], yerr=d["std"], marker="o", ms=4, capsize=2, color=col[lab], label=lab)
        ax.set_xlabel("TX level (dBFS)"); ax.set_title(f"{rs} baud")
    axes[0][0].set_ylabel("MER (dB), mean ± std over modulations/repeats")
    axes[0][-1].legend()
    fig.suptitle("MER vs transmit level — bending below the 1 dB-per-dB slope = reverb (ISI) limit", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "mer_vs_level.png"); plt.close(fig)

    # 2) SER vs SNR for QPSK and 16QAM
    from acomm.analysis import theory_ser
    mods = [m for m in ["QPSK", "16QAM"] if m in set(df["mod"])]
    fig, axes = plt.subplots(1, len(mods), figsize=(5.2 * len(mods), 4.4), squeeze=False)
    g = np.linspace(-2, 25, 200)
    for ax, mod in zip(axes[0], mods):
        ax.semilogy(g, theory_ser(mod, 10 ** (g / 10)), "k--", lw=1, label="AWGN theory")
        for lab in labels:
            d = df[(df["label"] == lab) & (df["mod"] == mod)]
            floor = 1 / (d["n_bits"] / M.bits_per_symbol(mod))
            ax.semilogy(d["snr_db"], np.maximum(d["ser"], floor), "o", ms=4, alpha=0.7, color=col[lab], label=lab)
        ax.set_ylim(1e-4, 1); ax.set_xlim(-2, 25)
        ax.set_xlabel("SNR from noise floor (dB)"); ax.set_ylabel("Symbol error rate"); ax.set_title(mod)
        ax.legend(fontsize=8)
    fig.suptitle("Gap to the AWGN curve = error caused by the room, not by noise", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "ser_vs_snr.png"); plt.close(fig)

    # 3) Error budget at the two loudest levels
    top = sorted(df["level_db"].unique())[-2:]
    d = df[df["level_db"].isin(top)].groupby(["label", "rs"])[["frac_" + c for c in COMPONENTS]].mean()
    fig, ax = plt.subplots(figsize=(1.0 * len(d) + 3, 4.8))
    x = np.arange(len(d)); bottom = np.zeros(len(d))
    for c, cc, name in zip(COMPONENTS, COMP_COLORS, COMP_NAMES):
        v = d["frac_" + c].values * 100
        ax.bar(x, v, bottom=bottom, color=cc, label=name, width=0.75); bottom += v
    ax.set_xticks(x); ax.set_xticklabels([f"{a}\n{b} bd" for a, b in d.index], fontsize=8)
    ax.set_ylabel("Share of error power (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"Error budget at TX level {top[0]:.0f} and {top[1]:.0f} dBFS", fontweight="bold")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); fig.savefig(out / "error_budget.png", bbox_inches="tight"); plt.close(fig)

    # 4) Impulse response from loud QPSK/BPSK captures
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for lab in labels:
        files = sorted((Path(args.data) / lab).glob("*.sigmf-meta"))
        cir = []
        for f in files:
            n = f.stem
            if ("_QPSK_" in n or "_BPSK_" in n) and n.endswith(("_0dB", "_3dB", "_6dB")) and "_250bd_" in n:
                r = impulse_response(f)
                if r is not None:
                    cir.append(r)
        if not cir:
            continue
        t = cir[0][0]
        p = np.mean([c[1] for c in cir], axis=0)
        ax.plot(t, 10 * np.log10(p + 1e-6), color=col[lab], lw=1.5, label=f"{lab} (spread {delay_spread_ms(t, p):.1f} ms)")
        rows.append(dict(label=lab, rms_delay_spread_ms=delay_spread_ms(t, p),
                         direct_path_energy_pct=100 * p.max(), n_cir=len(cir)))
    ax.set_xlabel("Delay (ms)"); ax.set_ylabel("Normalized energy (dB)"); ax.set_ylim(-40, 0)
    ax.set_title("Room impulse response (250 baud, loudest QPSK/BPSK captures)", fontweight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "impulse_response.png"); plt.close(fig)

    # summary table
    hi = df[df["level_db"].isin(top)]
    s = hi.groupby(["label", "rs"]).agg(mer_ceiling_db=("mer_db", "mean"), snr_db=("snr_db", "mean"),
                                         isi_share=("frac_isi_distortion", "mean"),
                                         noise_share=("frac_noise", "mean")).reset_index()
    if rows:
        s = s.merge(pd.DataFrame(rows), on="label", how="left")
    s.round(3).to_csv(out / "summary.csv", index=False)
    print(s.round(2).to_string(index=False))
    print(f"\nFigures -> {out}")


if __name__ == "__main__":
    main()
