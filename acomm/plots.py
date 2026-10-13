"""Clean, labeled figures (saved to PNG, never shown)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from . import modem as M
from .analysis import COMPONENTS, theory_ser

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
COLORS = {"BPSK": "#1f77b4", "QPSK": "#ff7f0e", "8PSK": "#2ca02c", "16QAM": "#d62728", "64QAM": "#9467bd"}
COMP_COLORS = ["#4c72b0", "#dd8452", "#937860", "#c44e52", "#8172b3", "#da8bc3", "#55a868", "#ccb974"]
COMP_NAMES = ["Rotation / gain", "I/Q imbalance", "DC offset", "Compression",
              "Phase jitter", "Amplitude jitter", "Noise", "ISI / distortion"]


def _const_ax(ax, pts, mod, title, lim=None):
    ideal = M.constellation(mod)
    ax.scatter(pts.real, pts.imag, s=3, alpha=0.35, color="#1f4e79", edgecolors="none")
    ax.scatter(ideal.real, ideal.imag, marker="x", s=40, color="#c00000", label="Ideal")
    lim = lim or 1.6 * np.max(np.abs(ideal))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase (I)"); ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)


def capture_figure(frames, mod, m, title, path):
    raw = np.concatenate([f["raw"] for f in frames])
    raw = raw / np.sqrt(np.mean(np.abs(raw) ** 2))
    coarse = np.concatenate([f["coarse"] for f in frames])
    eq = np.concatenate([f["eq"] for f in frames])
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9])
    lim = 1.6 * np.max(np.abs(M.constellation(mod)))
    _const_ax(fig.add_subplot(gs[0]), raw, mod, "1) Matched-filter output\n(no correction)", max(lim, 2.2))
    _const_ax(fig.add_subplot(gs[1]), coarse, mod, "2) After CFO + one-tap\nchannel correction", lim)
    _const_ax(fig.add_subplot(gs[2]), eq, mod, "3) After equalizer + PLL\n(final decisions)", lim)
    ax = fig.add_subplot(gs[3])
    vals = [m["frac_" + c] * 100 for c in COMPONENTS]
    ax.barh(COMP_NAMES[::-1], vals[::-1], color=COMP_COLORS[::-1])
    ax.set_xlabel("Share of error power (%)")
    ax.set_title("Error budget")
    ax.set_xlim(0, 100)
    txt = (f"EVM {m['evm_pct']:.1f}%   MER {m['mer_db']:.1f} dB\n"
           f"SNR(noise floor) {m['snr_db']:.1f} dB\n"
           f"SER {m['ser']:.2e}   BER {m['ber']:.2e}")
    ax.text(0.5, -0.22, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_figures(df, outdir):
    labels = sorted(df["label"].unique())
    markers = dict(zip(labels, ["o", "s", "^", "D", "v", "P", "X", "*"] * 4))

    # MER vs SNR: how far real impairments push you below the noise-only limit
    fig, ax = plt.subplots(figsize=(7, 5.5))
    lo, hi = np.nanmin(df["snr_db"]) - 2, np.nanmax(df["snr_db"]) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="MER = SNR (noise only)")
    for mod in M.MODS:
        for lab in labels:
            d = df[(df["mod"] == mod) & (df["label"] == lab)]
            if len(d):
                ax.scatter(d["snr_db"], d["mer_db"], color=COLORS[mod], marker=markers[lab], s=30,
                           label=f"{mod} | {lab}")
    ax.set_xlabel("SNR from noise floor (dB)"); ax.set_ylabel("Measured MER (dB)")
    ax.set_title("MER vs SNR — gap below the line = non-noise impairments")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(outdir / "summary_mer_vs_snr.png"); plt.close(fig)

    # SER vs SNR with AWGN theory
    fig, ax = plt.subplots(figsize=(7, 5.5))
    g = np.linspace(-5, 40, 300)
    for mod in M.MODS:
        d = df[df["mod"] == mod]
        if not len(d):
            continue
        ax.semilogy(g, theory_ser(mod, 10 ** (g / 10)), color=COLORS[mod], lw=1.2, alpha=0.7)
        floor = 1 / (d["n_bits"] / M.bits_per_symbol(mod))
        ax.semilogy(d["snr_db"], np.maximum(d["ser"], floor), "o", color=COLORS[mod], ms=5, label=mod)
    ax.set_ylim(1e-5, 1); ax.set_xlim(-5, 40)
    ax.set_xlabel("SNR from noise floor (dB)"); ax.set_ylabel("Symbol error rate")
    ax.set_title("Measured SER (dots) vs AWGN theory (lines)")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "summary_ser_vs_snr.png"); plt.close(fig)

    # Error budget per modulation across TX levels
    mods = [m for m in M.MODS if m in set(df["mod"])]
    fig, axes = plt.subplots(1, len(mods), figsize=(3.4 * len(mods), 4.6), sharey=True, squeeze=False)
    for ax, mod in zip(axes[0], mods):
        d = df[df["mod"] == mod].groupby("level_db")[["frac_" + c for c in COMPONENTS]].mean().sort_index()
        bottom = np.zeros(len(d))
        x = np.arange(len(d))
        for c, col, name in zip(COMPONENTS, COMP_COLORS, COMP_NAMES):
            v = d["frac_" + c].values * 100
            ax.bar(x, v, bottom=bottom, color=col, label=name, width=0.8)
            bottom += v
        ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}" for v in d.index])
        ax.set_xlabel("TX level (dBFS)"); ax.set_title(mod)
    axes[0][0].set_ylabel("Share of error power (%)")
    axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Error budget vs transmit level", fontweight="bold")
    fig.tight_layout(); fig.savefig(outdir / "summary_error_budget.png", bbox_inches="tight"); plt.close(fig)

    # EVM vs TX level
    fig, ax = plt.subplots(figsize=(7, 5))
    for mod in mods:
        for lab in labels:
            d = df[(df["mod"] == mod) & (df["label"] == lab)].groupby("level_db")["evm_pct"].mean()
            if len(d):
                ax.plot(d.index, d.values, marker=markers[lab], color=COLORS[mod], label=f"{mod} | {lab}")
    ax.set_xlabel("TX level (dBFS)"); ax.set_ylabel("EVM (%)")
    ax.set_title("EVM vs transmit level")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(outdir / "summary_evm_vs_level.png"); plt.close(fig)


def grid_figure(eq_store, outdir, label):
    """Final constellations: rows = modulation, columns = TX level."""
    items = [k for k in eq_store if k[0] == label]
    mods = [m for m in M.MODS if any(k[1] == m for k in items)]
    levels = sorted({k[2] for k in items}, reverse=True)
    if not mods:
        return
    fig, axes = plt.subplots(len(mods), len(levels), figsize=(2.4 * len(levels), 2.4 * len(mods)), squeeze=False)
    for i, mod in enumerate(mods):
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            pts = eq_store.get((label, mod, lv))
            ideal = M.constellation(mod)
            lim = 1.6 * np.max(np.abs(ideal))
            if pts is not None:
                ax.scatter(pts.real, pts.imag, s=1, alpha=0.3, color="#1f4e79", edgecolors="none")
            ax.scatter(ideal.real, ideal.imag, marker="x", s=12, color="#c00000")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{lv:.0f} dBFS", fontsize=9)
            if j == 0:
                ax.set_ylabel(mod, fontsize=10)
    fig.suptitle(f"Received constellations — {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / f"grid_{label}.png"); plt.close(fig)
