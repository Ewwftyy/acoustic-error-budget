"""Software experiment: how far back in time can the equalizer reach, and does the error floor
follow the room's energy-decay curve at that reach?

For each capture the receiver is re-run with different equalizer half-spans H (in symbols).
The equalizer can cancel echoes up to about H symbols = H / R seconds after the direct path.
Reverberant energy arriving later than that is uncancellable and acts as a floor:

      predicted floor(t_reach) = -10 log10( EDC(t_reach) )

where EDC(t) is the fraction of the impulse-response energy arriving after t (Schroeder
energy-decay curve, measured from the same recordings). If the room sets the floor, all symbol
rates should collapse onto this one curve when plotted against t_reach = H / R.

Usage
  python span_sweep.py --label canonical --results results_paper
Outputs: <results>/span_sweep/<label>/span_sweep.csv, edc.csv, mer_vs_reach.png
Combine several labels afterwards with compare_reach.py.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geometry as G
from acomm import io_sigmf
from acomm import receiver as R
from acomm.analysis import capture_metrics

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
COL = {250: "#1f77b4", 300: "#17becf", 400: "#2ca02c", 500: "#ff7f0e", 600: "#bcbd22", 750: "#8c564b",
       800: "#e377c2", 1000: "#d62728", 1200: "#9467bd", 1500: "#7f7f7f", 2000: "#000000"}


def edc_db(label, data, nper=32768):
    files = sorted((Path(data) / label).glob("*.sigmf-meta"))
    best = G.best_ir_capture(files)
    hs = []
    for f in best:
        y, meta = io_sigmf.load(f)
        r = G.estimate_ir(y.astype(float), meta, nper=nper)
        if r is not None:
            hs.append(r[0])
            fs, fc = meta["fs"], meta["fc"]
    L = min(len(h) for h in hs)
    e = np.mean([np.abs(G.to_baseband(h[:L], fs, fc)) ** 2 for h in hs], axis=0)
    t = (np.arange(L) / fs - G.IR_PRE_MS * 1e-3) * 1e3
    tail = np.cumsum(e[::-1])[::-1] / e.sum()
    return t, 10 * np.log10(tail + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--label", default="canonical")
    ap.add_argument("--results", default="results_paper")
    ap.add_argument("--spans", nargs="+", type=int, default=[4, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--mods", nargs="+", default=["QPSK", "16QAM"])
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (resumable); 0 = no limit")
    args = ap.parse_args()
    out = Path(args.results) / "span_sweep" / args.label
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "span_sweep.csv"
    done = set()
    rows = []
    if csv.exists():
        old = pd.read_csv(csv)
        rows = old.to_dict("records")
        done = {(r["file"], int(r["half_span"])) for r in rows}

    files = []
    for f in sorted((Path(args.data) / args.label).glob("*.sigmf-meta")):
        g = json.load(open(f))["global"]
        if g["acomm:mod"] in args.mods and g["acomm:level_db"] == args.level:
            files.append(f)
    t0 = time.time()
    finished = True
    for f in files:
        y, meta = io_sigmf.load(f)
        for H in args.spans:
            key = (f.stem.replace(".sigmf", ""), H)
            if key in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                finished = False
                break
            R.EQ_HALF = H  # half-span in T/2 taps; the equalizer reaches H/2 symbols each side
            m = capture_metrics(R.receive_capture(y, meta), meta["mod"])
            if m is None:
                continue
            rows.append(dict(file=key[0], mod=meta["mod"], rs=meta["rs"], half_span=H,
                             reach_symbols=H / 2, reach_ms=H / 2 / meta["rs"] * 1e3,
                             mer_db=m["mer_db"], snr_db=m["snr_db"], ser=m["ser"]))
            pd.DataFrame(rows).to_csv(csv, index=False)
        if not finished:
            break
    R.EQ_HALF = 48
    if not finished:
        print(f"partial: {len(rows)} results saved; run the same command again to continue")
        return

    df = pd.DataFrame(rows)
    t, edc = edc_db(args.label, args.data)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(t[t > 0], -edc[t > 0], "k-", lw=2, label="room: -10 log10(energy arriving after t)")
    for rs in sorted(df.rs.unique()):
        d = df[df.rs == rs].groupby("reach_ms")["mer_db"].agg(["mean", "std"]).reset_index()
        ax.errorbar(d.reach_ms, d["mean"], yerr=d["std"], fmt="o-", color=COL.get(rs, None), ms=4, capsize=2,
                    label=f"measured MER, {rs} baud")
    ax.set_xscale("log")
    ax.set_xlabel("Equalizer reach after the direct path, t = (half-span in symbols) / (symbol rate)  [ms]")
    ax.set_ylabel("dB")
    ax.set_xlim(0.8, 400); ax.set_ylim(0, 35)
    ax.set_title(f"Error floor vs equalizer reach — {args.label}, TX {args.level:g} dBFS", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "mer_vs_reach.png"); plt.close(fig)
    pd.DataFrame(dict(t_ms=t, edc_db=edc)).iloc[::12].to_csv(out / "edc.csv", index=False)
    print(df.groupby(["rs", "half_span", "reach_ms"])["mer_db"].mean().round(2).to_string())
    print(f"\nFigures -> {out}")


if __name__ == "__main__":
    main()
