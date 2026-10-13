"""Overlay the equalizer-reach results of several labels (run span_sweep.py on each first).

  python compare_reach.py --labels p127 p511 absorb --results results_paper --rs 1000
Output: <results>/span_sweep/compare_<labels>_<rs>bd.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--results", default="results_paper")
    ap.add_argument("--rs", nargs="+", type=int, default=[250, 1000])
    args = ap.parse_args()
    base = Path(args.results) / "span_sweep"
    for rs in args.rs:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for i, lab in enumerate(args.labels):
            d = base / lab
            if not (d / "span_sweep.csv").exists():
                print(f"{lab}: run span_sweep.py first"); continue
            df = pd.read_csv(d / "span_sweep.csv")
            edc = pd.read_csv(d / "edc.csv")
            e = edc[edc.t_ms > 0]
            ax.plot(e.t_ms, -e.edc_db, "-", color=PAL[i % len(PAL)], lw=1.4, alpha=0.7, label=f"{lab}: room decay curve")
            m = df[df.rs == rs].groupby("reach_ms")["mer_db"].agg(["mean", "std"]).reset_index()
            if len(m):
                ax.errorbar(m.reach_ms, m["mean"], yerr=m["std"], fmt="o-", color=PAL[i % len(PAL)], ms=5, capsize=2,
                            label=f"{lab}: measured MER")
        ax.set_xscale("log"); ax.set_xlim(0.8, 600); ax.set_ylim(0, 35)
        ax.set_xlabel("Equalizer reach after the direct path (ms)"); ax.set_ylabel("dB")
        ax.set_title(f"MER vs equalizer reach at {rs} baud", fontweight="bold"); ax.legend(fontsize=8)
        fig.tight_layout()
        name = base / f"compare_{'_'.join(args.labels)}_{rs}bd.png"
        fig.savefig(name); plt.close(fig)
        print(f"-> {name}")


if __name__ == "__main__":
    main()
