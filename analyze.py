"""Analyze every SigMF capture under a data folder.

Outputs (in --out):
  metrics.csv                  one row per capture (EVM, MER, SNR, SER, BER, error budget, ...)
  captures/*.png               per-capture constellation stages + error budget
  grid_<label>.png             final constellations, modulation x TX level
  summary_*.png                MER vs SNR, SER vs theory, error budget, EVM vs level
  symbols.npz                  received/transmitted symbols for later ML work
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from acomm import io_sigmf, plots
from acomm.analysis import capture_metrics
from acomm.receiver import receive_capture

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-capture-plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "captures").mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data).rglob("*.sigmf-meta"))
    rows, eq_store, sym_store = [], {}, {}
    failed = []
    for i, fpath in enumerate(files, 1):
        print(f"\r[{i}/{len(files)}] {fpath.stem}", end=" " * 10, flush=True)
        y, meta = io_sigmf.load(fpath)
        frames = receive_capture(y, meta)
        m = capture_metrics(frames, meta["mod"])
        if m is None:
            failed.append(fpath.stem)
            continue
        row = dict(file=fpath.stem, label=meta["label"], mod=meta["mod"], level_db=meta["level_db"],
                   rs=meta["rs"], distance_m=meta.get("distance_m"), sim=meta.get("sim"), **m)
        rows.append(row)
        eq = np.concatenate([f["eq"] for f in frames])
        tx = np.concatenate([f["tx_sym"] for f in frames])
        key = (meta["label"], meta["mod"], meta["level_db"])
        eq_store.setdefault(key, eq)
        sym_store[fpath.stem + "__rx"] = eq
        sym_store[fpath.stem + "__tx"] = tx
        if not args.no_capture_plots:
            title = f"{meta['label']} | {meta['mod']} | {meta['level_db']:+.0f} dBFS | {meta['rs']} baud"
            plots.capture_figure(frames, meta["mod"], m, title, out / "captures" / f"{fpath.stem}.png")
    print()
    if not rows:
        print("No frames detected in any capture. Check volume / mic input.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(out / "symbols.npz", **sym_store)
    plots.summary_figures(df, out)
    for lab in df["label"].unique():
        plots.grid_figure(eq_store, out, lab)
    msg = f"{len(rows)} captures analyzed -> {out}"
    if failed:
        msg += f" | {len(failed)} with no detected frames (see failed.txt)"
        (out / "failed.txt").write_text("\n".join(failed))
    print(msg)


if __name__ == "__main__":
    main()
