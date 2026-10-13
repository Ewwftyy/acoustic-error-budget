"""Neural receiver (NRX) vs classical LS receiver on the acoustic dataset.

The classical receiver is the one in acomm/receiver.py: preamble sync, CFO, T/2 least-squares
equalizer + LMS + decision-directed PLL. The NRX sees, for every data symbol:
  - the T/2-spaced received window around it (97 complex samples),
  - the frame's LS equalizer taps (a summary of the channel, estimated from the preamble),
  - the classical LS output for that symbol,
and outputs a probability for each constellation point. Both are scored on the same test symbols.

Splits (--split):
  repeat  train on repeat r0 captures, test on r1, r2... (same conditions, different time)
  label   train on --train-labels, test on --test-labels (generalization to a new environment)
  sim2real train only on simulated captures (acomm/channel_sim.py), test on real --test-labels

Examples
  python train_nrx.py --labels canonical --split repeat
  python train_nrx.py --split label --train-labels canonical --test-labels open_room
  python train_nrx.py --split sim2real --test-labels canonical --sim-captures 300
  python train_nrx.py --labels canonical --split repeat --no-taps --no-ls      (ablation: window only)
  python train_nrx.py --labels canonical --split repeat --ls-prior            (residual on top of LS decision)
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from acomm import modem as M
from acomm.features import load_capture, simulate_capture

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data
def gather(data_dir, labels, mods, rates, cache):
    caps = []
    files = sorted(Path(data_dir).rglob("*.sigmf-meta"))
    for i, f in enumerate(files, 1):
        lab = f.parent.name
        if labels and lab not in labels:
            continue
        stem = f.name
        if not any(f"_{m}_" in stem for m in mods) or not any(f"_{r}bd_" in stem for r in rates):
            continue
        print(f"\r  features {i}/{len(files)} {lab}/{f.stem[:40]}", end=" " * 8, flush=True)
        info, frames = load_capture(f, cache)
        if frames:
            caps.append((info, frames))
    print()
    return caps


def to_arrays(caps, use_taps, use_ls):
    X, Y, LS, keys = [], [], [], []
    for ci, (info, frames) in enumerate(caps):
        for fr in frames:
            n = len(fr["tx"])
            parts = [np.concatenate([fr["win"].real, fr["win"].imag], 1)]
            if use_taps:
                t = np.tile(fr["taps"], (n, 1))
                parts.append(np.concatenate([t.real, t.imag], 1))
            if use_ls:
                parts.append(np.stack([fr["ls"].real, fr["ls"].imag], 1))
            X.append(np.concatenate(parts, 1).astype(np.float32))
            Y.append(fr["tx"].astype(np.int64))
            LS.append(fr["ls"])
            keys.append(np.full(n, ci))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(LS), np.concatenate(keys)


# ----------------------------------------------------------------------------- model
class NRX(nn.Module):
    """Two-branch network: a 1D CNN over the received window (+ taps as 2 more channels),
    concatenated with the LS output, then an MLP classifier over the M constellation points."""

    def __init__(self, n_win, use_taps, use_ls, M_points, width=64, const=None):
        super().__init__()
        # LS prior: the network starts from the classical decision (distance of the LS output to each
        # constellation point) and learns a correction on top, so it begins no worse than LS.
        if const is not None:
            self.register_buffer("const", torch.tensor(np.stack([const.real, const.imag], 1), dtype=torch.float32))
        else:
            self.const = None
        self.log_alpha = nn.Parameter(torch.tensor(1.5))
        self.n_win, self.use_taps, self.use_ls = n_win, use_taps, use_ls
        cin = 4 if use_taps else 2
        self.cnn = nn.Sequential(
            nn.Conv1d(cin, width, 7, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 7, padding=3, stride=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        mlp_in = width * 4 + (2 if use_ls else 0) + 2  # +2: centre sample (real, imag) passed straight through
        self.head = nn.Sequential(nn.Linear(mlp_in, 256), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(256, 256), nn.GELU(), nn.Linear(256, M_points))

    def forward(self, x):
        n = self.n_win
        win = torch.stack([x[:, :n], x[:, n:2 * n]], 1)
        o = 2 * n
        chans = [win]
        if self.use_taps:
            chans.append(torch.stack([x[:, o:o + n], x[:, o + n:o + 2 * n]], 1))
            o += 2 * n
        h = self.cnn(torch.cat(chans, 1)).flatten(1)
        c = n // 2
        feats = [h, win[:, :, c]]
        logits = None
        if self.use_ls:
            ls = x[:, o:o + 2]
            feats.append(ls)
        logits = self.head(torch.cat(feats, 1))
        if self.use_ls and self.const is not None:
            d2 = ((ls[:, None, :] - self.const[None]) ** 2).sum(-1)
            logits = logits - self.log_alpha.exp() * d2
        return logits


def train(model, X, Y, Xv, Yv, epochs, device, lr=2e-3, bs=1024):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(X) // bs + 1))
    lossf = nn.CrossEntropyLoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for b in range(0, len(Xt), bs):
            idx = perm[b:b + bs]
            xb, yb = Xt[idx].to(device), Yt[idx].to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        ser_v = (predict(model, Xv, device) != Yv).mean()
        if ser_v < best:
            best, best_state = ser_v, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep + 1:3d}/{epochs}  train loss {tot / len(Xt):.4f}  val SER {ser_v:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, device, bs=8192):
    model.eval()
    out = []
    for b in range(0, len(X), bs):
        out.append(model(torch.tensor(X[b:b + bs]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--labels", nargs="*", default=None, help="labels used by --split repeat")
    ap.add_argument("--split", choices=["repeat", "label", "sim2real"], default="repeat")
    ap.add_argument("--train-labels", nargs="*", default=[])
    ap.add_argument("--test-labels", nargs="*", default=[])
    ap.add_argument("--mods", nargs="+", default=["QPSK", "16QAM"], choices=M.MODS)
    ap.add_argument("--rs", nargs="+", type=int, default=[250, 500, 1000])
    ap.add_argument("--sim-captures", type=int, default=300, help="simulated captures per modulation (sim2real)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-taps", action="store_true", help="ablation: do not give the network the LS taps")
    ap.add_argument("--no-ls", action="store_true", help="ablation: do not give the network the LS output")
    ap.add_argument("--ls-prior", action="store_true",
                    help="start the network from the classical decision and learn a correction (safer with little data)")
    ap.add_argument("--out", default="results_nrx")
    ap.add_argument("--cache", default="cache_features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_taps, use_ls = not args.no_taps, not args.no_ls
    tag = f"{args.split}_{'+'.join(args.train_labels or args.labels or ['all'])}" \
          f"_to_{'+'.join(args.test_labels or ['heldout'])}" + ("_notaps" if not use_taps else "") + ("_nols" if not use_ls else "") + ("_lsprior" if args.ls_prior else "")
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   run: {tag}")

    rows = []
    for mod in args.mods:
        print(f"\n=== {mod} ===")
        if args.split == "repeat":
            caps = gather(args.data, args.labels, [mod], args.rs, args.cache)
            train_caps = [c for c in caps if c[0]["repeat"] == 0]
            test_caps = [c for c in caps if c[0]["repeat"] > 0]
        elif args.split == "label":
            train_caps = gather(args.data, args.train_labels, [mod], args.rs, args.cache)
            test_caps = gather(args.data, args.test_labels, [mod], args.rs, args.cache)
        else:
            rng = np.random.default_rng(args.seed)
            train_caps = []
            for k in range(args.sim_captures):
                print(f"\r  simulating {k + 1}/{args.sim_captures}", end="", flush=True)
                info, fr = simulate_capture(mod, int(rng.choice(args.rs)), float(rng.choice([0, -3, -6, -9, -12, -15])), rng)
                if fr:
                    train_caps.append((info, fr))
            print()
            test_caps = gather(args.data, args.test_labels, [mod], args.rs, args.cache)
        if not train_caps or not test_caps:
            print(f"  not enough captures for {mod} (train {len(train_caps)}, test {len(test_caps)}) - skipped")
            continue

        Xtr, Ytr, _, _ = to_arrays(train_caps, use_taps, use_ls)
        Xte, Yte, LSte, Kte = to_arrays(test_caps, use_taps, use_ls)
        # 10% of the training captures held out for model selection (never the test set)
        rng = np.random.default_rng(args.seed)
        ntr = len(train_caps)
        val_caps = set(rng.choice(ntr, max(1, ntr // 10), replace=False)) if ntr > 1 else set()
        _, _, _, Ktr = to_arrays(train_caps, use_taps, use_ls)
        vmask = np.isin(Ktr, list(val_caps))
        if not vmask.any() or vmask.all():
            vmask = np.zeros(len(Ktr), bool); vmask[::10] = True
        print(f"  train symbols {int((~vmask).sum())}  val {int(vmask.sum())}  test {len(Yte)}")

        n_win = train_caps[0][1][0]["win"].shape[1]
        model = NRX(n_win, use_taps, use_ls, 2 ** M.bits_per_symbol(mod), const=M.constellation(mod) if args.ls_prior else None).to(device)
        t0 = time.time()
        model = train(model, Xtr[~vmask], Ytr[~vmask], Xtr[vmask], Ytr[vmask], args.epochs, device)
        print(f"  trained in {time.time() - t0:.0f} s")
        torch.save(model.state_dict(), out / f"nrx_{mod}.pt")

        nn_idx = predict(model, Xte, device)
        _, ls_idx = M.decide(LSte, mod)
        for ci, (info, _) in enumerate(test_caps):
            m = Kte == ci
            tx = Yte[m]
            b_tx = M.symbols_to_bits(tx, mod)
            rows.append(dict(mod=mod, label=info["label"], rs=info["rs"], level_db=info["level_db"],
                             capture=info["name"], n_sym=int(m.sum()),
                             ser_ls=float((ls_idx[m] != tx).mean()), ser_nrx=float((nn_idx[m] != tx).mean()),
                             ber_ls=float((M.symbols_to_bits(ls_idx[m], mod) != b_tx).mean()),
                             ber_nrx=float((M.symbols_to_bits(nn_idx[m], mod) != b_tx).mean())))
        d = pd.DataFrame([r for r in rows if r["mod"] == mod])
        print(f"  TEST {mod}: SER classical {d.ser_ls.mean():.4f}  ->  NRX {d.ser_nrx.mean():.4f}")

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_capture.csv", index=False)
    summ = df.groupby(["mod", "rs", "level_db"])[["ser_ls", "ser_nrx", "ber_ls", "ber_nrx"]].mean().reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)
    plot(summ, out, tag)
    print("\n" + summ.round(4).to_string(index=False))
    print(f"\nResults -> {out}")


def plot(summ, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    mods, rates = sorted(summ["mod"].unique()), sorted(summ["rs"].unique())
    fig, axes = plt.subplots(len(mods), len(rates), figsize=(4.2 * len(rates), 3.6 * len(mods)), squeeze=False, sharey=True)
    for i, mod in enumerate(mods):
        for j, rs in enumerate(rates):
            ax = axes[i][j]
            d = summ[(summ["mod"] == mod) & (summ["rs"] == rs)].sort_values("level_db")
            if len(d):
                ax.semilogy(d.level_db, np.maximum(d.ser_ls, 1e-4), "o-", color="#7f7f7f", label="Classical LS")
                ax.semilogy(d.level_db, np.maximum(d.ser_nrx, 1e-4), "s-", color="#d62728", label="Neural receiver")
            ax.set_ylim(1e-4, 1)
            ax.set_title(f"{mod}, {rs} baud"); ax.set_xlabel("TX level (dBFS)")
            if j == 0:
                ax.set_ylabel("Symbol error rate (test)")
    axes[0][-1].legend()
    fig.suptitle(f"Neural vs classical receiver — {tag}  (points at 1e-4 = no errors)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "ser_nrx_vs_ls.png"); plt.close(fig)


if __name__ == "__main__":
    main()
