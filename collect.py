"""Collect a labeled acoustic dataset: play known frames through the speaker, record with the mic.

Every capture is saved as SigMF (raw float32 audio + JSON metadata with everything
needed to regenerate the transmitted bits).
"""
import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from acomm import modem as M
from acomm import io_sigmf


def _dev(v):
    if v is None:
        return None
    return int(v) if str(v).isdigit() else v  # index or name text, e.g. "Microphone Array MME"


def play_record(x, fs, in_dev, out_dev, exclusive=False, tries=6):
    """Retry wrapper: if Windows re-numbers audio devices mid-run (e.g. Bluetooth reconnects),
    refresh the device list and try again."""
    import sounddevice as sd
    for t in range(tries):
        try:
            return _play_record(x, fs, in_dev, out_dev, exclusive)
        except sd.PortAudioError as e:
            if t == tries - 1:
                raise
            print(f"\n  audio device error ({e}); refreshing device list and retrying...")
            time.sleep(2 + 2 * t)
            sd._terminate()
            sd._initialize()


def _play_record(x, fs, in_dev, out_dev, exclusive=False):
    """Separate input and output streams, so any mic can pair with any speaker/driver."""
    import sounddevice as sd
    chunks = []

    def cb(indata, frames, t, status):
        chunks.append(indata[:, 0].copy())

    dev = _dev(in_dev)
    info = sd.query_devices(dev if dev is not None else sd.default.device[0], "input")
    ch = min(2, int(info["max_input_channels"]))
    kw = dict(dtype="float32")
    if exclusive:  # WASAPI exclusive mode bypasses Windows/driver effects (echo cancel, noise suppression)
        kw = dict(dtype="int16", extra_settings=sd.WasapiSettings(exclusive=True))
    with sd.InputStream(samplerate=fs, channels=ch, device=dev, callback=cb, **kw):
        sd.sleep(200)
        sd.play(x, fs, device=_dev(out_dev), blocking=True)
        sd.sleep(400)
    y = np.concatenate(chunks).astype(np.float32)
    return y / 32768.0 if exclusive else y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="test", help="scenario name, e.g. lab_1m, corridor_5m")
    ap.add_argument("--distance", type=float, default=None, help="speaker-mic distance in metres")
    ap.add_argument("--notes", default="", help="free-text environment notes")
    ap.add_argument("--mods", nargs="+", default=M.MODS, choices=M.MODS)
    ap.add_argument("--levels", nargs="+", type=float, default=[0, -6, -12, -18, -24, -30],
                    help="TX levels in dBFS")
    ap.add_argument("--rs", nargs="+", type=int, default=[1000], help="symbol rates (baud); 48000/rs must be an even integer")
    ap.add_argument("--fc", type=float, default=M.FC)
    ap.add_argument("--frames", type=int, default=5, help="frames per capture")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--preamble", type=int, default=127, choices=[127, 255, 511, 1023],
                    help="training preamble length in symbols (longer = longer trainable equalizer)")
    ap.add_argument("--lead", type=float, default=1.0,
                    help="silence before the first frame (s); used to measure background noise")
    ap.add_argument("--gap", type=float, default=0.6,
                    help="silence between frames (s); longer than the room reverberation")
    ap.add_argument("--out", default="data")
    ap.add_argument("--sim", action="store_true", help="use simulated room channel instead of audio hardware")
    ap.add_argument("--in-dev", default=None, help="input device index or name text")
    ap.add_argument("--out-dev", default=None, help="output device index or name text")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--exclusive", action="store_true", help="WASAPI exclusive mic capture (use with a WASAPI input device)")
    ap.add_argument("--check-devices", action="store_true", help="print which devices --in-dev/--out-dev select")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.check_devices:
        import sounddevice as sd
        print("IN :", sd.query_devices(_dev(args.in_dev) if args.in_dev else sd.default.device[0])["name"])
        print("OUT:", sd.query_devices(_dev(args.out_dev) if args.out_dev else sd.default.device[1])["name"])
        return

    M.set_preamble_len(args.preamble)
    rng = np.random.default_rng()
    combos = list(itertools.product(range(args.repeats), args.rs, args.mods, args.levels))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) / args.label
    for i, (rep, rs, mod, lv) in enumerate(combos, 1):
        sps = M.FS // rs
        assert M.FS % rs == 0 and sps % 2 == 0, f"bad symbol rate {rs}"
        seed = int(rng.integers(0, 2 ** 31 - 1))
        x = M.build_capture(mod, 10 ** (lv / 20), rs, args.frames, seed, fc=args.fc,
                            lead_s=args.lead, gap_s=args.gap)
        if args.sim:
            from acomm.channel_sim import room
            y = room(x, M.FS, rng)
        else:
            y = play_record(x, M.FS, args.in_dev, args.out_dev, args.exclusive)
            time.sleep(0.2)
            peak = float(np.max(np.abs(y)))
            ratio = len(y) / (len(x) + 0.6 * M.FS)
            if ratio > 1.08:
                print(f"\n  WARNING: recording is {ratio:.2f}x longer than played -> audio driver is duplicating buffers;"
                      " this capture is corrupt. Use MME input without --exclusive.")
            if i == 1 or peak < 0.03:
                print(f"\n  mic peak level {peak:.3f}" + ("  <-- too quiet: raise volume / move closer" if peak < 0.03 else ""))
        meta = dict(mod=mod, level_db=lv, rs=rs, fs=M.FS, fc=args.fc, beta=M.BETA, span=M.SPAN,
                    preamble_len=M.PREAMBLE_LEN, n_data=M.N_DATA, n_frames=args.frames,
                    base_seed=seed, lead_s=args.lead, gap_s=args.gap, label=args.label,
                    distance_m=args.distance, notes=args.notes, sim=args.sim,
                    time=datetime.now().isoformat(timespec="seconds"))
        name = f"{stamp}_r{rep}_{rs}bd_{mod}_{abs(lv):.0f}dB"
        io_sigmf.save(outdir / name, y, meta)
        print(f"\r[{i}/{len(combos)}] {args.label}: {mod} {lv:+.0f} dBFS {rs} bd", end="", flush=True)
    print(f"\nSaved {len(combos)} captures to {outdir}")


if __name__ == "__main__":
    main()
