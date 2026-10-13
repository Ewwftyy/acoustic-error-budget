# acoustic-error-budget

**Where does the error come from?** This repository measures it on a real link: a laptop speaker and microphone, every transmitted bit known. The error of each received constellation is decomposed into its physical causes, and the error floor is shown to follow the room's energy-decay curve read at the equalizer's reach.

- Dataset description: [`DATASET_README.md`](DATASET_README.md) (data on Zenodo)
- Paper: in preparation

![MER vs equalizer reach](figures/reach_canonical.png)

## Quick start
```bash
pip install -r requirements.txt
python collect.py --sim --label sim_room           # no hardware: simulated room channel
python analyze.py data --out results               # receiver + error budget for every capture
python validate_budget.py --out results/validation # validation of the decomposition on synthetic data
```
With real audio hardware (Windows example; list devices with `python collect.py --list-devices`):
```bash
python collect.py --label myroom --rs 250 1000 --mods QPSK 16QAM --levels 0 -6 \
  --in-dev "Microphone Array Realtek MME" --out-dev "Speakers Realtek MME" --notes "describe the room"
```
**Turn off all microphone and speaker "audio enhancements"** first. Echo cancellation erases the signal within about a second.

## What each script does
| Script | Purpose |
|---|---|
| `collect.py` | Plays known frames and records them, or uses `--sim`. Saves SigMF. Options: symbol rates, modulations, TX levels, preamble length (127–1023), lead-in and gap lengths |
| `analyze.py` | Receiver (sync, CFO, T/2 LS+LMS equalizer, PLL) → EVM, MER, SNR, SER, BER and an 8-part error budget per capture → `metrics.csv`, figures, `symbols.npz` |
| `compare.py` | Side-by-side comparison of recording conditions |
| `geometry.py` | Impulse response and decay from the recordings; ideal (MMSE) equalizer bound vs measured MER |
| `span_sweep.py` | Re-runs the receiver with different equalizer lengths: MER vs equalizer reach, overlaid on the room's energy-decay curve |
| `compare_reach.py` | Overlays the reach results of several conditions |
| `validate_budget.py` | Validates the decomposition on synthetic data with known impairments |
| `train_nrx.py` | Neural receiver vs classical receiver on the same symbols (PyTorch) |

`acomm/` holds the library: `modem.py` (waveforms), `receiver.py`, `analysis.py` (metrics and decomposition), `io_sigmf.py`, `channel_sim.py`, `plots.py`, `features.py`.

## Data format
Each capture is `<name>.sigmf-data` (float32 little-endian mono audio, 48 kHz) plus `<name>.sigmf-meta` (JSON). Everything needed to regenerate the transmitted bits is stored under the `acomm:` keys: modulation, symbol rate, TX level, seed, preamble length, frame and gap timing, label, free-text notes.

## Citation
See [`CITATION.cff`](CITATION.cff).

## License
Code: MIT ([`LICENSE`](LICENSE)). Data: CC BY 4.0.
