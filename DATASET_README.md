# Known-bit acoustic link dataset (SigMF)

Recorded by Efty Al Rabby (IUT) with the code in this repository. A laptop's built-in speakers transmit, through the air of a small room, to the same laptop's built-in microphone array. **Every transmitted bit is known.** The dataset is designed for error analysis and for training and testing receivers.

## Signal
48 kHz audio, 3 kHz carrier, RRC roll-off 0.35, symbol rates 250–2000 baud. Modulations BPSK/QPSK/8PSK/16QAM/64QAM (Gray, unit power). A frame is an m-sequence BPSK preamble (127 or 511 symbols) followed by 600 data symbols; 5 frames per capture. Payload bits come from `numpy.random.default_rng(base_seed + frame_index)`. Use `acomm.modem.frame_symbols` / `payload_bits` to regenerate them.

## Archives in this record
| Archive | Contents |
|---|---|
| `canonical_r0.zip`, `canonical_r1.zip`, `canonical_r2.zip` | `data/canonical/`, split by repeat (90 captures each) |
| `p511_and_fan.zip` | `data/p511/`, `data/fan_off/`, `data/fan_on/` |

Unzip all of them into the same folder to recreate `data/<label>/`.

## Files
`<label>/<timestamp>_r<repeat>_<rate>bd_<mod>_<|level|>dB.sigmf-data` plus `.sigmf-meta`. The meta JSON holds `core:` SigMF fields and `acomm:` fields: `mod, level_db, rs, fs, fc, beta, span, preamble_len, n_data, n_frames, base_seed, lead_s, gap_s, label, notes, time, sim`.

## Conditions (labels)
| Label | Captures | Conditions |
|---|---|---|
| `canonical` | 270 | Table corner (near walls 15–20 cm; far walls 6/7/8 ft; ceiling 6–7 ft), 30° stand, volume 100, all enhancements off, fan off, Bluetooth off. 5 mods × 250/500/1000 baud × 0…−15 dBFS × 3 repeats. Preamble 127 |
| `fan_off`, `fan_on` | 15, 16 | Same spot; QPSK/16QAM; 250/1000 baud; 0/−6 dBFS; fan off vs on, recorded minutes apart |
| `p511` | 12 | Same spot and settings as `canonical`, same day, **511-symbol preamble**; QPSK/16QAM; 250/500/1000 baud; 0 dBFS; 2 repeats; lead-in 1.0 s, gaps 0.6 s |

## Known issues
- The playback latency is ~0.6 s, so the actual silent lead-in is longer than `lead_s`.
- The inter-frame gaps contain the decaying reverberation of the previous frame (−26 to −36 dB at 80–250 ms). Use the lead-in, not the gaps, for noise.
- Speaker and microphone share one chassis: there is no TX–RX distance variable.

## License
CC BY 4.0. Please cite the dataset (see `CITATION.cff`).
