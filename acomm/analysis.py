"""Error metrics and constellation-based error-budget decomposition."""
import numpy as np
from scipy.special import erfc
from . import modem as M

COMPONENTS = ["rotation_gain", "iq_imbalance", "dc_offset", "compression",
              "phase_jitter", "amplitude_jitter", "noise", "isi_distortion"]


def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def theory_ser(mod, snr_lin):
    m = 2 ** M.bits_per_symbol(mod)
    g = np.asarray(snr_lin, float)
    if mod == "BPSK":
        return Q(np.sqrt(2 * g))
    if mod == "QPSK":
        q = Q(np.sqrt(g))
        return 2 * q - q ** 2
    if mod.endswith("PSK"):
        return np.minimum(1, 2 * Q(np.sqrt(2 * g) * np.sin(np.pi / m)))
    sm = np.sqrt(m)
    p = 2 * (1 - 1 / sm) * Q(np.sqrt(3 * g / (m - 1)))
    return 1 - (1 - p) ** 2


def decompose(y, s, mod):
    """Split mean|y - s|^2 into physically distinct parts using the known symbols s."""
    if np.allclose(s.imag, 0):  # BPSK: s == conj(s), so I/Q imbalance is not separable
        a, c = np.linalg.lstsq(np.column_stack([s, np.ones_like(s)]), y, rcond=None)[0]
        b = 0.0
    else:
        A = np.column_stack([s, np.conj(s), np.ones_like(s)])
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    z = (y - b * np.conj(s) - c) / a

    # ring-dependent gain (amplitude compression); single ring for PSK
    rings = np.round(np.abs(s), 6)
    g = np.ones_like(s, dtype=complex)
    ring_gain = {}
    for r in np.unique(rings):
        m = rings == r
        gr = np.mean((z[m] * np.conj(s[m])).real / np.abs(s[m]) ** 2)
        ring_gain[r] = gr
        g[m] = gr
    if len(ring_gain) > 1:
        g = g / np.mean(list(ring_gain.values()))
        rv = sorted(ring_gain)
        comp_db = 20 * np.log10(ring_gain[rv[-1]] / ring_gain[rv[0]])
    else:
        g[:] = 1
        comp_db = 0.0

    eps = z - g * s
    u = np.conj(s) / np.abs(s)
    er = (eps * u).real
    et = (eps * u).imag
    Er, Et = np.mean(er ** 2), np.mean(et ** 2)
    aa = abs(a) ** 2
    ps = np.mean(abs(s) ** 2)
    uniq = np.unique(rings)
    if len(uniq) >= 2:
        # Multi-ring constellations (QAM): phase and amplitude jitter grow with ring radius^2,
        # noise and ISI do not. Fit per ring:  Er_i = I/2 + A*rho_i^2,  Et_i = I/2 + P*rho_i^2
        # (non-negative least squares, weighted by symbol count) -> separates all three.
        from scipy.optimize import nnls
        rowsA, rhs = [], []
        for r in uniq:
            m = rings == r
            wgt = np.sqrt(m.sum())
            rho2 = np.mean(abs(s[m]) ** 2)
            rowsA += [[wgt, wgt * rho2, 0.0], [wgt, 0.0, wgt * rho2]]
            rhs += [wgt * np.mean(er[m] ** 2), wgt * np.mean(et[m] ** 2)]
        (half_iso, A_coef, P_coef), _ = nnls(np.array(rowsA), np.array(rhs))
        phase_j, amp_j, iso = P_coef * ps, A_coef * ps, 2 * half_iso
    else:
        # Single ring (PSK): only the net radial/tangential imbalance is identifiable.
        phase_j, amp_j, iso = max(0.0, Et - Er), max(0.0, Er - Et), 2 * min(Er, Et)
    parts = dict(
        rotation_gain=abs(a - 1) ** 2 * ps,
        iq_imbalance=abs(b) ** 2 * ps,
        dc_offset=abs(c) ** 2,
        compression=aa * np.mean(abs((g - 1) * s) ** 2),
        phase_jitter=aa * phase_j,
        amplitude_jitter=aa * amp_j,
        isotropic=aa * iso,
    )
    extra = dict(
        irr_db=-20 * np.log10(max(abs(b / a), 1e-12)),
        dc_dbc=10 * np.log10(max(abs(c) ** 2, 1e-15)),
        compression_db=comp_db,
        rot_deg=np.degrees(np.angle(a)),
        phase_err_rms_deg=np.degrees(np.std(np.angle(z / (g * s)))),
    )
    return parts, extra


def frame_metrics(fr, mod):
    y, s = fr["eq"], fr["tx_sym"]
    err = np.mean(np.abs(y - s) ** 2) / np.mean(np.abs(s) ** 2)
    _, idx = M.decide(y, mod)
    _, tx_idx = M.decide(s, mod)
    bits = M.symbols_to_bits(idx, mod)
    parts, extra = decompose(y, s, mod)
    noise = fr["noise_out"]
    iso = parts.pop("isotropic")
    if np.isfinite(noise):
        parts["noise"] = min(noise, iso)
        parts["isi_distortion"] = iso - parts["noise"]
    else:
        parts["noise"], parts["isi_distortion"] = iso, 0.0
    out = dict(
        evm_pct=100 * np.sqrt(err),
        mer_db=-10 * np.log10(err),
        snr_db=10 * np.log10(1 / noise) if np.isfinite(noise) and noise > 0 else np.nan,
        gap_snr_db=10 * np.log10(1 / fr["gap_out"]) if np.isfinite(fr.get("gap_out", np.nan)) and fr["gap_out"] > 0 else np.nan,
        ser=np.mean(idx != tx_idx),
        bit_errors=int(np.sum(bits != fr["tx_bits"])),
        n_bits=len(bits),
        **extra,
    )
    tot = sum(parts.values())
    for k in COMPONENTS:
        out["pow_" + k] = parts[k]
        out["frac_" + k] = parts[k] / tot if tot > 0 else 0.0
    return out


def capture_metrics(frames, mod):
    rows = [frame_metrics(f, mod) for f in frames]
    if not rows:
        return None
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ("bit_errors", "n_bits")}
    agg["bit_errors"] = sum(r["bit_errors"] for r in rows)
    agg["n_bits"] = sum(r["n_bits"] for r in rows)
    agg["ber"] = agg["bit_errors"] / agg["n_bits"]
    agg["frames"] = len(rows)
    return agg
