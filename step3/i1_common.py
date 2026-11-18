"""Step 3, Stage 0: the independent synthetic validation construction (Path I1).

Written fresh for Step 3 on 2026-11-18. Nothing here imports or copies the
archived simulator, converter or label code. The settled facts it builds on
(the bandext edge definition, the observed-cells label arithmetic, the
maskfix mask form, the organisers' stated variation classes) are facts of
the project record, re-implemented here from their definitions.

Construction summary
--------------------
Background: a real screened pool window (data/bg_windows.npz), drawn from
the role partitions in bg_roles.npz. With probability 0.3 the background is
the energy sum of two windows from the same site (the background-mixture
mode of the record); with probability 0.5 a smooth per-band channel jitter
within plus/minus 2 dB is applied to the background energies.

Passage: an event assembled at the waveform level from kept-side clips of
one side (family or i1) and one kind (vessel or ambient). Variations are
applied to the waveform before conversion, mirroring the organisers'
stated classes: time warp by a log-uniform factor in [0.5, 2.0], time
reversal with probability 0.25, per-band colouring gains (smooth random
walk across bands, clipped to plus/minus 7 dB) applied in the waveform's
FFT domain, and a smooth per-minute gain envelope within plus/minus
3.5 dB. The varied waveform is converted to per-minute band powers by the
bandext conversion defined below, implemented here from its definition:
deci-decade band edges (centre times 10 to the plus/minus 0.05) for the
interior bands, band 0 integrating from 0 Hz, band 24 up to the 16 kHz
Nyquist; 2-second Hann segments with constant detrend, periodogram
density scaling, band power the integral of the density over the band,
averaged over each minute's segments.

Mixing: linear-energy addition. The passage is scaled by a single gain so
the target share holds exactly over the scaling domain (observed cells
for drop-first order, all cells for mix-first order, drawn 50/50). The
mixture is exported on the 0.1 dB grid and the mask applied. The label is
the realised share over observed cells, computed from the unquantised
component energies by the settled observed-cells arithmetic, implemented
here independently.

Share prior (deliberately different from the frozen simulator's):
25% exact zeros; otherwise uniform on [0, 0.9]. Kind for a nonzero draw:
vessel 0.7, ambient 0.3. Event durations are drawn from the empirical
realised-duration distribution recorded in the frozen bandext_maskfix
validation set (durations above zero), a recorded fact of the
construction the organisers' process produced in the frozen mirror.
Mask form (maskfix, measured): complete 0.517, minute 1 only 0.413,
minute 15 only 0.065, both edge minutes 0.005.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
FS = 32000
N_MIN, N_BAND = 15, 25
CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(N_BAND))
EDGES = np.empty((N_BAND, 2))
EDGES[:, 0] = CENTRES / 10 ** 0.05
EDGES[:, 1] = CENTRES * 10 ** 0.05
EDGES[0, 0] = 0.0
EDGES[24, 1] = FS / 2.0

MASK_PATTERNS = [  # (missing minutes tuple, probability) -- maskfix form
    ((), 0.517),
    ((0,), 0.413),
    ((14,), 0.065),
    ((0, 14), 0.005),
]

_wav_cache = {}


def load_wav(clip_key):
    if clip_key not in _wav_cache:
        if len(_wav_cache) > 14:
            _wav_cache.pop(next(iter(_wav_cache)))
        _wav_cache[clip_key] = np.load(
            HERE / "wav_cache" / (clip_key.replace("/", "_") + ".npy"))
    return _wav_cache[clip_key]


def db_to_lin(db):
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def lin_to_db(lin):
    return 10.0 * np.log10(np.maximum(lin, 1e-30))


def convert_waveform_to_minutes(x, fs=FS):
    """Waveform -> (n_minutes, 25) linear band powers, bandext definition.

    2 s Hann segments, constant detrend, periodogram density; each
    minute's power is the mean over its 30 segments.
    """
    seg = 2 * fs
    n_min = len(x) // (60 * fs)
    if n_min == 0:
        return np.zeros((0, N_BAND))
    x = x[:n_min * 60 * fs].reshape(n_min, 30, seg).astype(np.float32)
    win = np.hanning(seg).astype(np.float32)
    xw = (x - x.mean(axis=2, keepdims=True)) * win
    spec = np.fft.rfft(xw, axis=2)
    psd = (np.abs(spec) ** 2) / (fs * np.sum(win.astype(np.float64) ** 2))
    psd[..., 1:-1] *= 2.0  # one-sided density
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    df = freqs[1] - freqs[0]
    out = np.empty((n_min, 30, N_BAND))
    for b in range(N_BAND):
        lo, hi = EDGES[b]
        bins = (freqs >= lo) & (freqs < hi)
        out[:, :, b] = psd[:, :, bins].sum(axis=2) * df
    return out.mean(axis=1)


def _smooth_walk(rng, n, step_sd, clip):
    w = np.cumsum(rng.normal(0.0, step_sd, n))
    w -= w.mean()
    return np.clip(w, -clip, clip)


def build_event_waveform(rng, clip_keys, target_minutes):
    """Assemble and vary an event waveform of exactly target_minutes minutes.

    clip_keys: the side+kind pool to draw pieces from. Returns the varied
    waveform (float32) and the list of clip keys used.
    """
    warp = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    need_src = int(np.ceil(target_minutes * warp))
    pieces, used, have_min = [], [], 0
    guard = 0
    while have_min < need_src and guard < 12:
        guard += 1
        key = clip_keys[int(rng.integers(len(clip_keys)))]
        wav = load_wav(key)
        total_min = len(wav) // (60 * FS)
        if total_min == 0:
            continue
        take = min(total_min, need_src - have_min)
        start = int(rng.integers(0, total_min - take + 1)) if total_min > take else 0
        piece = wav[start * 60 * FS:(start + take) * 60 * FS].astype(np.float32)
        piece *= np.float32(10 ** (rng.uniform(-3.0, 3.0) / 20.0))  # piece gain jitter
        pieces.append(piece)
        used.append(key)
        have_min += take
    if have_min == 0:
        return None, used
    src = np.concatenate(pieces)[:need_src * 60 * FS]
    # Time warp (FFT resample) to the target duration.
    n_out = target_minutes * 60 * FS
    if len(src) != n_out:
        spec = np.fft.rfft(src)
        if n_out < len(src):
            spec = spec[:n_out // 2 + 1]
        else:
            spec = np.concatenate(
                [spec, np.zeros(n_out // 2 + 1 - len(spec), dtype=spec.dtype)])
        ev = np.fft.irfft(spec, n=n_out) * np.float32(n_out / len(src))
    else:
        ev = src
    if rng.random() < 0.25:
        ev = ev[::-1].copy()
    # Per-band colouring in the FFT domain (smooth walk over bands).
    gains_db = _smooth_walk(rng, N_BAND, 2.5, 7.0)
    spec = np.fft.rfft(ev)
    freqs = np.fft.rfftfreq(len(ev), d=1.0 / FS)
    band_idx = np.clip(np.searchsorted(EDGES[:, 1], freqs), 0, N_BAND - 1)
    spec = spec * np.power(10.0, gains_db[band_idx] / 20.0).astype(np.float32)
    ev = np.fft.irfft(spec, n=len(ev))
    # Per-minute gain envelope (smooth walk, plus/minus 3.5 dB).
    env_db = _smooth_walk(rng, target_minutes, 1.2, 3.5)
    ev = ev.reshape(target_minutes, 60 * FS) * np.power(10.0, env_db / 20.0)[:, None]
    return ev.reshape(-1).astype(np.float32), used


class BackgroundPool:
    def __init__(self):
        z = np.load(ROOT / "data" / "bg_windows.npz")
        self.X = z["X"]
        self.site = np.asarray(z["site"])
        r = np.load(HERE / "bg_roles.npz")
        self.role = np.asarray(r["role"])
        self.by_role_site = {}
        for role in set(self.role.tolist()):
            for s in set(self.site.tolist()):
                idx = np.where((self.role == role) & (self.site == s))[0]
                if len(idx):
                    self.by_role_site[(role, s)] = idx
        self.sites_by_role = {}
        for (role, s) in self.by_role_site:
            self.sites_by_role.setdefault(role, []).append(s)

    def draw(self, rng, role, site=None):
        sites = self.sites_by_role[role] if site is None else [site]
        s = sites[int(rng.integers(len(sites)))]
        idx = self.by_role_site[(role, s)]
        i = int(idx[int(rng.integers(len(idx)))])
        return self.X[i].astype(np.float64), s, i


def draw_mask(rng):
    r = rng.random()
    acc = 0.0
    for missing, p in MASK_PATTERNS:
        acc += p
        if r <= acc:
            return missing
    return ()


def make_window(rng, pool, bg_spec, clip_pools, dur_values):
    """Build one mixed window.

    bg_spec: callable(rng) -> (bg_db (15,25), site, tag) drawing the
    background in dB before jitter/mixture modes are applied here.
    clip_pools: dict kind -> list of clip keys for the relevant side, or
    None to force a pure background window (null set).
    Returns a record dict, or None on a failed event build.
    """
    bg_db, site, bg_tag = bg_spec(rng)
    E_bg = db_to_lin(bg_db)
    if rng.random() < 0.30 and bg_spec.allow_mixture:
        bg2_db, _s2, _t2 = bg_spec(rng, same_site=site)
        E_bg = E_bg + db_to_lin(bg2_db)
        bg_tag = bg_tag + "+mix"
    if rng.random() < 0.5:
        jit = _smooth_walk(rng, N_BAND, 0.8, 2.0)
        E_bg = E_bg * np.power(10.0, jit / 10.0)[None, :]

    missing = draw_mask(rng)
    obs = np.ones(N_MIN, dtype=bool)
    obs[list(missing)] = False

    target = 0.0
    kind = "none"
    dur = 0
    used = []
    E_add = np.zeros((N_MIN, N_BAND))
    order = "drop-first" if rng.random() < 0.5 else "mix-first"
    if clip_pools is not None and rng.random() >= 0.25:
        target = float(rng.uniform(0.0, 0.9))
        kind = "vessel" if rng.random() < 0.7 else "ambient"
        dur = int(dur_values[int(rng.integers(len(dur_values)))])
        dur = max(1, min(N_MIN, dur))
        wav, used = build_event_waveform(rng, clip_pools[kind], dur)
        if wav is None:
            return None
        powers = convert_waveform_to_minutes(wav)  # (dur, 25) linear
        start_min = int(rng.integers(0, N_MIN - dur + 1))
        frame = np.zeros((N_MIN, N_BAND))
        frame[start_min:start_min + dur] = powers
        if target <= 0.0:
            g = 0.0
        else:
            dom = obs if order == "drop-first" else np.ones(N_MIN, dtype=bool)
            denom = frame[dom].sum()
            if denom <= 0:
                return None
            g = (target / (1.0 - target)) * E_bg[dom].sum() / denom
        E_add = g * frame
    E_mix = E_bg + E_add
    label = float(E_add[obs].sum() / E_mix[obs].sum()) if E_mix[obs].sum() > 0 else 0.0
    X = np.round(lin_to_db(E_mix) * 10.0) / 10.0
    X = X.astype(np.float32)
    X[~obs] = np.nan
    return {
        "X": X, "y": label, "E_bg": E_bg, "E_add": E_add, "obs": obs,
        "site": site, "bg_tag": bg_tag, "kind": kind, "dur": dur,
        "target": target, "order": order, "clips": used,
    }
