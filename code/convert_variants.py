"""Phase 5: passage-conversion variants (PLAN 4.10).

Re-converts the VTUAD clips of data/passages.npz under alternative
minute_band_power-style conversions, to test whether the organisers'
(unverified) conversion differs from the starter notebook's in a way
that changes the added-vessel signature. Every variant keeps the
notebook's band grid and band integration; exactly one axis changes
per variant (see CONFIGS). Output schema is identical to
passages.npz (names / kinds / fs / rel_db), with clip order, clip set
and minute counts identical to the original harvest, so the v3
simulator's split discipline reproduces pool memberships exactly.

Relative-dB normalisation is the original rule (10*log10(power /
clip-median power)); any frequency-independent scaling constant of a
conversion cancels under it, which is why a density-vs-spectrum
scaling variant is pointless and is not run (recorded in
RESULTS_conversion.md).

Usage: python convert_variants.py <config> [<config> ...]
Configs: fam2s (base family: hann 2 s, detrend constant -> writes the
mean reproduction for verification plus the median and dbmean
aggregation variants), rect, rawfft, seg5, seg10, min60, welch1s.
"""
import io
import sys
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import periodogram

from common import DATA

VT = DATA / "vtuad"
MAX_MIN = 60
CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(25))
EDGES = [(c / 10 ** 0.05, c * 10 ** 0.05) for c in CENTRES]

# archive stem -> (kind, cache file) in original harvest order
ARCHIVES = [
    ("vtuad_strict_wav_vessel", "vessel"),
    ("vtuad_wav_pilotvessel", "vessel"),
    ("vtuad_wav_rescue", "vessel"),
    ("vtuad_wav_tanker", "vessel"),
    ("vtuad_wav_sailing", "vessel"),
    ("vtuad_strict_wav_background", "ambient"),
]

# config -> (window, detrend, segment_seconds, hop_seconds, aggregations)
CONFIGS = {
    "fam2s":   ("hann", "constant", 2, 2, ("mean", "median", "dbmean")),
    "rect":    ("boxcar", "constant", 2, 2, ("mean",)),
    "rawfft":  ("hann", False, 2, 2, ("mean",)),
    "seg5":    ("hann", "constant", 5, 5, ("mean",)),
    "seg10":   ("hann", "constant", 10, 10, ("mean",)),
    "min60":   ("hann", "constant", 60, 60, ("mean",)),
    "welch1s": ("hann", "constant", 1, 0.5, ("mean",)),
}
# aggregation -> output tag (fam2s "mean" is the base reproduction check)
AGG_TAG = {"mean": None, "median": "median", "dbmean": "dbmean"}


def clip_list():
    """(zip_path, entry_name, kind, display_name) in passages.npz order."""
    out = []
    for stem, kind in ARCHIVES:
        cache = np.load(DATA / f"passages_{stem}.npz", allow_pickle=True)
        names = list(cache["names"])
        zp = VT / f"{stem}.zip"
        with zipfile.ZipFile(zp) as z:
            by_base = {}
            for n in z.namelist():
                if n.lower().endswith(".wav"):
                    by_base.setdefault(Path(n).name, n)
        for disp in names:
            base = Path(str(disp)).name
            assert base in by_base, f"{base} not in {zp}"
            out.append((zp, by_base[base], kind, str(disp)))
    return out


def segments_of_minute(x, fs, seg_s, hop_s):
    """All segments fully inside one minute of audio."""
    seg_n = int(round(seg_s * fs))
    if hop_s == seg_s:
        n = len(x) // seg_n
        return x[:n * seg_n].reshape(n, seg_n)
    hop_n = int(round(hop_s * fs))
    starts = np.arange(0, len(x) - seg_n + 1, hop_n)
    idx = starts[:, None] + np.arange(seg_n)[None, :]
    return x[idx]


def minute_powers(x_min, fs, window, detrend, seg_s, hop_s):
    """(n_segments, 25) band powers for one minute, notebook integration."""
    segs = segments_of_minute(x_min, fs, seg_s, hop_s)
    frequency, psd = periodogram(
        segs, fs=fs, window=window, detrend=detrend,
        return_onesided=True, scaling="density", axis=1)
    df = frequency[1] - frequency[0]
    power = np.full((len(segs), 25), np.nan)
    for band, (low, high) in enumerate(EDGES):
        bins = (frequency >= low) & (frequency < high)
        if high <= fs / 2 and bins.any():
            power[:, band] = psd[:, bins].sum(axis=1) * df
    return power


def aggregate(seg_power, how):
    if how == "mean":
        return seg_power.mean(axis=0)
    if how == "median":
        return np.median(seg_power, axis=0)
    if how == "dbmean":
        db = 10.0 * np.log10(np.maximum(seg_power, 1e-300))
        return np.power(10.0, db.mean(axis=0) / 10.0)
    raise ValueError(how)


def convert_clip(buf, cfg):
    """WAV bytes -> {aggregation: (minutes,25) power} or None."""
    window, detrend, seg_s, hop_s, aggs = cfg
    info = sf.info(io.BytesIO(buf))
    fs = info.samplerate
    per_agg = {a: [] for a in aggs}
    with sf.SoundFile(io.BytesIO(buf)) as f:
        block_frames = int(10 * 60 * fs)
        got_min = 0
        while got_min < MAX_MIN:
            data = f.read(block_frames, dtype="float64", always_2d=True)
            if len(data) == 0:
                break
            mono = data[:, 0]
            n_min = min(len(mono) // (60 * fs), MAX_MIN - got_min)
            for m in range(n_min):
                x = mono[m * 60 * fs:(m + 1) * 60 * fs]
                sp = minute_powers(x, fs, window, detrend, seg_s, hop_s)
                for a in aggs:
                    per_agg[a].append(aggregate(sp, a))
            got_min += n_min
            if len(data) < block_frames:
                break
    if got_min == 0:
        return None, fs
    return {a: np.stack(v) for a, v in per_agg.items()}, fs


def to_rel_db(power):
    med = np.nanmedian(power[np.isfinite(power)])
    if not np.isfinite(med) or med <= 0:
        return None
    return (10.0 * np.log10(np.maximum(power, 1e-30) / med)).astype(np.float32)


def _save_partial(tag, rel_list, names, kinds, fss):
    out = DATA / f"passages_conv_{tag}.npz"
    np.savez_compressed(out, names=np.array(names),
                        kinds=np.array(kinds), fs=np.array(fss),
                        rel_db=np.array(rel_list, dtype=object))
    return out


def _resume(tag, clips):
    """If a checkpoint for tag exists with a matching clip prefix,
    return (done_count, rel_list, names, kinds, fss)."""
    out = DATA / f"passages_conv_{tag}.npz"
    if not out.exists():
        return 0, [], [], [], []
    d = np.load(out, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    want = [c[3] for c in clips[:len(names)]]
    if names != want or len(names) >= len(clips):
        return 0, [], [], [], []
    print(f"[resume] {tag}: {len(names)} clips already done", flush=True)
    return (len(names), list(d["rel_db"]), names,
            [str(k) for k in d["kinds"]], [int(f) for f in d["fs"]])


def run_config(cfg_name):
    cfg = CONFIGS[cfg_name]
    clips = clip_list()
    print(f"[{cfg_name}] {len(clips)} clips", flush=True)
    # Checkpoint tag per aggregation. Checkpoints are written every 50
    # clips to the output file itself, so a killed run resumes where
    # it stopped. fam2s's mean aggregation is checkpointed under a
    # scratch tag (its deliverable is the verification, not a file).
    if cfg_name == "fam2s":
        ck = {"mean": "_ck_fam2s_mean", "median": "median",
              "dbmean": "dbmean"}
    else:
        ck = {a: cfg_name for a in cfg[4]}
    states = {a: list(_resume(ck[a], clips)) for a in cfg[4]}
    start = min(s[0] for s in states.values())
    results = {a: list(states[a][1]) for a in cfg[4]}
    names = list(states[cfg[4][0]][2])
    kinds = list(states[cfg[4][0]][3])
    fss = list(states[cfg[4][0]][4])
    for i, (zp, entry, kind, disp) in enumerate(clips):
        if i < start:
            continue
        with zipfile.ZipFile(zp) as z:
            buf = z.read(entry)
        try:
            powers, fs = convert_clip(buf, cfg)
        except Exception as e:
            print(f"[{cfg_name}] FAIL {disp}: {type(e).__name__} {e}",
                  flush=True)
            powers, fs = None, None
        assert powers is not None, f"clip {disp} failed under {cfg_name}"
        for a in cfg[4]:
            results[a].append(to_rel_db(powers[a]))
        names.append(disp)
        kinds.append(kind)
        fss.append(fs)
        if (i + 1) % 50 == 0 or i + 1 == len(clips):
            for a in cfg[4]:
                _save_partial(ck[a], results[a], names, kinds, fss)
            print(f"[{cfg_name}] {i + 1}/{len(clips)} (checkpointed)",
                  flush=True)
    if cfg_name == "fam2s":
        _verify_base(results["mean"], names)
    for a in cfg[4]:
        if ck[a] not in ("_ck_fam2s_mean",):
            print(f"[{cfg_name}] wrote passages_conv_{ck[a]}.npz "
                  f"({len(names)} clips)", flush=True)


def _verify_base(rel_list, names):
    """The fam2s mean aggregation must reproduce passages.npz."""
    orig = np.load(DATA / "passages.npz", allow_pickle=True)
    o_rel, o_names = list(orig["rel_db"]), list(orig["names"])
    assert [str(n) for n in o_names] == names, "clip order mismatch"
    worst = 0.0
    for a, b in zip(rel_list, o_rel):
        assert a.shape == b.shape, "clip shape mismatch"
        worst = max(worst, float(np.abs(a - b).max()))
    print(f"[fam2s] base reproduction: max |d rel_db| vs passages.npz "
          f"= {worst:.2e} dB over {len(names)} clips", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run_config(c)
    print("ALL DONE", flush=True)
