"""Shared constants and feature engineering for the Zindi vessel-noise simulator.

Used identically by training, validation and test inference (fixed per-window
transforms only — nothing is ever fitted on Test.csv).
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 20261116

# The 25 common bands, in the parquet export naming (== Test.csv grid
# 10^(1.5+0.1k), k=0..24, 31.6 Hz .. 7943.3 Hz).
BANDS = ["31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5",
         "199.5", "251.2", "316.2", "398.1", "501.2", "631", "794.3", "1000",
         "1258.9", "1584.9", "1995.3", "2511.9", "3162.3", "3981.1", "5011.9",
         "6309.6", "7943.3"]
BAND_COLS = [f"TOL_{b}" for b in BANDS]
CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(25))
LOW = CENTRES <= 1000.0          # 16 bands where vessel noise concentrates
N_MIN, N_BAND = 15, 25


def db_to_lin(db):
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def lin_to_db(lin):
    return 10.0 * np.log10(np.maximum(lin, 1e-30))


def b1_statistic(win_db):
    """The B1 low-band excess statistic (Phase 1), as a single feature.

    Fixed transform, nothing fitted: per band, 25th percentile across the
    window's observed minutes is the background; excess energy above it,
    summed over bands <= 1 kHz, divided by total window energy.
    """
    E = db_to_lin(win_db)
    obs = np.isfinite(win_db)
    E = np.where(obs, E, np.nan)
    with np.errstate(all="ignore"):
        bg = np.nanpercentile(E, 25, axis=0)          # (25,)
        excess = np.nansum(np.maximum(E - bg, 0.0), axis=0)  # (25,)
        total = np.nansum(E)
    if not np.isfinite(total) or total <= 0:
        return 0.0
    return float(np.clip(excess[LOW].sum() / total, 0.0, 1.0))


def featurize(win_db):
    """(15,25) dB window (NaN = missing minute) -> fixed-length feature vector.

    All features are scale-free (shares, ratios, dB differences) except a few
    absolute-level summaries, because the organisers discarded absolute gain
    for the passage component (notebook cell 9).
    """
    win = np.asarray(win_db, dtype=np.float64)
    obs = np.isfinite(win)
    E = np.where(obs, db_to_lin(win), 0.0)
    tot = E.sum()
    feats = []
    names = []

    def add(name, arr):
        a = np.asarray(arr, dtype=np.float64).ravel()
        feats.append(a)
        if a.size == 1:
            names.append(name)
        else:
            names.extend(f"{name}_{i}" for i in range(a.size))

    if tot <= 0:
        # Degenerate window: return zeros of the right length via a template.
        E = np.full_like(E, 1e-30)
        tot = E.sum()

    share_cell = E / tot                                   # (15,25)
    add("cell_share", share_cell)
    band_share = share_cell.sum(axis=0)                    # (25,)
    add("band_share", band_share)
    min_share = share_cell.sum(axis=1)                     # (15,)
    add("minute_share", min_share)

    lowE = E[:, LOW].sum(axis=1)
    min_tot = E.sum(axis=1)
    low_frac = np.where(min_tot > 0, lowE / np.maximum(min_tot, 1e-30), 0.0)
    add("minute_low_frac", low_frac)                       # (15,)
    add("low_frac_total", lowE.sum() / tot)

    # Per-band dB temporal stats over observed minutes.
    with np.errstate(all="ignore"):
        band_mean = np.where(obs.any(0), np.nanmean(np.where(obs, win, np.nan), axis=0), 0.0)
        band_max = np.where(obs.any(0), np.nanmax(np.where(obs, win, np.nan), axis=0), 0.0)
        band_min = np.where(obs.any(0), np.nanmin(np.where(obs, win, np.nan), axis=0), 0.0)
        band_std = np.where(obs.any(0), np.nanstd(np.where(obs, win, np.nan), axis=0), 0.0)
    ref = np.nanmedian(win[obs]) if obs.any() else 0.0
    add("band_mean_db_rel", band_mean - ref)               # (25,)
    add("band_range_db", band_max - band_min)              # (25,)
    add("band_std_db", band_std)                           # (25,)

    # Per-minute level relative to window median, and spectral centroid.
    min_db = np.where(obs.any(1), lin_to_db(np.maximum(min_tot, 1e-30)), np.nan)
    add("minute_db_rel", np.where(np.isfinite(min_db), min_db - np.nanmedian(min_db[np.isfinite(min_db)]), 0.0))
    with np.errstate(all="ignore"):
        cent = np.where(min_tot > 0, (share_cell * CENTRES).sum(axis=1) / np.maximum(min_share, 1e-30), 0.0)
    add("minute_centroid_frac", cent / CENTRES.max())      # (15,)

    # Trajectory of the low-band fraction: slope and curvature over minutes.
    t = np.arange(N_MIN, dtype=np.float64)
    ok = obs.any(1)
    if ok.sum() >= 3:
        sl, ic = np.polyfit(t[ok], low_frac[ok], 1)
        mid = low_frac[ok][len(low_frac[ok]) // 2]
    else:
        sl, ic, mid = 0.0, 0.0, 0.0
    add("low_frac_slope", sl)
    add("low_frac_intercept", ic)
    add("low_frac_mid", mid)
    add("low_frac_max", low_frac.max())
    add("low_frac_range", low_frac.max() - low_frac.min())

    add("b1", b1_statistic(win))
    add("max_minute_share", min_share.max())
    add("top3_minute_share", np.sort(min_share)[-3:].sum())
    add("total_db", lin_to_db(tot))
    add("n_missing_minutes", (~ok).sum())
    add("missing_m01", 0.0 if ok[0] else 1.0)
    add("missing_m15", 0.0 if ok[14] else 1.0)
    # Energy concentration: share of the single loudest cell.
    add("max_cell_share", share_cell.max())
    return np.concatenate(feats), names


def featurize_batch(X_db):
    """X_db: (N,15,25) -> (N,F) float32, list of feature names."""
    F, names = [], None
    for i in range(X_db.shape[0]):
        f, names = featurize(X_db[i])
        F.append(f)
    return np.asarray(F, dtype=np.float32), names
