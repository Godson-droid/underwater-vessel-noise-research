"""Phase 6 workstream 3: extra feature families, computed from the
raw window and concatenated onto the frozen 549 base features.
Each family is justified a priori (see RESULTS_phase6_model.md
workstream 3) and tested one family at a time on identical windows.

FAM A "per-band excess" (50): the label is an energy share; an added
  passage appears as energy in excess of the window's own per-band
  background level. B1 aggregates exactly this over <=1 kHz only.
  Per band: excess over the band's 25th-percentile minute level,
  as (i) share of total window energy, (ii) fraction of the band's
  own energy. Background-relative by construction.
FAM B "temporal concentration" (25): a passing vessel concentrates
  a band's energy into a few minutes; ambient is diffuse. Per band:
  the loudest 3 minutes' share of the band's window energy.
  Scale-free shape information the base set lacks per band.
FAM C "spectral shape residuals" (28): site identity lives in the
  smooth spectral tilt of a recording; an added vessel is a
  localised low-frequency bump on top of it. Residual of the
  window's mean dB spectrum after a 5-band moving-average smooth
  (25), plus band-0 contrasts against bands 1-3 in dB and in share,
  plus the mean residual over bands 0-5 (3).
"""
import numpy as np
from multiprocessing import Pool
from common import DATA, db_to_lin, lin_to_db

FAMS = ("A", "B", "C")


def extra_features(win_db, fams=("A",)):
    win = np.asarray(win_db, dtype=np.float64)
    obs = np.isfinite(win)
    E = np.where(obs, db_to_lin(win), 0.0)
    tot = E.sum()
    feats, names = [], []

    def add(name, arr):
        a = np.asarray(arr, dtype=np.float64).ravel()
        feats.append(a)
        if a.size == 1:
            names.append(name)
        else:
            names.extend(f"{name}_{i}" for i in range(a.size))

    if "A" in fams:
        with np.errstate(all="ignore"):
            q25 = np.where(obs.any(0),
                           np.nanpercentile(np.where(obs, E, np.nan),
                                            25, axis=0), 0.0)
        excess = np.maximum(E - q25[None, :], 0.0).sum(axis=0)
        band_tot = E.sum(axis=0)
        add("xA_excess_share", excess / max(tot, 1e-30))
        add("xA_excess_frac",
            np.where(band_tot > 0, excess / np.maximum(band_tot, 1e-30),
                     0.0))
    if "B" in fams:
        band_tot = E.sum(axis=0)
        top3 = np.sort(E, axis=0)[-3:, :].sum(axis=0)
        add("xB_top3_share",
            np.where(band_tot > 0, top3 / np.maximum(band_tot, 1e-30),
                     0.0))
    if "C" in fams:
        band_tot = E.sum(axis=0)
        with np.errstate(all="ignore"):
            spec = np.where(band_tot > 0,
                            lin_to_db(np.maximum(band_tot, 1e-30)), 0.0)
        pad = np.pad(spec, 2, mode="edge")
        smooth = np.convolve(pad, np.ones(5) / 5.0, mode="valid")
        resid = spec - smooth
        add("xC_spec_resid", resid)
        band_share = band_tot / max(tot, 1e-30)
        add("xC_band0_db_contrast", spec[0] - spec[1:4].mean())
        add("xC_band0_share_contrast",
            band_share[0] - band_share[1:4].mean())
        add("xC_lowbump", resid[:6].mean())
    return np.concatenate(feats), names


def _one(args):
    x, fams = args
    return extra_features(x, fams)[0]


def featurize_extra_mp(X, fams, procs=4):
    with Pool(procs) as p:
        F = p.map(_one, [(X[i], fams) for i in range(len(X))],
                  chunksize=256)
    return np.asarray(F, dtype=np.float32)


def load_extra(tag, X, fams):
    """Cache of extra features only; tag identifies the window set."""
    key = "".join(fams)
    cache = DATA / f"feat_p6x{key}_{tag}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"])
    F = featurize_extra_mp(X, fams)
    _, names = extra_features(X[0], fams)
    np.savez_compressed(cache, F=F, names=np.array(names))
    return F, names
