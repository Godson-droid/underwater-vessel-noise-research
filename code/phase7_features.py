"""Phase 7 workstream 1: family D — self-referenced background
deviation features, computed from the raw window and concatenated
onto the frozen 549 base features.

Justification (a priori, from Phase 6's diagnosis): the
site-holdout failure is a *calibration* failure — at unseen sites
the natural low-band level sits higher than the training sites'
range, and features built on absolute share levels over-fire.
Family C (Phase 6) referenced the window's spectrum against its
own smooth; family D goes further and references every band
against the window's OWN estimated background spectrum, taken
from its quietest minutes (a joint, per-window estimate), then
featurises (i) the per-band deviation from that self-estimate,
(ii) the spectral shape of the excess above it, and (iii) the
temporal concentration of the low-band excess. For a true-zero
window the excess is background fluctuation: small, spectrally
diffuse, temporally spread. For an added vessel it is large,
low-band concentrated and temporally localised. The features
are scale-free by construction (ratios to the window's own
background), which is exactly the form Phase 6 workstream 1
named as untested: "contrasts and within-window references
rather than absolute share levels".

Distinct from family A (Phase 6, Rejected): A used excess over
the per-band 25th percentile across ALL minutes as a share of
total energy (an absolute-energy quantity). D's reference is the
median spectrum of the 5 quietest minutes, and its primary
quantities are log-ratios and the normalised excess spectrum,
not energy shares.
"""
import numpy as np
from multiprocessing import Pool
from common import DATA, LOW, db_to_lin

N_QUIET = 5


def extra_features_d(win_db):
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

    obs_min = obs.any(axis=1)
    n_obs = int(obs_min.sum())
    if n_obs == 0 or tot <= 0:
        # Degenerate window: zeros of the right length.
        add("xD_dev_db", np.zeros(25))
        add("xD_excess_spec", np.zeros(25))
        add("xD_excess_total", 0.0)
        add("xD_lowbump_dev", 0.0)
        add("xD_band0_dev_contrast", 0.0)
        add("xD_lowex_top3", 0.0)
        add("xD_lowex_maxmin", 0.0)
        add("xD_lowex_nmin_frac", 0.0)
        return np.concatenate(feats), names

    min_tot = E.sum(axis=1)
    quiet_idx = np.argsort(np.where(obs_min, min_tot, np.inf))[:N_QUIET]
    quiet_idx = quiet_idx[obs_min[quiet_idx]]
    # Window's own background spectrum: per-band median energy of
    # its quietest minutes.
    B = np.median(E[quiet_idx], axis=0)                    # (25,)
    B = np.maximum(B, 1e-30)
    band_tot = E.sum(axis=0)                               # (25,)

    # (i) Per-band deviation of the window's mean level from its
    # own background estimate, in dB.
    dev = 10.0 * np.log10(np.maximum(band_tot / (n_obs * B), 1e-30))
    add("xD_dev_db", np.clip(dev, -3.0, 30.0))

    # (ii) Excess spectrum: energy above the self-estimate, and
    # its normalised spectral shape.
    excess = np.maximum(band_tot - n_obs * B, 0.0)
    ex_sum = excess.sum()
    add("xD_excess_spec",
        excess / ex_sum if ex_sum > 0 else np.zeros(25))
    add("xD_excess_total", ex_sum / tot)

    # Contrasts of the deviation profile.
    add("xD_lowbump_dev", dev[:6].mean() - dev[6:16].mean())
    add("xD_band0_dev_contrast", dev[0] - dev[1:4].mean())

    # (iii) Temporal concentration of the low-band excess.
    low_bg = B[LOW].sum()
    lex = np.maximum(E[:, LOW].sum(axis=1) - low_bg, 0.0)
    lex = np.where(obs_min, lex, 0.0)
    lex_sum = lex.sum()
    if lex_sum > 0:
        srt = np.sort(lex)[::-1]
        add("xD_lowex_top3", srt[:3].sum() / lex_sum)
        add("xD_lowex_maxmin", srt[0] / lex_sum)
    else:
        add("xD_lowex_top3", 0.0)
        add("xD_lowex_maxmin", 0.0)
    add("xD_lowex_nmin_frac",
        float((lex > 0.5 * low_bg).sum()) / n_obs if low_bg > 0 else 0.0)
    return np.concatenate(feats), names


def _one(x):
    return extra_features_d(x)[0]


def featurize_d_mp(X, procs=4):
    with Pool(procs) as p:
        F = p.map(_one, [X[i] for i in range(len(X))], chunksize=256)
    return np.asarray(F, dtype=np.float32)


def load_extra_d(tag, X):
    """Cache of family-D features only; tag identifies the set."""
    cache = DATA / f"feat_p7xD_{tag}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"])
    F = featurize_d_mp(X)
    _, names = extra_features_d(X[0])
    np.savez_compressed(cache, F=F, names=np.array(names))
    return F, names
