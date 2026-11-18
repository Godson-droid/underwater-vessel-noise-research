"""Phase 3 synthetic generator: mirrors the organisers' stated process.

Background = a real screened pool window (identity recorded). Addition =
a VTUAD passage segment (vessel 70% / ambient 30%, as in our simulator),
time-warped, channel-coloured, scaled in the linear domain to an exact
target share, added, converted back to dB, ROUNDED to the 0.1 dB grid
(the measured test quantisation), then the measured test missingness
pattern is applied. The label is the realised share over the full window
before missingness, exactly as the organisers define it.
"""
import numpy as np
from common import DATA, db_to_lin, lin_to_db
from simulate import warp_time, colour, MISS_P

_rng_bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
BG_X, BG_SITE, BG_START = _rng_bg["X"], _rng_bg["site"], _rng_bg["start"]
_pas = np.load(DATA / "passages.npz", allow_pickle=True)
_rel, _kinds = list(_pas["rel_db"]), _pas["kinds"]
VESSEL = [_rel[i] for i in range(len(_rel)) if _kinds[i] == "vessel"]
AMBIENT = [_rel[i] for i in range(len(_rel)) if _kinds[i] == "ambient"]

EVENT_MIX = [(0.45, 12, 15), (0.25, 8, 12), (0.20, 4, 8), (0.10, 1, 4)]


def make_synth(bg_idx, target_share, r):
    """Returns (window_db float64 with NaN, info dict)."""
    bg_db = BG_X[bg_idx].astype(np.float64)
    E = db_to_lin(bg_db)
    s = float(target_share)
    kind = "none"
    if s > 0:
        use_ambient = r.random() < 0.30
        pool = AMBIENT if use_ambient else VESSEL
        clip = pool[r.integers(len(pool))]
        u = r.random()
        acc = 0.0
        lo, hi = 1, 4
        for p, a, b in EVENT_MIX:
            acc += p
            if u <= acc:
                lo, hi = a, b
                break
        ev_len = int(r.integers(lo, hi + 1))
        M = clip.shape[0]
        L = int(min(M, ev_len))
        st = int(r.integers(0, M - L + 1)) if M > L else 0
        seg = db_to_lin(clip[st:st + L])
        if r.random() < 0.30:
            seg = seg[::-1].copy()
        seg = warp_time(seg, ev_len, r)
        seg = colour(seg, r)
        off = int(r.integers(0, 15 - ev_len + 1))
        p_tot, e_tot = seg.sum(), E.sum()
        alpha = s * e_tot / ((1.0 - s) * p_tot)
        E[off:off + ev_len] += alpha * seg
        kind = "ambient" if use_ambient else "vessel"
        s = float(alpha * p_tot / (e_tot + alpha * p_tot))
    win = lin_to_db(E)
    win = np.round(win * 10.0) / 10.0          # organisers' 0.1 dB grid
    drops = [m for m, p in MISS_P.items() if r.random() < p]
    if len(drops) > 3:
        drops = list(r.choice(drops, size=3, replace=False))
    for m in drops:
        win[m, :] = np.nan
    info = dict(bg_idx=int(bg_idx), bg_site=str(BG_SITE[bg_idx]),
                bg_start=int(BG_START[bg_idx]), true_share=s, kind=kind,
                target_share=float(target_share))
    return win, info


def est_from_bg(T_db, B_db):
    """Clipped-residual share estimate from a candidate background."""
    obs = np.isfinite(T_db)
    Tl = db_to_lin(np.where(obs, T_db, 0.0))
    Bl = db_to_lin(B_db)
    resid = np.maximum(Tl - np.where(obs, Bl, 0.0), 0.0)
    tot = Tl.sum()
    return float(resid.sum() / tot) if tot > 0 else 0.0
