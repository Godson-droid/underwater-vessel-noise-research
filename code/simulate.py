"""Simulator: manufacture labelled mixes exactly as the organisers did.

Backgrounds: screened SA windows (data/bg_windows.npz). Passages: VTUAD
v1.3.2 clips converted to relative-dB minute x band summaries
(data/passages.npz). Mixing is in the LINEAR energy domain; the label is the
realised share of added energy over the full 15-min window, computed before
the test missingness pattern is synthesised on top.

Variation mirrors notebook cell 9: absolute gain discarded (passage stored
relative, then re-scaled to hit the target share exactly), time trajectories
varied (reversal, time-warp, random placement), relative channel varied
(smooth random per-band colouring). With probability AMBIENT_P, the "added"
component is an ambient clip segment instead of a vessel clip — the
organisers warn that passage references carry their own ambient, and that
energy still counts as added.

Splits (all disjoint at the background-window AND passage-clip level):
  train      : 8 sites, early periods, train passage clips
  val_random : same sites/periods as train but unseen windows and clips
  val_site   : 2 held-out sites (Bird Island, Fish Hoek)
  val_period : late periods of the train sites
Fixed seed throughout.
"""
import numpy as np
from common import DATA, SEED, db_to_lin, lin_to_db

rng = np.random.default_rng(SEED)

HOLDOUT_SITES = {"4 Bird Island", "9 Fish Hoek"}
P_ZERO = 0.25            # mass point at share exactly 0
AMBIENT_P = 0.30         # of passage events, use an ambient clip as the addition
# Test missingness (Phase 0 section 3, measured): whole minutes only.
MISS_P = {0: 0.416, 14: 0.067, 1: 0.0027, 13: 0.0027, 7: 0.0007, 8: 0.0007, 12: 0.0007}

SIZES = dict(train=60000, val_random=6000, val_site=6000, val_period=6000)


def draw_share(r):
    if r.random() < P_ZERO:
        return 0.0
    u = r.random()
    if u < 0.30:
        return r.uniform(0.0, 0.05)
    if u < 0.65:
        return r.uniform(0.05, 0.20)
    if u < 0.85:
        return r.uniform(0.20, 0.50)
    return r.uniform(0.50, 0.95)


def warp_time(seg, new_len, r):
    """Linear-interpolate a (L,25) linear-energy segment to new_len minutes."""
    if new_len == seg.shape[0]:
        return seg
    x_old = np.linspace(0, 1, seg.shape[0])
    x_new = np.linspace(0, 1, new_len)
    return np.stack([np.interp(x_new, x_old, seg[:, b]) for b in range(25)], axis=1)


def colour(seg_lin, r):
    """Smooth random per-band gain (relative channel variation), +-4 dB."""
    steps = r.normal(0, 1.2, size=25).cumsum()
    steps -= np.linspace(steps[0], steps[-1], 25)
    g = np.clip(steps, -4, 4)
    return seg_lin * (10.0 ** (g / 10.0))


def make_example(bg_db, clips, r):
    E = db_to_lin(bg_db)                       # (15,25)
    s = draw_share(r)
    kind = "none"
    if s > 0:
        use_ambient = r.random() < AMBIENT_P
        pool = clips["ambient"] if use_ambient else clips["vessel"]
        if len(pool) == 0:
            pool = clips["vessel"]
            use_ambient = False
        clip = pool[r.integers(len(pool))]
        M = clip.shape[0]
        L = int(min(M, r.integers(4, 16)))
        start = int(r.integers(0, M - L + 1)) if M > L else 0
        seg = db_to_lin(clip[start:start + L])
        if r.random() < 0.30:
            seg = seg[::-1].copy()             # trajectory reversal
        max_len = 15
        new_len = int(np.clip(round(L * r.uniform(0.7, 1.4)), 1, max_len))
        seg = warp_time(seg, new_len, r)
        seg = colour(seg, r)
        off = int(r.integers(0, 15 - new_len + 1))
        p_tot = seg.sum()
        e_tot = E.sum()
        alpha = s * e_tot / ((1.0 - s) * p_tot)
        E[off:off + new_len] += alpha * seg
        kind = "ambient" if use_ambient else "vessel"
        s = float(alpha * p_tot / (e_tot + alpha * p_tot))   # realised share
    win = lin_to_db(E).astype(np.float64)
    # Synthesise the test missingness pattern (whole minutes).
    drops = [m for m, p in MISS_P.items() if r.random() < p]
    if len(drops) > 3:
        drops = list(r.choice(drops, size=3, replace=False))
    for m in drops:
        win[m, :] = np.nan
    return win.astype(np.float32), s, kind


def gen(n, bg_X, bg_site, clips, r):
    X = np.empty((n, 15, 25), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    sites = np.empty(n, dtype=object)
    kinds = np.empty(n, dtype=object)
    for i in range(n):
        j = int(r.integers(len(bg_X)))
        X[i], y[i], kinds[i] = make_example(bg_X[j], clips, r)
        sites[i] = bg_site[j]
    return X, y, sites, kinds


def main():
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    BX, BS, BT = bg["X"], bg["site"], bg["start"]
    pas = np.load(DATA / "passages.npz", allow_pickle=True)
    rel = list(pas["rel_db"])
    kinds = pas["kinds"]
    vessel = [rel[i] for i in range(len(rel)) if kinds[i] == "vessel"]
    ambient = [rel[i] for i in range(len(rel)) if kinds[i] == "ambient"]
    print(f"passages: {len(vessel)} vessel, {len(ambient)} ambient")

    r = np.random.default_rng(SEED)
    # Passage split: 85% train clips / 15% validation clips (disjoint).
    def split_clips(pool):
        idx = r.permutation(len(pool))
        k = max(1, int(0.85 * len(pool)))
        return [pool[i] for i in idx[:k]], [pool[i] for i in idx[k:]]
    v_tr, v_va = split_clips(vessel)
    a_tr, a_va = split_clips(ambient) if len(ambient) >= 2 else (ambient, ambient)
    clips_tr = dict(vessel=v_tr, ambient=a_tr)
    clips_va = dict(vessel=v_va, ambient=a_va)

    is_hold = np.array([s in HOLDOUT_SITES for s in BS])
    # Period split within train sites: per-site 85th percentile of starts.
    is_late = np.zeros(len(BS), dtype=bool)
    for s in np.unique(BS[~is_hold]):
        m = (~is_hold) & (BS == s)
        if m.sum() > 100:
            is_late[m] = BT[m] >= np.percentile(BT[m], 85)
    tr_mask = ~is_hold & ~is_late
    idx = np.where(tr_mask)[0]
    r.shuffle(idx)
    n_vr = int(0.08 * len(idx))
    vr_idx, tr_idx = idx[:n_vr], idx[n_vr:]
    pools = dict(
        train=(BX[tr_idx], BS[tr_idx], clips_tr),
        val_random=(BX[vr_idx], BS[vr_idx], clips_va),
        val_site=(BX[is_hold], BS[is_hold], clips_va),
        val_period=(BX[is_late], BS[is_late], clips_va),
    )
    for name, (px, ps, pc) in pools.items():
        X, y, sites, kinds_out = gen(SIZES[name], px, ps, pc, r)
        np.savez_compressed(DATA / f"sim_{name}.npz", X=X, y=y, site=sites, kind=kinds_out)
        print(f"{name}: {X.shape} from {len(px)} bg windows; "
              f"share mean {y.mean():.3f}, zero frac {(y == 0).mean():.3f}")


if __name__ == "__main__":
    main()
