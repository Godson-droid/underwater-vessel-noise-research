"""Simulator v3: attacks the measured sim-to-real gap (Phase 3, PLAN 4.8).

Changes vs simulate_v2.py, each tied to a Phase 3 measurement:

  1. LABEL / MISSINGNESS CONSTRUCTION (the decisive fix). v1/v2 defined
     the label as the realised share over the full window BEFORE
     missingness dropped minutes. code/label_variants.py measured the
     oracle ceiling (true-background subtraction) of that construction
     at 0.9510 — matching Phase 3's 0.9528 — while the real top score is
     0.9943, so the organisers cannot be using it. Under the alternative
     construction the oracle scores 0.9995. v3 therefore defines:
        label = realised share over the OBSERVED cells of the final
        window (added energy in observed cells / total energy in
        observed cells),
     and mixes the two generative orders that share that label function
     50/50:
        "drop-first": the missingness mask is drawn first and the
                      addition is scaled against the observed background
                      cells only (label = target share);
        "mix-first":  the addition is scaled against the full window
                      (v2-style), the mask applied after, and the label
                      recomputed over observed cells.

  2. BACKGROUNDS WITH WITHHELD CHARACTER. Phase 3 refuted that test
     backgrounds are release windows; the v2 model over-reads real
     windows (mean prediction 0.330 vs a plausible true mean of
     0.13-0.20), consistent with test backgrounds carrying more genuine
     vessel traffic than our clean pool teaches. v3 background draws:
        "clean"   (0.40) screened pool window, level-stratified (as v2);
        "busy"    (0.30) pool window from the top traffic quartile of
                         its split pool (traffic score = the B1-style
                         low-band excess of the window itself — real
                         vessel traffic already in the recording; the
                         label still counts only the controlled addition);
        "mixture" (0.30) two pool windows combined in the energy domain,
                         E = w*E1 + (1-w)*E2, w ~ U(0.3, 0.7), rescaled
                         to the first component's total level — a blended
                         character no single release window has.
     Plus, with probability 0.5, a mild smooth per-band gain jitter
     (+-2 dB) on the background: a different export chain's channel
     (Phase 3 section 3, explanation (b)). And, with probability 0.5,
     CHANGING AMBIENT CONDITIONS over the window (Godson's Phase 4
     brief, point C): a smooth per-minute level envelope (+-2.5 dB)
     plus a spectral character that drifts across the 15 minutes
     (gains interpolate between two smooth +-2 dB colourings) — the
     structured signature of weather, sea state and distant traffic
     evolving during a recording, deliberately NOT uniform noise.

  3. PASSAGE VARIATION PER MINUTE (PLAN 4.7 / Phase 3 section 7). v2
     applied ONE static colouring per event and a smooth +-3.5 dB gain
     envelope. v3: the colouring is time-varying (gains interpolate
     between two independent smooth colourings across the event), and
     the gain profile is a smooth envelope widened to +-5 dB plus iid
     per-minute jitter +-1.5 dB plus, with probability 0.3, a single
     step of +-2..4 dB at a random minute.

  4. GRID ROUNDING. Mixed windows are rounded to the 0.1 dB grid before
     missingness, matching Test.csv's measured quantisation (the Phase 3
     generator did this; v1/v2 did not).

Unchanged from v2: share prior, ambient-addition probability (0.30),
ambient segment slicing, event duration mixture, split discipline
(holdout sites Bird Island + Fish Hoek, period holdout = top 15% by
start within site, 8% val_random carve-out, passage clip split
membership identical to v1/v2), fixed seed 20261116. Test.csv is never
touched here.

Extra sets vs v2: trainA / trainB (disjoint halves of the training
background pool AND of the training passage clips) for the gate's
independent cross-check, and an "oracle" set that also stores the true
composite background of every example so the v3 construction's own
oracle ceiling can be measured.
"""
import json
import os
import numpy as np
from common import DATA, SEED, LOW, db_to_lin, lin_to_db
from simulate_v2 import (StratPool, build_event_lin, draw_duration,
                         draw_share, slice_ambient, warp_time, MISS_P)

HOLDOUT_SITES = {"4 Bird Island", "9 Fish Hoek"}
AMBIENT_P = 0.30
BG_MODE_P = (0.40, 0.70)          # <0.40 clean, <0.70 busy, else mixture
BG_JITTER_P = 0.5

SMOKE = bool(os.environ.get("SIM_V3_SMOKE"))
if SMOKE:
    SIZES = dict(train=800, trainA=400, trainB=400, val_random=200,
                 val_site=200, val_period=200, val_long=200, oracle=100)
else:
    SIZES = dict(train=80000, trainA=40000, trainB=40000, val_random=6000,
                 val_site=6000, val_period=6000, val_long=6000, oracle=2000)
PREFIX = "sim_v3smoke" if SMOKE else "sim_v3"


# ---------------------------------------------------------------- traffic
def traffic_scores(X_db):
    """B1-style low-band excess of each window itself, vectorised.

    Per band, the 25th percentile across the window's 15 minutes is the
    window's own background; the score is the excess energy above it in
    the <=1 kHz bands, divided by total energy. High score = the window
    already contains a vessel-like event.
    """
    out = np.empty(len(X_db), dtype=np.float64)
    for i in range(0, len(X_db), 4096):
        E = db_to_lin(X_db[i:i + 4096].astype(np.float64))
        bg = np.percentile(E, 25, axis=1)                    # (n,25)
        excess = np.maximum(E - bg[:, None, :], 0.0).sum(axis=1)  # (n,25)
        tot = E.sum(axis=(1, 2))
        out[i:i + 4096] = excess[:, LOW].sum(axis=1) / np.maximum(tot, 1e-30)
    return out


# ---------------------------------------------------------------- pool
class V3Pool:
    """Split-disciplined background pool with clean / busy / mixture draws."""

    def __init__(self, X, site, traffic):
        self.X, self.site = X, site
        self.clean = StratPool(X, site)
        thr = np.percentile(traffic, 75)
        self.busy_local = np.where(traffic >= thr)[0]
        self.busy = StratPool(X[self.busy_local], site[self.busy_local])
        self.traffic_thr = float(thr)

    def _jitter(self, E, r):
        steps = r.normal(0, 0.9, size=25).cumsum()
        steps -= np.linspace(steps[0], steps[-1], 25)
        g = np.clip(steps, -2, 2)
        return E * (10.0 ** (g / 10.0))

    def _drift(self, E, r):
        """Changing ambient conditions over the window: smooth per-minute
        level envelope (+-2.5 dB) and a spectral character drifting
        across the minutes (interpolation of two smooth colourings)."""
        env = r.normal(0, 0.8, size=15).cumsum()
        env -= np.linspace(env[0], env[-1], 15)
        env = np.clip(env, -2.5, 2.5)
        def colouring():
            s = r.normal(0, 0.9, size=25).cumsum()
            s -= np.linspace(s[0], s[-1], 25)
            return np.clip(s, -2, 2)
        c1, c2 = colouring(), colouring()
        t = np.linspace(0.0, 1.0, 15)[:, None]
        g = env[:, None] + (1.0 - t) * c1[None, :] + t * c2[None, :]
        return E * (10.0 ** (g / 10.0))

    def draw_background(self, r):
        """Returns (E_lin (15,25) float64, site, mode)."""
        u = r.random()
        if u < BG_MODE_P[0]:
            j = self.clean.draw(r)
            E, site, mode = db_to_lin(self.X[j]), str(self.site[j]), "clean"
        elif u < BG_MODE_P[1]:
            j = int(self.busy_local[self.busy.draw(r)])
            E, site, mode = db_to_lin(self.X[j]), str(self.site[j]), "busy"
        else:
            j1 = self.clean.draw(r)
            j2 = self.clean.draw(r)
            w = r.uniform(0.3, 0.7)
            E1, E2 = db_to_lin(self.X[j1]), db_to_lin(self.X[j2])
            E = w * E1 + (1.0 - w) * E2
            E *= E1.sum() / E.sum()          # first component's total level
            site, mode = str(self.site[j1]), "mixture"
        if r.random() < BG_JITTER_P:
            E = self._jitter(E, r)
        if r.random() < BG_JITTER_P:
            E = self._drift(E, r)
        return E, site, mode


# ---------------------------------------------------------------- event
def smooth_colour(r):
    steps = r.normal(0, 1.8, size=25).cumsum()
    steps -= np.linspace(steps[0], steps[-1], 25)
    return np.clip(steps, -7, 7)


def colour_tv(seg_lin, r):
    """Time-varying channel colouring: gains interpolate between two
    independent smooth colourings across the event's minutes."""
    L = seg_lin.shape[0]
    g1, g2 = smooth_colour(r), smooth_colour(r)
    t = np.linspace(0.0, 1.0, L)[:, None]
    g = (1.0 - t) * g1[None, :] + t * g2[None, :]
    return seg_lin * (10.0 ** (g / 10.0))


def gain_profile(seg_lin, r):
    """Per-minute gain: smooth envelope +-5 dB + iid jitter +-1.5 dB +
    (p=0.3) one step of +-2..4 dB at a random minute."""
    L = seg_lin.shape[0]
    if L <= 1:
        return seg_lin
    steps = r.normal(0, 1.2, size=L).cumsum()
    steps -= np.linspace(steps[0], steps[-1], L)
    g = np.clip(steps, -5, 5) + r.uniform(-1.5, 1.5, size=L)
    if r.random() < 0.3:
        k = int(r.integers(1, L))
        g[k:] += r.choice([-1.0, 1.0]) * r.uniform(2, 4)
    g = np.clip(g, -8, 8)
    return seg_lin * (10.0 ** (g / 10.0))[:, None]


def build_event_v3(blocks, r, force_long=False):
    """Assemble, vary and place an event. Returns (placed (15,25) linear
    unscaled, duration)."""
    D = r.uniform(12.0, 15.01) if force_long else draw_duration(r)
    w = r.uniform(0.5, 2.0)
    src_len = int(np.clip(round(D / w), 1, 40))
    seg = build_event_lin(blocks, src_len, r)
    if r.random() < 0.30:
        seg = seg[::-1].copy()
    new_len = int(np.clip(round(seg.shape[0] * w), 1, 15))
    seg = warp_time(seg, new_len, r)
    seg = colour_tv(seg, r)
    seg = gain_profile(seg, r)
    off = int(r.integers(0, 15 - new_len + 1))
    placed = np.zeros((15, 25))
    placed[off:off + new_len] = seg
    return placed, new_len


def draw_mask(r):
    drops = [m for m, p in MISS_P.items() if r.random() < p]
    if len(drops) > 3:
        drops = list(r.choice(drops, size=3, replace=False))
    mask = np.ones(15, dtype=bool)
    mask[drops] = False
    return mask


# ---------------------------------------------------------------- example
def make_example(pool, clips, r, force_long=False, want_bg=False):
    E_bg, site, mode = pool.draw_background(r)
    B_db = lin_to_db(E_bg)
    s = draw_share(r)
    if force_long:
        while s <= 0:
            s = draw_share(r)
    mask = draw_mask(r)
    order = "drop-first" if r.random() < 0.5 else "mix-first"
    kind, dur, label = "none", 0, 0.0
    E = E_bg
    if s > 0:
        use_ambient = r.random() < AMBIENT_P
        blocks = clips["ambient"] if use_ambient else clips["vessel"]
        if len(blocks) == 0:
            blocks, use_ambient = clips["vessel"], False
        placed, dur = build_event_v3(blocks, r, force_long)
        obs = mask[:, None]
        bg_obs = np.where(obs, E_bg, 0.0).sum()
        p_obs = np.where(obs, placed, 0.0).sum()
        if p_obs <= 0.0:
            # The event fell entirely inside dropped minutes: nothing of
            # it is observable, so the observed window is a share-0
            # window. (Guard added after the first generation run
            # produced 34 NaN labels via alpha = s/p_obs = inf; those
            # files were discarded and regenerated.)
            kind = "ambient" if use_ambient else "vessel"
        else:
            if order == "drop-first":
                alpha = s * bg_obs / ((1.0 - s) * p_obs)
            else:
                alpha = s * E_bg.sum() / ((1.0 - s) * placed.sum())
            E = E_bg + alpha * placed
            label = float(alpha * p_obs / (bg_obs + alpha * p_obs))
            kind = "ambient" if use_ambient else "vessel"
    win = np.round(lin_to_db(E) * 10.0) / 10.0
    win[~mask, :] = np.nan
    out = (win.astype(np.float32), label, kind, dur, mode, order, site)
    if want_bg:
        out = out + (B_db.astype(np.float32),)
    return out


def gen(n, pool, clips, r, force_long=False, want_bg=False):
    X = np.empty((n, 15, 25), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    sites = np.empty(n, dtype=object)
    kinds = np.empty(n, dtype=object)
    durs = np.empty(n, dtype=np.int32)
    modes = np.empty(n, dtype=object)
    orders = np.empty(n, dtype=object)
    B = np.empty((n, 15, 25), dtype=np.float32) if want_bg else None
    for i in range(n):
        res = make_example(pool, clips, r, force_long, want_bg)
        X[i], y[i], kinds[i], durs[i], modes[i], orders[i], sites[i] = res[:7]
        if want_bg:
            B[i] = res[7]
    return dict(X=X, y=y, site=sites, kind=kinds, dur=durs, mode=modes,
                order=orders, **(dict(B=B) if want_bg else {}))


# ---------------------------------------------------------------- main
def main():
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    BX, BS, BT = bg["X"], bg["site"], bg["start"]
    pas = np.load(DATA / "passages.npz", allow_pickle=True)
    rel, kinds_arr = list(pas["rel_db"]), pas["kinds"]
    vessel = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "vessel"]
    ambient = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "ambient"]
    print(f"passages: {len(vessel)} vessel, {len(ambient)} ambient", flush=True)

    print("computing traffic scores...", flush=True)
    traffic = traffic_scores(BX)
    print(f"traffic score: median {np.median(traffic):.3f}, "
          f"p75 {np.percentile(traffic, 75):.3f}, max {traffic.max():.3f}", flush=True)

    # Passage split: identical first RNG calls to v1/v2 -> identical
    # train/val clip membership.
    r_split = np.random.default_rng(SEED)
    def split_idx(m):
        idx = r_split.permutation(m)
        k = max(1, int(0.85 * m))
        return idx[:k], idx[k:]
    v_tr_i, v_va_i = split_idx(len(vessel))
    a_tr_i, a_va_i = split_idx(len(ambient))
    v_tr = [vessel[i] for i in v_tr_i]
    v_va = [vessel[i] for i in v_va_i]
    a_tr = [ambient[i] for i in a_tr_i]
    a_va = [ambient[i] for i in a_va_i]

    r = np.random.default_rng(SEED)
    a_tr_seg = [s for c in a_tr for s in slice_ambient(c, r)]
    a_va_seg = [s for c in a_va for s in slice_ambient(c, r)]
    print(f"ambient segments: {len(a_tr_seg)} train / {len(a_va_seg)} val", flush=True)
    clips_tr = dict(vessel=v_tr, ambient=a_tr_seg)
    clips_va = dict(vessel=v_va, ambient=a_va_seg)

    # Cross-check clip halves (disjoint within the training clips).
    def halve(items, r):
        idx = r.permutation(len(items))
        h = len(items) // 2
        return [items[i] for i in idx[:h]], [items[i] for i in idx[h:]]
    vA, vB = halve(v_tr, r)
    aA, aB = halve(a_tr_seg, r)
    clips_A = dict(vessel=vA, ambient=aA)
    clips_B = dict(vessel=vB, ambient=aB)

    # Background split discipline: identical masks to v1/v2.
    is_hold = np.array([s in HOLDOUT_SITES for s in BS])
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
    # Cross-check background halves (disjoint within the training pool).
    perm = r.permutation(len(tr_idx))
    half = len(tr_idx) // 2
    trA_idx, trB_idx = tr_idx[perm[:half]], tr_idx[perm[half:]]

    def P(ii):
        return V3Pool(BX[ii], BS[ii], traffic[ii])
    pools = dict(
        train=(P(tr_idx), clips_tr, False, False),
        trainA=(P(trA_idx), clips_A, False, False),
        trainB=(P(trB_idx), clips_B, False, False),
        val_random=(P(vr_idx), clips_va, False, False),
        val_site=(P(np.where(is_hold)[0]), clips_va, False, False),
        val_period=(P(np.where(is_late)[0]), clips_va, False, False),
        val_long=(P(vr_idx), clips_va, True, False),
        oracle=(P(vr_idx), clips_va, False, True),
    )
    meta = dict(sizes=SIZES, pools={k: int(len(v[0].X)) for k, v in pools.items()},
                traffic_p75=float(np.percentile(traffic, 75)), sets={})
    for name, (pool, pc, fl, wb) in pools.items():
        d = gen(SIZES[name], pool, pc, r, force_long=fl, want_bg=wb)
        np.savez_compressed(DATA / f"{PREFIX}_{name}.npz", **d)
        modes, cnts = np.unique(d["mode"], return_counts=True)
        orders, ocnts = np.unique(d["order"], return_counts=True)
        meta["sets"][name] = dict(
            label_mean=float(d["y"].mean()), zero_frac=float((d["y"] == 0).mean()),
            modes={str(m): float(c / len(d["y"])) for m, c in zip(modes, cnts)},
            orders={str(o): float(c / len(d["y"])) for o, c in zip(orders, ocnts)})
        print(f"{name}: {d['X'].shape} from {len(pool.X)} bg windows; "
              f"label mean {d['y'].mean():.3f}, zero {(d['y'] == 0).mean():.3f}; "
              f"modes {meta['sets'][name]['modes']}", flush=True)
    (DATA / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
