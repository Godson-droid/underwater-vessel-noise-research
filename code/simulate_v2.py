"""Simulator v2: same construction as v1, with longer passage events, wider
variation, sliced ambient references, and level-stratified backgrounds.

Changes vs simulate.py (v1), motivated by RESULTS_sim_v1.md:
  1. LONG EVENTS. v1 placed one short clip segment (pool median 3 min) once
     per window. v2 assembles events that span most or all of the 15-min
     window: the longest clips are used whole, and clips are concatenated
     end to end (1-minute boundary blend — the data are minute summaries,
     so that is the finest fade that exists) until the target duration is
     reached. Duration mixture (final, post-warp): 45% U(12,15), 25%
     U(8,12), 20% U(4,8), 10% U(1,4) minutes. Realised durations are
     recorded per example and summarised in data/sim_v2_meta.json.
  2. WIDER VARIATION. Time-warp x0.5-2.0 (v1: x0.7-1.4); per-band channel
     colouring +-7 dB (v1: +-4 dB); NEW per-piece relative gain jitter
     +-3 dB inside concatenated events, and a NEW smooth per-minute gain
     envelope (+-3.5 dB) over each event (approach/recede profile).
     Reversal probability unchanged (0.3).
  3. AMBIENT SEGMENTS. The 16 ambient clips are sliced (after the v1
     clip-level split, so train/val disjointness is preserved) into
     non-overlapping 4-8 min segments (~85 blocks); ambient additions are
     assembled from segments exactly like vessel events.
  4. LEVEL-STRATIFIED BACKGROUNDS. Background windows are sampled
     uniformly across total-level deciles of their pool (v1 sampled
     uniformly over windows, which over-weights mid-level Richards Bay
     windows), so quiet backgrounds are well represented.

Unchanged from v1 (comparability): share prior, missingness pattern,
ambient-addition probability (0.30), split discipline — the split code and
RNG call order in main() are identical to v1, so the background-window and
passage-clip pool memberships are exactly v1's. Set sizes identical, plus
a new val_long set (6,000 mixes from the val_random background pool and
validation clips, every example carrying a forced long event, share > 0).

Fixed seed throughout. Test.csv is never touched here.
"""
import json
import os
import numpy as np
from common import DATA, SEED, db_to_lin, lin_to_db

HOLDOUT_SITES = {"4 Bird Island", "9 Fish Hoek"}
P_ZERO = 0.25
AMBIENT_P = 0.30
MISS_P = {0: 0.416, 14: 0.067, 1: 0.0027, 13: 0.0027, 7: 0.0007, 8: 0.0007, 12: 0.0007}

SMOKE = bool(os.environ.get("SIM_V2_SMOKE"))
SIZES = dict(train=400, val_random=200, val_site=200, val_period=200, val_long=200) if SMOKE \
    else dict(train=60000, val_random=6000, val_site=6000, val_period=6000, val_long=6000)
PREFIX = "sim_v2smoke" if SMOKE else "sim_v2"


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


def draw_duration(r):
    """Target final (post-warp) event duration in minutes."""
    u = r.random()
    if u < 0.45:
        return r.uniform(12.0, 15.01)
    if u < 0.70:
        return r.uniform(8.0, 12.0)
    if u < 0.90:
        return r.uniform(4.0, 8.0)
    return r.uniform(1.0, 4.0)


def warp_time(seg, new_len, r):
    if new_len == seg.shape[0]:
        return seg
    x_old = np.linspace(0, 1, seg.shape[0])
    x_new = np.linspace(0, 1, new_len)
    return np.stack([np.interp(x_new, x_old, seg[:, b]) for b in range(25)], axis=1)


def colour(seg_lin, r):
    """Smooth random per-band gain, +-7 dB (v1: +-4 dB)."""
    steps = r.normal(0, 1.8, size=25).cumsum()
    steps -= np.linspace(steps[0], steps[-1], 25)
    g = np.clip(steps, -7, 7)
    return seg_lin * (10.0 ** (g / 10.0))


def gain_envelope(seg_lin, r):
    """Smooth per-minute relative gain over the event, +-3.5 dB."""
    L = seg_lin.shape[0]
    if L <= 1:
        return seg_lin
    steps = r.normal(0, 0.9, size=L).cumsum()
    steps -= np.linspace(steps[0], steps[-1], L)
    g = np.clip(steps, -3.5, 3.5)
    return seg_lin * (10.0 ** (g / 10.0))[:, None]


def build_event_lin(blocks, target_len, r):
    """Assemble a linear-energy event of ~target_len minutes from blocks.

    A single block longer than needed contributes a random contiguous
    segment; otherwise blocks are concatenated end to end, each with its
    own +-3 dB relative gain jitter, joined by blending the shared
    boundary minute (a 1-minute crossfade at minute resolution).
    """
    blk = blocks[int(r.integers(len(blocks)))]
    if blk.shape[0] >= target_len:
        start = int(r.integers(0, blk.shape[0] - target_len + 1))
        seg = db_to_lin(blk[start:start + target_len])
        return seg * (10.0 ** (r.uniform(-3, 3) / 10.0))
    out = db_to_lin(blk) * (10.0 ** (r.uniform(-3, 3) / 10.0))
    total = blk.shape[0]
    n_pieces = 1
    while total < target_len and n_pieces < 16:
        b2 = blocks[int(r.integers(len(blocks)))]
        seg = db_to_lin(b2) * (10.0 ** (r.uniform(-3, 3) / 10.0))
        out[-1] = 0.5 * out[-1] + 0.5 * seg[0]     # boundary-minute blend
        out = np.concatenate([out, seg[1:]], axis=0)
        total += b2.shape[0] - 1
        n_pieces += 1
    return out[:target_len] if out.shape[0] > target_len else out


def slice_ambient(clip, r):
    """Cut one ambient clip into non-overlapping 4-8 min segments."""
    segs = []
    i, L = 0, clip.shape[0]
    while L - i >= 4:
        n = int(r.integers(4, 9))
        if 0 < L - i - n < 4:
            n = L - i                     # absorb a short remainder
        n = min(n, L - i)
        segs.append(clip[i:i + n])
        i += n
    return segs


class StratPool:
    """Background pool sampled uniformly across total-level deciles."""

    def __init__(self, X, site):
        self.X, self.site = X, site
        tot = 10.0 * np.log10(np.maximum(
            db_to_lin(X.astype(np.float64)).sum(axis=(1, 2)), 1e-30))
        edges = np.percentile(tot, np.linspace(0, 100, 11))
        b = np.digitize(tot, edges[1:-1])
        self.by_bin = [np.where(b == k)[0] for k in range(10) if (b == k).any()]

    def draw(self, r):
        idxs = self.by_bin[int(r.integers(len(self.by_bin)))]
        return int(idxs[int(r.integers(len(idxs)))])


def make_example(bg_db, clips, r, force_long=False):
    E = db_to_lin(bg_db)                       # (15,25)
    s = draw_share(r)
    if force_long:
        while s <= 0:
            s = draw_share(r)
    kind = "none"
    dur = 0
    if s > 0:
        use_ambient = r.random() < AMBIENT_P
        pool = clips["ambient"] if use_ambient else clips["vessel"]
        if len(pool) == 0:
            pool = clips["vessel"]
            use_ambient = False
        D = r.uniform(12.0, 15.01) if force_long else draw_duration(r)
        w = r.uniform(0.5, 2.0)                # time warp (v1: 0.7-1.4)
        src_len = int(np.clip(round(D / w), 1, 40))
        seg = build_event_lin(pool, src_len, r)
        if r.random() < 0.30:
            seg = seg[::-1].copy()             # trajectory reversal
        new_len = int(np.clip(round(seg.shape[0] * w), 1, 15))
        seg = warp_time(seg, new_len, r)
        seg = colour(seg, r)
        seg = gain_envelope(seg, r)
        off = int(r.integers(0, 15 - new_len + 1))
        p_tot = seg.sum()
        e_tot = E.sum()
        alpha = s * e_tot / ((1.0 - s) * p_tot)
        E[off:off + new_len] += alpha * seg
        kind = "ambient" if use_ambient else "vessel"
        dur = new_len
        s = float(alpha * p_tot / (e_tot + alpha * p_tot))   # realised share
    win = lin_to_db(E).astype(np.float64)
    drops = [m for m, p in MISS_P.items() if r.random() < p]
    if len(drops) > 3:
        drops = list(r.choice(drops, size=3, replace=False))
    for m in drops:
        win[m, :] = np.nan
    return win.astype(np.float32), s, kind, dur


def gen(n, pool, clips, r, force_long=False):
    X = np.empty((n, 15, 25), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    sites = np.empty(n, dtype=object)
    kinds = np.empty(n, dtype=object)
    durs = np.empty(n, dtype=np.int32)
    for i in range(n):
        j = pool.draw(r)
        X[i], y[i], kinds[i], durs[i] = make_example(pool.X[j], clips, r, force_long)
        sites[i] = pool.site[j]
    return X, y, sites, kinds, durs


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
    # Passage split: IDENTICAL rng calls to v1 -> identical pool membership.
    def split_clips(pool):
        idx = r.permutation(len(pool))
        k = max(1, int(0.85 * len(pool)))
        return [pool[i] for i in idx[:k]], [pool[i] for i in idx[k:]]
    v_tr, v_va = split_clips(vessel)
    a_tr, a_va = split_clips(ambient) if len(ambient) >= 2 else (ambient, ambient)

    # Slice ambient clips into non-overlapping segments, per split.
    a_tr_seg = [s for c in a_tr for s in slice_ambient(c, r)]
    a_va_seg = [s for c in a_va for s in slice_ambient(c, r)]
    print(f"ambient segments: {len(a_tr_seg)} train / {len(a_va_seg)} val "
          f"(from {len(a_tr)} / {len(a_va)} clips)")
    clips_tr = dict(vessel=v_tr, ambient=a_tr_seg)
    clips_va = dict(vessel=v_va, ambient=a_va_seg)

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
    pools = dict(
        train=(StratPool(BX[tr_idx], BS[tr_idx]), clips_tr, False),
        val_random=(StratPool(BX[vr_idx], BS[vr_idx]), clips_va, False),
        val_site=(StratPool(BX[is_hold], BS[is_hold]), clips_va, False),
        val_period=(StratPool(BX[is_late], BS[is_late]), clips_va, False),
        val_long=(StratPool(BX[vr_idx], BS[vr_idx]), clips_va, True),
    )
    meta = dict(ambient_segments_train=len(a_tr_seg), ambient_segments_val=len(a_va_seg),
                durations={})
    for name, (pool, pc, fl) in pools.items():
        X, y, sites, kinds_out, durs = gen(SIZES[name], pool, pc, r, force_long=fl)
        np.savez_compressed(DATA / f"{PREFIX}_{name}.npz",
                            X=X, y=y, site=sites, kind=kinds_out, dur=durs)
        ev = durs[durs > 0]
        meta["durations"][name] = dict(
            n_events=int(len(ev)),
            frac_ge_12=float((ev >= 12).mean()) if len(ev) else 0.0,
            frac_eq_15=float((ev == 15).mean()) if len(ev) else 0.0,
            mean=float(ev.mean()) if len(ev) else 0.0,
            median=float(np.median(ev)) if len(ev) else 0.0,
            hist=np.histogram(ev, bins=[0, 2, 4, 6, 8, 10, 12, 14, 16])[0].tolist())
        print(f"{name}: {X.shape} from {len(pool.X)} bg windows; "
              f"share mean {y.mean():.3f}, zero frac {(y == 0).mean():.3f}; "
              f"event dur mean {meta['durations'][name]['mean']:.1f} min, "
              f">=12 min {meta['durations'][name]['frac_ge_12']:.2f}")
    (DATA / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=1))
    print("meta:", json.dumps(meta["durations"]["train"]))


if __name__ == "__main__":
    main()
