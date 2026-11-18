"""Phase 5, workstream 4: background channel (domain shift) variant.

One variant family, per the assessment: test backgrounds may come
through a different sensor/export chain than the released minutes.
v3 already applies a generic +-2 dB smooth jitter; this family
instead draws the frequency response from the RELEASE'S OWN inter
site spectral differences — the grounded size and shape of export
variation available to us.

Construction: per site, the median spectrum (per band, median over
the site's pool windows of the window's mean dB spectrum). For each
example, after the v3 background draw (clean / busy / mixture +
jitter + drift), an ordered pair of distinct sites (A, B) is drawn
and the background is multiplied per band by
10^((spec_A - spec_B) * u / 10), u ~ U(0.5, 1.0) — a withheld style
channel response, constant across the window's minutes, as a sensor
response would be. Passages are the BASE conversion's (passages.npz);
split discipline is v3's; labels are v3's observed-cells rule.

Modes mirror simulate_conv.py: screen (trainA + val_* + oracle,
recorded fresh seed) and full (all sets at v3 sizes, same recorded
seed stream — the channel draws change RNG consumption, so unlike
the conversion variants the windows are not v3's own; pairing is by
cross-scoring on identical files, as elsewhere).

Usage: python simulate_bgchan.py <screen|full>
"""
import json
import sys
import numpy as np
from common import DATA, SEED, db_to_lin
from simulate_v2 import slice_ambient
from simulate_v3 import V3Pool, gen, traffic_scores

HOLDOUT_SITES = {"4 Bird Island", "9 Fish Hoek"}
FULL_SIZES = dict(train=80000, trainA=40000, trainB=40000,
                  val_random=6000, val_site=6000, val_period=6000,
                  val_long=6000, oracle=2000)
SCREEN_SIZES = dict(trainA=40000, val_random=6000, val_site=6000,
                    val_period=6000, val_long=6000, oracle=2000)
GSEED = SEED + 92000


def site_spectra(BX, BS):
    specs = {}
    for s in np.unique(BS):
        W = BX[BS == s].astype(np.float64)
        specs[str(s)] = np.median(W.mean(axis=1), axis=0)   # (25,) dB
    return specs


class ChanPool(V3Pool):
    def __init__(self, X, site, traffic, specs):
        super().__init__(X, site, traffic)
        self.specs = specs
        self.site_names = sorted(specs.keys())
        self.chan_mag = []

    def draw_background(self, r):
        E, site, mode = super().draw_background(r)
        i = int(r.integers(len(self.site_names)))
        j = int(r.integers(len(self.site_names) - 1))
        if j >= i:
            j += 1
        u = r.uniform(0.5, 1.0)
        ch = (self.specs[self.site_names[i]]
              - self.specs[self.site_names[j]]) * u
        self.chan_mag.append(float(np.sqrt((ch ** 2).mean())))
        return E * (10.0 ** (ch / 10.0))[None, :], site, mode


def build_chan_pools():
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    BX, BS, BT = bg["X"], bg["site"], bg["start"]
    pas = np.load(DATA / "passages.npz", allow_pickle=True)
    rel, kinds_arr = list(pas["rel_db"]), pas["kinds"]
    vessel = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "vessel"]
    ambient = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "ambient"]
    traffic = traffic_scores(BX)
    specs = site_spectra(BX, BS)
    diffs = []
    names = sorted(specs)
    for a in names:
        for b in names:
            if a != b:
                diffs.append(np.sqrt(((specs[a] - specs[b]) ** 2).mean()))
    print(f"inter-site spectral difference RMS: median "
          f"{np.median(diffs):.2f} dB, p90 {np.percentile(diffs, 90):.2f} dB",
          flush=True)

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
    clips_tr = dict(vessel=v_tr, ambient=a_tr_seg)
    clips_va = dict(vessel=v_va, ambient=a_va_seg)

    def halve(items, r):
        idx = r.permutation(len(items))
        h = len(items) // 2
        return [items[i] for i in idx[:h]], [items[i] for i in idx[h:]]

    vA, vB = halve(v_tr, r)
    aA, aB = halve(a_tr_seg, r)
    clips_A = dict(vessel=vA, ambient=aA)
    clips_B = dict(vessel=vB, ambient=aB)

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
    perm = r.permutation(len(tr_idx))
    half = len(tr_idx) // 2
    trA_idx, trB_idx = tr_idx[perm[:half]], tr_idx[perm[half:]]

    def P(ii):
        return ChanPool(BX[ii], BS[ii], traffic[ii], specs)

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
    return pools


def main():
    mode = sys.argv[1]
    assert mode in ("screen", "full")
    pools = build_chan_pools()
    prefix = f"sim_c{'f' if mode == 'full' else 's'}_bgchan"
    sizes = FULL_SIZES if mode == "full" else SCREEN_SIZES
    order = (list(FULL_SIZES) if mode == "full" else
             ["trainA", "val_random", "val_site", "val_period",
              "val_long", "oracle"])
    r = np.random.default_rng(GSEED)
    meta = dict(tag="bgchan", mode=mode, seed=SEED, generation_seed=GSEED,
                pools={k: int(len(v[0].X)) for k, v in pools.items()},
                sets={})
    for name in order:
        pool, pc, fl, wb = pools[name]
        d = gen(sizes[name], pool, pc, r, force_long=fl, want_bg=wb)
        np.savez_compressed(DATA / f"{prefix}_{name}.npz", **d)
        meta["sets"][name] = dict(
            label_mean=float(d["y"].mean()),
            zero_frac=float((d["y"] == 0).mean()),
            chan_mag_mean=float(np.mean(pool.chan_mag)) if pool.chan_mag
            else None)
        print(f"{name}: {d['X'].shape}; label mean {d['y'].mean():.3f}; "
              f"channel RMS mean {meta['sets'][name]['chan_mag_mean']}",
              flush=True)
        pool.chan_mag = []
    (DATA / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
