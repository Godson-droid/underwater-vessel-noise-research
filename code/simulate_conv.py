"""Phase 5 generator: the v3 construction with a variant passage pool.

Replicates simulate_v3.py's split discipline EXACTLY (same seed, same
RNG call order for the passage split, ambient slicing, clip halving
and background masks), so pool memberships are identical to v3's; only
the passage band values differ (they come from
data/passages_conv_<tag>.npz, produced by convert_variants.py with the
same clip order, clip set and minute counts).

Modes:
  screen — screening sets only (trainA 40k, val_* 6k, oracle 2k),
           generated with a fresh recorded RNG stream (seed
           SEED + 91000 + tag index). Prefix sim_cs_<tag>.
  full   — the identical RNG continuation as simulate_v3.main():
           generation consumes the same `r` used for the splits, in
           v3's set order and sizes, so the windows are v3's windows
           with variant passages substituted. Prefix sim_cf_<tag>.

Usage: python simulate_conv.py <tag> <screen|full>
"""
import json
import sys
import numpy as np
from common import DATA, SEED
from simulate_v2 import slice_ambient
from simulate_v3 import V3Pool, gen, traffic_scores

TAG_ORDER = ["rect", "rawfft", "seg5", "seg10", "min60", "welch1s",
             "median", "dbmean", "magmean", "bandext", "bandextdb"]
FULL_SIZES = dict(train=80000, trainA=40000, trainB=40000,
                  val_random=6000, val_site=6000, val_period=6000,
                  val_long=6000, oracle=2000)
SCREEN_SIZES = dict(trainA=40000, val_random=6000, val_site=6000,
                    val_period=6000, val_long=6000, oracle=2000)
HOLDOUT_SITES = {"4 Bird Island", "9 Fish Hoek"}


def build_pools(tag):
    """Returns (pools dict of (V3Pool, clips, force_long, want_bg),
    split_rng) with memberships identical to simulate_v3.main()."""
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    BX, BS, BT = bg["X"], bg["site"], bg["start"]
    pas = np.load(DATA / f"passages_conv_{tag}.npz", allow_pickle=True)
    rel, kinds_arr = list(pas["rel_db"]), pas["kinds"]
    vessel = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "vessel"]
    ambient = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "ambient"]
    print(f"passages[{tag}]: {len(vessel)} vessel, {len(ambient)} ambient",
          flush=True)

    traffic = traffic_scores(BX)

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
    print(f"ambient segments: {len(a_tr_seg)} train / {len(a_va_seg)} val",
          flush=True)
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
    return pools, r


def emit(prefix, name, pool, clips, fl, wb, n, r, meta):
    d = gen(n, pool, clips, r, force_long=fl, want_bg=wb)
    np.savez_compressed(DATA / f"{prefix}_{name}.npz", **d)
    modes, cnts = np.unique(d["mode"], return_counts=True)
    orders, ocnts = np.unique(d["order"], return_counts=True)
    meta["sets"][name] = dict(
        label_mean=float(d["y"].mean()),
        zero_frac=float((d["y"] == 0).mean()),
        modes={str(m): float(c / len(d["y"])) for m, c in zip(modes, cnts)},
        orders={str(o): float(c / len(d["y"])) for o, c in zip(orders, ocnts)})
    print(f"{name}: {d['X'].shape} from {len(pool.X)} bg windows; "
          f"label mean {d['y'].mean():.3f}, zero {(d['y'] == 0).mean():.3f}",
          flush=True)


def main():
    tag, mode = sys.argv[1], sys.argv[2]
    assert tag in TAG_ORDER, tag
    assert mode in ("screen", "full"), mode
    pools, r_split_state = build_pools(tag)
    meta = dict(tag=tag, mode=mode, seed=SEED,
                pools={k: int(len(v[0].X)) for k, v in pools.items()},
                sets={})
    if mode == "full":
        prefix = f"sim_cf_{tag}"
        meta["generation"] = "identical RNG continuation as simulate_v3"
        for name, (pool, pc, fl, wb) in pools.items():
            emit(prefix, name, pool, pc, fl, wb, FULL_SIZES[name],
                 r_split_state, meta)
    else:
        prefix = f"sim_cs_{tag}"
        gseed = SEED + 91000 + TAG_ORDER.index(tag)
        meta["generation"] = f"fresh stream, seed {gseed}"
        meta["generation_seed"] = gseed
        r = np.random.default_rng(gseed)
        for name in ["trainA", "val_random", "val_site", "val_period",
                     "val_long", "oracle"]:
            pool, pc, fl, wb = pools[name]
            emit(prefix, name, pool, pc, fl, wb, SCREEN_SIZES[name], r, meta)
    (DATA / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
