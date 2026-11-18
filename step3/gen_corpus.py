"""Step 3: generate the I1 corpora (calibration, main, null).

Design (proposal Sections 4 and 6):
  calib : family-side clips + FIT-partition backgrounds only. Used to fit
          Stage 1 training-side constants. Never scored as validation.
  main  : I1-side clips; backgrounds drawn from evaluation partitions of
          the pool (60% seen-site eval, 15% period holdout, 25% unseen
          sites). The scored independent synthetic validation set.
  null  : pure background windows from the null partition (unseen sites),
          mask applied, nothing added. True share exactly zero.

Estimator-facing files (corpus/<set>_X.npz) carry only the summary and
subset tags. Ground truth lives in corpus/<set>_truth.npz and is read
only by the scorer, the oracle check and the hand-check.

Generation is chunked (250 windows per chunk, one seed per chunk) and
resumable: a chunk whose two files already exist is skipped, so a lost
process costs at most one chunk of work. A merge step concatenates the
chunks into the final per-set files.
"""
import json
import sys
from pathlib import Path
import numpy as np

import i1_common as i1

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus"
CHUNKS = OUT / "chunks"
BASE_SEED = 20261118
CHUNK = 250
SETS = {"calib": 1500, "main": 2000}


def clip_pools():
    import pandas as pd
    sides = pd.read_csv(HERE / "clip_sides.csv")
    pools = {}
    for side in ("family", "i1"):
        sub = sides[sides["side"] == side]
        pools[side] = {
            "vessel": sub.loc[~sub["is_ambient_label"], "clip_key"].tolist(),
            "ambient": sub.loc[sub["is_ambient_label"], "clip_key"].tolist(),
        }
    return pools


class BgSpec:
    """Callable background-draw plan matching i1_common.make_window's
    protocol: spec(rng) -> (bg_db, site, tag); spec(rng, same_site=s)
    draws a second background at the same site and role (mixture mode).
    Plan entries are (role, probability, site_weights or None); without
    explicit site weights, sites are chosen in proportion to their
    window counts within the role."""

    allow_mixture = True

    def __init__(self, pool, plan):
        self.pool = pool
        self.plan = plan
        self.last_role = None

    def _pick_site(self, rng, role, site_w):
        sites = self.pool.sites_by_role[role]
        if site_w is not None:
            sites = [s for s in sites if s in site_w]
            w = np.array([site_w[s] for s in sites], dtype=float)
        else:
            w = np.array([len(self.pool.by_role_site[(role, s)]) for s in sites],
                         dtype=float)
        w = w / w.sum()
        return sites[int(rng.choice(len(sites), p=w))]

    def __call__(self, rng, same_site=None):
        if same_site is not None:
            X, s, i = self.pool.draw(rng, role=self.last_role, site=same_site)
            return X, s, f"{self.last_role}:{i}"
        r = rng.random()
        acc = 0.0
        role, site_w = self.plan[-1][0], self.plan[-1][2]
        for cand_role, p, cand_w in self.plan:
            acc += p
            if r <= acc:
                role, site_w = cand_role, cand_w
                break
        self.last_role = role
        site = self._pick_site(rng, role, site_w)
        X, s, i = self.pool.draw(rng, role=role, site=site)
        return X, s, f"{role}:{i}"


def build_chunk(set_name, chunk_idx, pool, pools, dur_values):
    cx = CHUNKS / f"{set_name}_{chunk_idx:02d}_X.npz"
    ct = CHUNKS / f"{set_name}_{chunk_idx:02d}_truth.npz"
    if cx.exists() and ct.exists():
        print(f"[{set_name} chunk {chunk_idx}] exists, skipped", flush=True)
        return
    if set_name == "calib":
        spec = BgSpec(pool, [("fit", 1.0, None)])
        clips = pools["family"]
        seed = BASE_SEED + 1000 + chunk_idx
    else:
        spec = BgSpec(pool, [("eval_seen", 0.60, None),
                             ("eval_period_holdout", 0.15, None),
                             ("eval_unseen", 0.25, {"4 Bird Island": 0.85,
                                                     "9 Fish Hoek": 0.15})])
        clips = pools["i1"]
        seed = BASE_SEED + 2000 + chunk_idx
    rng = np.random.default_rng(seed)
    recs = []
    tries = 0
    while len(recs) < CHUNK and tries < CHUNK * 4:
        tries += 1
        r = i1.make_window(rng, pool, spec, clips, dur_values)
        if r is not None:
            recs.append(r)
    assert len(recs) == CHUNK, (set_name, chunk_idx, len(recs))
    np.savez_compressed(cx,
                        X=np.stack([r["X"] for r in recs]).astype(np.float32),
                        site=np.array([r["site"] for r in recs]),
                        bg_tag=np.array([r["bg_tag"] for r in recs]),
                        kind=np.array([r["kind"] for r in recs]),
                        dur=np.array([r["dur"] for r in recs]))
    np.savez_compressed(ct,
                        y=np.array([r["y"] for r in recs]),
                        E_bg=np.stack([r["E_bg"] for r in recs]).astype(np.float32),
                        E_add=np.stack([r["E_add"] for r in recs]).astype(np.float32),
                        obs=np.stack([r["obs"] for r in recs]))
    print(f"[{set_name} chunk {chunk_idx}] done "
          f"(label mean {np.mean([r['y'] for r in recs]):.3f})", flush=True)


def build_null(pool):
    rng = np.random.default_rng(BASE_SEED + 99)
    idx = np.where(pool.role == "null")[0]
    Xl, sites = [], []
    for i in idx:
        w = pool.X[i].astype(np.float32).copy()
        missing = i1.draw_mask(rng)
        if missing:
            w[list(missing), :] = np.nan
        Xl.append(w)
        sites.append(pool.site[i])
    np.savez_compressed(OUT / "null_X.npz",
                        X=np.stack(Xl).astype(np.float32),
                        site=np.array(sites),
                        bg_tag=np.array(["null"] * len(idx)),
                        kind=np.array(["none"] * len(idx)),
                        dur=np.zeros(len(idx), dtype=int))
    np.savez_compressed(OUT / "null_truth.npz", y=np.zeros(len(idx)))
    print(f"[null] {len(idx)} windows", flush=True)


def merge(set_name):
    n = SETS[set_name]
    xs, ts = [], []
    for ci in range(n // CHUNK):
        xs.append(np.load(CHUNKS / f"{set_name}_{ci:02d}_X.npz"))
        ts.append(np.load(CHUNKS / f"{set_name}_{ci:02d}_truth.npz"))
    X = {k: np.concatenate([x[k] for x in xs]) for k in ("X", "site", "bg_tag", "kind", "dur")}
    T = {k: np.concatenate([t[k] for t in ts]) for k in ("y", "E_bg", "E_add", "obs")}
    np.savez_compressed(OUT / f"{set_name}_X.npz", **X)
    np.savez_compressed(OUT / f"{set_name}_truth.npz", **T)
    y = T["y"]
    meta = {
        "set": set_name, "n": int(len(y)),
        "label_mean": float(y.mean()), "label_zero_frac": float((y == 0).mean()),
        "kinds": {k: int((X["kind"] == k).sum()) for k in sorted(set(X["kind"].tolist()))},
        "bg_roles": {r: int(sum(1 for t in X["bg_tag"] if t.split(":")[0] == r))
                     for r in sorted(set(t.split(":")[0] for t in X["bg_tag"]))},
    }
    with open(OUT / f"{set_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    CHUNKS.mkdir(exist_ok=True)
    pools = clip_pools()
    pool = i1.BackgroundPool()
    z = np.load(i1.ROOT / "data" / "sim_cf_bandext_maskfix_val_random.npz",
                allow_pickle=True)
    dur_values = z["dur"][z["dur"] > 0].astype(int)
    print("duration pool:", len(dur_values), "mean", float(dur_values.mean()), flush=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only in (None, "null"):
        build_null(pool)
    for set_name, n in SETS.items():
        if only not in (None, set_name):
            continue
        for ci in range(n // CHUNK):
            build_chunk(set_name, ci, pool, pools, dur_values)
        merge(set_name)


if __name__ == "__main__":
    main()
