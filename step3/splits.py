"""Step 3, Stage 0: material splits for the independent synthetic validation (Path I1).

Fresh code written for Step 3 (2026-11-18). It reads only metadata and pool
index information; it does not import any archived simulator, converter or
label code.

Clip split: the 525 VTUAD strict clips that carry hull (MMSI) and recording
date metadata (data/vtuad/metadata.csv, rows mapped 1:1 to the strict vessel
and background archives). The 17 class-archive clips (sailing, tanker,
rescue, pilotvessel) carry no hull/date metadata in the workspace and are
excluded from BOTH sides, recorded here. The split uses a rectangle
assignment: every hull and every recording date is assigned to one side,
and a clip is kept on a side only where its hull assignment and its date
assignment agree; clips whose two assignments disagree are excluded. The
kept sides are therefore disjoint in hull AND in date by construction.
Assignments are chosen by seeded simulated annealing maximising the
smaller kept side with an ambient-balance term. (A simpler whole-date
assignment with hull purging was tried first and kept far fewer clips;
it was discarded before any corpus was generated.)

Background split: the screened pool (data/bg_windows.npz, 87,653 windows,
8 sites). Bird Island and Fish Hoek are held out entirely (unseen-site
material). Within the six family sites, the latest 15% of window start
times form a period holdout excluded from all family-side fitting. The
remaining family-site windows are split into disjoint FIT and EVAL halves:
FIT feeds family fitting and calibration backgrounds; EVAL feeds the
seen-site part of the I1 main set. Bird Island / Fish Hoek windows are
partitioned between the null set and the I1 main set, disjointly.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "splits.json"
SEED = 20261118

UNSEEN_SITES = ["4 Bird Island", "9 Fish Hoek"]


def clip_split(rng):
    meta = pd.read_csv(DATA / "vtuad" / "metadata.csv")
    meta["clip_key"] = meta["path"].str.replace("./", "", regex=False)
    meta["is_ambient_label"] = meta["label"] == "background"
    dates = sorted(meta["date"].unique())
    hulls = sorted(meta["MMSI"].unique())
    di = meta["date"].map({d: i for i, d in enumerate(dates)}).to_numpy()
    hi = meta["MMSI"].map({h: i for i, h in enumerate(hulls)}).to_numpy()
    amb = meta["is_ambient_label"].to_numpy()

    def evaluate(ds, hs):
        s1 = (ds[di] == 1) & (hs[hi] == 1)
        s0 = (ds[di] == 0) & (hs[hi] == 0)
        return s1, s0

    def objective(ds, hs):
        s1, s0 = evaluate(ds, hs)
        a, b = int(s1.sum()), int(s0.sum())
        aa, ab = int((s1 & amb).sum()), int((s0 & amb).sum())
        pen = 200 if min(a, b) < 100 else 0
        return min(a, b) + 0.3 * (a + b) + 4 * min(aa, ab) - pen

    best = None
    for _restart in range(6):
        ds = (rng.random(len(dates)) < 0.45).astype(int)
        hs = (rng.random(len(hulls)) < 0.45).astype(int)
        cur = objective(ds, hs)
        for it in range(30000):
            temp = 2.0 * (1 - it / 30000) + 0.05
            if rng.random() < 0.5:
                j = int(rng.integers(len(dates)))
                ds[j] ^= 1
                new = objective(ds, hs)
                if new >= cur or rng.random() < np.exp((new - cur) / temp):
                    cur = new
                else:
                    ds[j] ^= 1
            else:
                j = int(rng.integers(len(hulls)))
                hs[j] ^= 1
                new = objective(ds, hs)
                if new >= cur or rng.random() < np.exp((new - cur) / temp):
                    cur = new
                else:
                    hs[j] ^= 1
        if best is None or cur > best[0]:
            best = (cur, ds.copy(), hs.copy())
    _score, ds, hs = best
    s1, s0 = evaluate(ds, hs)
    meta["side"] = np.where(s1, "i1", np.where(s0, "family", "excluded_disagreement"))
    i1_dates = sorted(int(dates[i]) for i in np.where(ds == 1)[0])
    fam_dates = sorted(int(dates[i]) for i in np.where(ds == 0)[0])
    return meta, i1_dates, fam_dates, []


def bg_split(rng):
    z = np.load(DATA / "bg_windows.npz")
    site = np.asarray(z["site"])
    start = np.asarray(z["start"])
    n = len(site)
    role = np.full(n, "unused", dtype=object)
    # Period holdout within family sites: latest 15% of starts.
    fam_mask = ~np.isin(site, UNSEEN_SITES)
    fam_idx = np.where(fam_mask)[0]
    thresh = np.quantile(start[fam_idx], 0.85)
    period_hold = fam_idx[start[fam_idx] >= thresh]
    role[period_hold] = "eval_period_holdout"
    rest = fam_idx[start[fam_idx] < thresh]
    perm = rng.permutation(rest)
    half = len(perm) // 2
    role[perm[:half]] = "fit"
    role[perm[half:]] = "eval_seen"
    # Unseen sites: partition between null set and main set, disjointly.
    for s, n_null in (("4 Bird Island", 400), ("9 Fish Hoek", 150)):
        idx = np.where(site == s)[0]
        perm = rng.permutation(idx)
        role[perm[:n_null]] = "null"
        role[perm[n_null:]] = "eval_unseen"
    return site, start, role, float(thresh)


def main():
    rng = np.random.default_rng(SEED)
    meta, i1_dates, fam_dates, clash = clip_split(rng)
    site, start, role, thresh = bg_split(rng)

    def clip_summary(side_name):
        m = meta[meta["side"] == side_name]
        return {
            "clips": int(len(m)),
            "vessel_label_clips": int((~m["is_ambient_label"]).sum()),
            "ambient_label_clips": int(m["is_ambient_label"].sum()),
            "unique_hulls_mmsi": int(m["MMSI"].nunique()),
            "unique_dates": int(m["date"].nunique()),
            "dates": sorted(int(d) for d in m["date"].unique()),
            "clip_keys": sorted(m["clip_key"].tolist()),
        }

    fam = clip_summary("family")
    i1 = clip_summary("i1")
    shared_hulls = set(meta.loc[meta["side"] == "family", "MMSI"]) & set(
        meta.loc[meta["side"] == "i1", "MMSI"])
    shared_dates = set(fam["dates"]) & set(i1["dates"])
    roles = {}
    for r in sorted(set(role.tolist())):
        roles[r] = int((role == r).sum())
    per_site_roles = {}
    for s in sorted(set(site.tolist())):
        m = site == s
        per_site_roles[s] = {r: int((role[m] == r).sum())
                             for r in sorted(set(role[m].tolist()))}
    out = {
        "seed": SEED,
        "clip_split": {
            "family": fam,
            "i1": i1,
            "excluded_disagreement_clips": int((meta["side"] == "excluded_disagreement").sum()),
            "excluded_class_archive_clips_no_metadata": 17,
            "shared_hulls_kept_sides": len(shared_hulls),
            "shared_dates_kept_sides": len(shared_dates),
        },
        "bg_split": {
            "period_holdout_start_threshold_epoch": thresh,
            "roles": roles,
            "per_site_roles": per_site_roles,
            "unseen_sites": UNSEEN_SITES,
        },
    }
    # Persist the per-clip side assignment and per-window roles for the
    # generator, alongside the human-readable summary above.
    meta_out = meta[["clip_key", "label", "date", "MMSI", "duration_sec",
                     "is_ambient_label", "side"]]
    meta_out.to_csv(Path(__file__).resolve().parent / "clip_sides.csv", index=False)
    np.savez_compressed(Path(__file__).resolve().parent / "bg_roles.npz",
                        site=site, start=start,
                        role=np.asarray(role, dtype="<U20"))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["clip_split"], indent=1)[:1200])
    print("bg roles:", roles)


if __name__ == "__main__":
    main()
