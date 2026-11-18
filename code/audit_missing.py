"""Phase 5, workstream 2: missingness dependence audit.

Our simulator (v1-v3) draws the missing-minute mask independently of
the added event. This audit tests that assumption. Three mechanisms
are compared on PAIRED synthetic windows (same seed per arm: same
background, same target share, same event; only the mask mechanism
differs). Pools and clips are the BASE conversion's (passages.npz)
with v3's split discipline; the audit's val pools are the val_random
background pool and validation clips.

  indep — v3's mask exactly (simulate_v3.draw_mask): per-minute
          Bernoulli drops from the measured test pattern, cap 3.
  event — the same drop-count draw, but WHICH minutes drop is drawn
          with probability proportional to (0.25 + the placed event's
          per-minute energy): minutes carrying the addition are
          preferentially lost (an export that drops saturated or
          flagged minutes).
  bg    — the same drop-count draw, but minutes drop with probability
          proportional to the background's own quietness (squared
          rank): the recording's weak minutes are preferentially
          lost (a gap process that precedes mixing).

Statistics compared across arms and against the REAL windows
(inference and structure only; nothing fitted on Test.csv):
  * observed total energy (dB) by missingness pattern group;
  * the discriminating contrast: mean energy of observed minutes
    2..15 in minute-1-missing windows minus complete windows;
  * frozen v3 main model predictions by pattern group and by number
    of observed minutes.
Pattern groups: complete, m01 only, m15 only, other single, >=2.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA, ROOT, SEED, db_to_lin, lin_to_db
from simulate_v2 import MISS_P, slice_ambient
from simulate_v3 import V3Pool, traffic_scores, build_event_v3, \
    draw_share, draw_mask
from train_conv import featurize_mp

N = 12000
AUDIT_SEED = SEED + 77000
HOLDOUT_SITES = {"4 Bird Island", "9 Fish Hoek"}


def base_val_pool_and_clips():
    """v3 split discipline on the base passages; returns the
    val_random background pool (V3Pool) and validation clips."""
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    BX, BS, BT = bg["X"], bg["site"], bg["start"]
    pas = np.load(DATA / "passages.npz", allow_pickle=True)
    rel, kinds_arr = list(pas["rel_db"]), pas["kinds"]
    vessel = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "vessel"]
    ambient = [rel[i] for i in range(len(rel)) if kinds_arr[i] == "ambient"]
    traffic = traffic_scores(BX)
    r_split = np.random.default_rng(SEED)

    def split_idx(m):
        idx = r_split.permutation(m)
        k = max(1, int(0.85 * m))
        return idx[:k], idx[k:]

    v_tr_i, v_va_i = split_idx(len(vessel))
    a_tr_i, a_va_i = split_idx(len(ambient))
    a_va = [ambient[i] for i in a_va_i]
    v_va = [vessel[i] for i in v_va_i]
    r = np.random.default_rng(SEED)
    a_tr_seg = [s for c in [ambient[i] for i in a_tr_i]
                for s in slice_ambient(c, r)]
    a_va_seg = [s for c in a_va for s in slice_ambient(c, r)]
    clips_va = dict(vessel=v_va, ambient=a_va_seg)
    # Consume r exactly as simulate_v3.main's halve() calls do, so the
    # background shuffle below lands on v3's own val_random pool.
    _ = r.permutation(max(1, int(0.85 * len(vessel))))
    _ = r.permutation(len(a_tr_seg))
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
    vr_idx = idx[:n_vr]
    return V3Pool(BX[vr_idx], BS[vr_idx], traffic[vr_idx]), clips_va


def draw_count(r):
    """Drop count under v3's mask process (same RNG consumption as
    simulate_v3.draw_mask)."""
    cand = [m for m, p in MISS_P.items() if r.random() < p]
    if len(cand) > 3:
        cand = list(r.choice(cand, size=3, replace=False))
    return len(cand)


def make_window(pool, clips, r, mech):
    E_bg, site, mode = pool.draw_background(r)
    s = draw_share(r)
    placed = np.zeros((15, 25))
    if s > 0:
        use_ambient = r.random() < 0.30
        blocks = clips["ambient"] if use_ambient else clips["vessel"]
        if len(blocks) == 0:
            blocks = clips["vessel"]
        placed, _dur = build_event_v3(blocks, r, False)
    if mech == "indep":
        mask = draw_mask(r)
    else:
        k = draw_count(r)
        if mech == "event":
            w = 0.25 + placed.sum(axis=1)
        elif mech == "bg":
            tot = E_bg.sum(axis=1)
            rank = np.argsort(np.argsort(tot))       # 0 = quietest
            w = (15 - rank).astype(np.float64) ** 2
        else:
            raise ValueError(mech)
        mask = np.ones(15, dtype=bool)
        if k > 0:
            w = w / w.sum()
            mask[list(r.choice(15, size=k, replace=False, p=w))] = False
    label = 0.0
    E = E_bg
    if s > 0:
        order = "drop-first" if r.random() < 0.5 else "mix-first"
        obs = mask[:, None]
        bg_obs = np.where(obs, E_bg, 0.0).sum()
        p_obs = np.where(obs, placed, 0.0).sum()
        if p_obs > 0:
            if order == "drop-first":
                alpha = s * bg_obs / ((1.0 - s) * p_obs)
            else:
                alpha = s * E_bg.sum() / ((1.0 - s) * placed.sum())
            E = E_bg + alpha * placed
            label = float(alpha * p_obs / (bg_obs + alpha * p_obs))
    win = np.round(lin_to_db(E) * 10.0) / 10.0
    win[~mask, :] = np.nan
    return win.astype(np.float32), label


def pattern_of(win):
    miss = np.where(~np.isfinite(win[:, 0]))[0]
    if len(miss) == 0:
        return "complete"
    if len(miss) == 1 and miss[0] == 0:
        return "m01_only"
    if len(miss) == 1 and miss[0] == 14:
        return "m15_only"
    if len(miss) == 1:
        return "other_single"
    return "ge2"


def stats_for(X, pred):
    pats = np.array([pattern_of(X[i]) for i in range(len(X))])
    nobs = np.isfinite(X[:, :, 0]).sum(axis=1)
    E = np.where(np.isfinite(X), db_to_lin(np.where(np.isfinite(X), X, 0.0)),
                 0.0)
    tot_db = lin_to_db(E.sum(axis=(1, 2)))
    mid_db = lin_to_db(E[:, 1:, :].sum(axis=(1, 2)))
    groups = {}
    for g in ["complete", "m01_only", "m15_only", "other_single", "ge2"]:
        m = pats == g
        if m.sum() == 0:
            continue
        groups[g] = dict(
            n=int(m.sum()), frac=float(m.mean()),
            tot_db_mean=float(tot_db[m].mean()),
            tot_db_median=float(np.median(tot_db[m])),
            mid215_db_mean=float(mid_db[m].mean()),
            pred_mean=float(pred[m].mean()),
            pred_median=float(np.median(pred[m])),
            pred_frac_below_0p05=float((pred[m] < 0.05).mean()))
    by_nobs = {}
    for k in [15, 14, 13, 12]:
        m = nobs == k
        if m.sum():
            by_nobs[str(k)] = dict(n=int(m.sum()),
                                   pred_mean=float(pred[m].mean()))
    m = nobs <= 11
    if m.sum():
        by_nobs["<=11"] = dict(n=int(m.sum()),
                               pred_mean=float(pred[m].mean()))
    contrast = None
    if "m01_only" in groups and "complete" in groups:
        contrast = (groups["m01_only"]["mid215_db_mean"]
                    - groups["complete"]["mid215_db_mean"])
    return dict(groups=groups, by_n_observed=by_nobs,
                contrast_m01_vs_complete_mid215_db=contrast)


def main():
    pool, clips = base_val_pool_and_clips()
    model = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_sim_v3.txt"))
    kw = dict(num_iteration=int(
        (ROOT / "models" / "lgbm_sim_v3_best_iter.txt").read_text()))
    out = dict(n_per_arm=N, seed=AUDIT_SEED, arms={})
    for mech in ["indep", "event", "bg"]:
        r = np.random.default_rng(AUDIT_SEED)
        X = np.empty((N, 15, 25), dtype=np.float32)
        y = np.empty(N)
        for i in range(N):
            X[i], y[i] = make_window(pool, clips, r, mech)
        F = featurize_mp(X)
        pred = np.clip(model.predict(F, **kw), 0, 1)
        st = stats_for(X, pred)
        st["label_mean"] = float(y.mean())
        out["arms"][mech] = st
        print(f"[{mech}] label mean {y.mean():.3f} | contrast "
              f"{st['contrast_m01_vs_complete_mid215_db']} | pred means "
              f"{ {g: round(v['pred_mean'], 3) for g, v in st['groups'].items()} }",
              flush=True)
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft = featurize_mp(Xt)
    pred_t = np.clip(model.predict(Ft, **kw), 0, 1)
    out["real"] = stats_for(Xt, pred_t)
    print("[real] contrast", out["real"]["contrast_m01_vs_complete_mid215_db"],
          "| pred means",
          {g: round(v["pred_mean"], 3)
           for g, v in out["real"]["groups"].items()}, flush=True)
    (ROOT / "models" / "audit_missing.json").write_text(json.dumps(out,
                                                                  indent=1))
    print("wrote models/audit_missing.json", flush=True)


if __name__ == "__main__":
    main()
