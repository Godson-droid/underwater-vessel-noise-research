"""Phase 6 workstream 1, diagnostic half.

Q: is band 0 helping identify added-vessel energy, or letting the
model memorise site identity?

Tests:
 1. Site classifier on PURE background windows from the release
    pool (no added passage): how accurately can the background's
    site be identified from (a) all 549 features, (b) the 19
    band-0-specific features alone, (c) just band-0 level
    (band_share_0 + band_mean_db_rel_0), (d) all features EXCEPT
    band-0-specific ones? If band 0 alone identifies site nearly
    as well as everything, it is a site fingerprint channel.
 2. Baseline main model gain importance: band-0 group's share.
Output: models/phase6_ws1_diag.json.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool
from common import DATA, featurize
from phase6_lib import MODEL_DIR, band0_feature_mask

SEEDC = 20261117


def _feat_one(x):
    return featurize(x)[0]


def get_bg_features():
    cache = DATA / "feat_p6_bgclf.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"]), d["site"], d["idx"]
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    X, S = bg["X"], bg["site"]
    rng = np.random.default_rng(SEEDC)
    idx = []
    for s in sorted(set(S.tolist())):
        ii = np.where(S == s)[0]
        rng.shuffle(ii)
        idx.append(ii[:3000])
    idx = np.sort(np.concatenate(idx))
    with Pool(4) as p:
        F = p.map(_feat_one, [X[i] for i in idx], chunksize=256)
    F = np.asarray(F, dtype=np.float32)
    _, names = featurize(X[0])
    np.savez_compressed(cache, F=F, names=np.array(names),
                        site=S[idx], idx=idx)
    return F, names, S[idx], idx


def classify(F, y, label):
    rng = np.random.default_rng(SEEDC)
    perm = rng.permutation(len(F))
    cut = int(0.8 * len(F))
    tri, vai = perm[:cut], perm[cut:]
    m = lgb.train(
        dict(objective="multiclass", num_class=len(set(y.tolist())),
             metric="multi_logloss", learning_rate=0.1, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=1, seed=SEEDC,
             verbose=-1, num_threads=2),
        lgb.Dataset(F[tri], y[tri]), num_boost_round=500,
        valid_sets=[lgb.Dataset(F[vai], y[vai])],
        callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(F[vai], num_iteration=m.best_iteration)
    acc = float((p.argmax(axis=1) == y[vai]).mean())
    print(f"site classifier [{label}]: acc {acc:.4f} "
          f"(best iter {m.best_iteration})", flush=True)
    return acc


def main():
    F, names, sites, idx = get_bg_features()
    classes = sorted(set(sites.tolist()))
    y = np.array([classes.index(s) for s in sites])
    counts = pd.Series(sites).value_counts()
    chance = float(counts.max() / counts.sum())
    print(f"bg sample n={len(F)}; sites={len(classes)}; "
          f"majority-class chance {chance:.4f}", flush=True)
    b0 = band0_feature_mask(names)
    lvl = np.array([n in ("band_share_0", "band_mean_db_rel_0")
                    for n in names])
    out = dict(n=int(len(F)), chance_majority=chance,
               site_counts={k: int(v) for k, v in counts.items()})
    out["acc_all"] = classify(F, y, "all 549")
    out["acc_band0_only"] = classify(F[:, b0], y, "band0 19")
    out["acc_band0_level2"] = classify(F[:, lvl], y, "band0 level 2")
    out["acc_no_band0"] = classify(F[:, ~b0], y, "no band0 (530)")

    imp = pd.read_csv(MODEL_DIR / "lgbm_conv_bandext_maskfix_importance.csv",
                      index_col=0).iloc[:, 0]
    tot = float(imp.sum())
    # align by name
    mask_by_name = dict(zip(names, b0))
    b0_names = [n for n in imp.index if mask_by_name.get(n, False)]
    out["baseline_importance"] = dict(
        band0_gain_share=float(imp[b0_names].sum() / tot),
        top15={k: float(v / tot) for k, v in
               imp.sort_values(ascending=False).head(15).items()})
    print("band0 gain share:",
          out["baseline_importance"]["band0_gain_share"], flush=True)
    print("top15:", json.dumps(out["baseline_importance"]["top15"],
                               indent=1), flush=True)
    (MODEL_DIR / "phase6_ws1_diag.json").write_text(
        json.dumps(out, indent=1))
    print("wrote models/phase6_ws1_diag.json", flush=True)


if __name__ == "__main__":
    main()
