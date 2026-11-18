"""Phase 6 workstream 5: training-coverage comparisons.

  moresame — baseline main training set (frozen train, 80k) PLUS
             the fresh same-distribution 80k (sim_p6ms): 160k total.
             Tests VOLUME at fixed distribution.
  diverse  — the broadened-distribution 80k alone (sim_p6dv).
             Tests DIVERSITY at fixed volume (80k = baseline size).
Both use the frozen protocol and 549 features, and are evaluated
on the identical frozen holdouts against baseline main
(0.9482 / 0.9030 / 0.9469 / 0.9422, real mean 0.2061).
Features for the new sets are computed with the frozen transform
and cached. Output: models/phase6_ws5.json.
"""
import json
import sys
import numpy as np
from multiprocessing import Pool
from common import DATA, featurize
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, fmt_scores)


def _one(x):
    return featurize(x)[0]


def feats_for(prefix, name):
    cache = DATA / f"feat_p6_{prefix}_{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"])
    d = np.load(DATA / f"{prefix}_{name}.npz", allow_pickle=True)
    X = d["X"]
    with Pool(4) as p:
        F = p.map(_one, [X[i] for i in range(len(X))], chunksize=256)
    F = np.asarray(F, dtype=np.float32)
    _, names = featurize(X[0])
    np.savez_compressed(cache, F=F, names=np.array(names))
    return F, names


feats, ys = {}, {}
names = None
for s in VAL_SETS:
    feats[s], names = load_base_features(s)
    ys[s] = load_set(s)["y"].astype(np.float64)
Ft, _ = load_test_features()
base_main = baseline_model("")
out = dict(baseline_main=eval_holdouts(base_main, feats, ys),
           baseline_main_real_mean=float(predict(base_main, Ft).mean()),
           variants={})
print(f"baseline main: {fmt_scores(out['baseline_main'])}", flush=True)

which = sys.argv[1:] or ["moresame", "diverse"]

if "moresame" in which:
    Ftr, _ = load_base_features("train")
    ytr = load_set("train")["y"].astype(np.float64)
    Fms, _ = feats_for("sim_p6ms", "train")
    yms = np.load(DATA / "sim_p6ms_train.npz",
                  allow_pickle=True)["y"].astype(np.float64)
    F = np.concatenate([Ftr, Fms])
    y = np.concatenate([ytr, yms])
    m, mkw, info = train_lgbm(F, y, names)
    sc = eval_holdouts((m, mkw), feats, ys)
    real = float(predict((m, mkw), Ft).mean())
    out["variants"]["moresame160k"] = dict(
        scores=sc, real_mean=real, best_iteration=info["best_iteration"],
        n_train=int(len(y)))
    print(f"moresame160k: {fmt_scores(sc)} | real {real:.4f}", flush=True)
    (MODEL_DIR / "phase6_ws5.json").write_text(json.dumps(out, indent=1))

if "diverse" in which:
    Fdv, _ = feats_for("sim_p6dv", "train")
    ydv = np.load(DATA / "sim_p6dv_train.npz",
                  allow_pickle=True)["y"].astype(np.float64)
    m, mkw, info = train_lgbm(Fdv, ydv, names)
    sc = eval_holdouts((m, mkw), feats, ys)
    real = float(predict((m, mkw), Ft).mean())
    out["variants"]["diverse80k"] = dict(
        scores=sc, real_mean=real, best_iteration=info["best_iteration"],
        n_train=int(len(ydv)))
    print(f"diverse80k: {fmt_scores(sc)} | real {real:.4f}", flush=True)
    (MODEL_DIR / "phase6_ws5.json").write_text(json.dumps(out, indent=1))
print("done", flush=True)
