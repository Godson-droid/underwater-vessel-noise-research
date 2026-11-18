"""Train the first simulator model (LightGBM) and validate per PLAN.md 4.5.

- Trains ONLY on synthetic mixes (sim_train.npz). Nothing is fitted,
  calibrated or tuned on Test.csv; test inference uses the identical fixed
  feature transform from common.py.
- Validation on three disjoint synthetic sets (val_random / val_site /
  val_period), with the B0 constant (0.1) scored on identical sets.
- score_added_vessel = sqrt(share prediction): a fixed monotone companion
  (the column is a zero-weight ROC-AUC diagnostic, notebook cell 5).
- Fixed seed; manual hyperparameters (no AutoML).
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA, ROOT, SEED, featurize_batch

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, verbose=-1, num_threads=2)


def load(name):
    d = np.load(DATA / f"sim_{name}.npz", allow_pickle=True)
    return d["X"], d["y"].astype(np.float64), d["site"], d["kind"]


def load_features(name, X):
    """Featurize with an on-disk cache (featurization is the slow step)."""
    cache = DATA / f"feat_{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"])
    F, names = featurize_batch(X)
    np.savez_compressed(cache, F=F, names=np.array(names))
    return F, names


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    Xtr, ytr, _, _ = load("train")
    print("featurizing train...", flush=True)
    Ftr, names = load_features("train", Xtr)
    # Internal split for early stopping (from train only).
    perm = rng.permutation(len(Ftr))
    cut = int(0.9 * len(Ftr))
    tri, vai = perm[:cut], perm[cut:]
    dtrain = lgb.Dataset(Ftr[tri], ytr[tri], feature_name=names)
    dval = lgb.Dataset(Ftr[vai], ytr[vai], reference=dtrain)
    model = lgb.train(PARAMS, dtrain, num_boost_round=1500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    print("best iteration:", model.best_iteration)
    model.save_model(str(MODEL_DIR / "lgbm_sim_v1.txt"),
                     num_iteration=model.best_iteration)
    imp = pd.Series(model.feature_importance("gain"), index=names)
    imp.sort_values(ascending=False).to_csv(MODEL_DIR / "lgbm_sim_v1_importance.csv")

    results = {}
    for name in ["val_random", "val_site", "val_period"]:
        X, y, sites, _ = load(name)
        F, _ = load_features(name, X)
        pred = np.clip(model.predict(F, num_iteration=model.best_iteration), 0, 1)
        b0 = np.full_like(y, 0.1)
        entry = dict(
            rmse_model=rmse(pred, y), score_model=1 - rmse(pred, y),
            rmse_b0=rmse(b0, y), score_b0=1 - rmse(b0, y),
            null_mean_pred=float(pred[y == 0].mean()) if (y == 0).any() else None,
            n=int(len(y)),
        )
        # Pressure map: RMSE by true share band.
        bands = [(0.0, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
        pm = {}
        for lo, hi in bands:
            m = (y == 0) if hi == 0.0 else ((y > lo) & (y <= hi))
            if m.sum() > 0:
                pm[f"{lo}-{hi}"] = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
        entry["by_share_band"] = pm
        # By site (meaningful for val_site; recorded for all).
        by_site = {}
        for s in np.unique(sites):
            m = sites == s
            if m.sum() >= 30:
                by_site[str(s)] = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
        entry["by_site"] = by_site
        # By missingness (count of NaN minutes).
        nmiss = np.isnan(X).any(axis=2).sum(axis=1)
        by_miss = {}
        for k in [0, 1, 2]:
            m = nmiss == k if k < 2 else nmiss >= 2
            if m.sum() > 0:
                by_miss[str(k)] = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
        entry["by_n_missing_minutes"] = by_miss
        results[name] = entry
        print(name, json.dumps(entry, indent=1), flush=True)

    (MODEL_DIR / "sim_v1_validation.json").write_text(json.dumps(results, indent=1))

    # Test inference (fixed transform only).
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft, _ = load_features("test", Xt)
    share = np.clip(model.predict(Ft, num_iteration=model.best_iteration), 0, 1)
    sub = pd.DataFrame(dict(ID=test["ID"], share_added_vessel=share,
                            score_added_vessel=np.sqrt(share)))
    out = ROOT / "submissions" / "submission_sim_v1.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "share mean/median:", float(share.mean()),
          float(np.median(share)))


if __name__ == "__main__":
    main()
