"""Train and validate the Simulator v3 models (PLAN 4.9 / Phase 4).

Models (all LightGBM, v2's exact manual hyperparameters, fixed seed, no
AutoML; identical 549-feature transform from common.py):
  main  — trained on sim_v3_train (80k mixes, full training pools);
  A / B — the gate's independent cross-check: trained on trainA / trainB,
          which are disjoint in BOTH background windows and passage clips;
  NM    — ablation for Godson's Phase 4 brief point B: the identical
          training examples minus the mixture-background ones, so the
          effect of background mixtures on the realism diagnostics is
          measured rather than asserted.

Validation: the identical holdout protocol as v1/v2 (val_random,
val_site, val_period, val_long from the v3 generator), scored for B0,
the v2 model, main, A and B, with pressure maps for main (share band,
site, missing minutes, background mode, construction order).

Consistency diagnostics required by the phase brief:
  1. oracle ceiling of the v3 construction (true-background subtraction
     on the oracle set, est_from_bg);
  2. prediction distribution on the REAL test windows (inference only —
     nothing is fitted, calibrated or tuned on Test.csv): the mean must
     sit in the plausible prior band 0.13-0.20 derived in Phase 3
     section 4, not at v2's 0.330.

Test predictions of the main model are written to
models/sim_v3_test_predictions.csv only. Whether they become a
submission file is decided by the gate judgement in RESULTS_sim_v3.md,
not by this script.
"""
import json
import platform
from multiprocessing import Pool
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
import scipy
from common import DATA, ROOT, SEED, featurize
from phase3_gen import est_from_bg

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, verbose=-1, num_threads=2)
NUM_ROUNDS = 6000
ES_PATIENCE = 200
VAL_SETS = ["val_random", "val_site", "val_period", "val_long"]
PRIOR_BAND = (0.13, 0.20)


def _feat_one(x):
    return featurize(x)[0]


def featurize_mp(X, procs=4):
    with Pool(procs) as p:
        F = p.map(_feat_one, [X[i] for i in range(len(X))], chunksize=256)
    return np.asarray(F, dtype=np.float32)


def load_features(tag, X):
    cache = DATA / f"feat_{tag}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"])
    F = featurize_mp(X)
    _, names = featurize(X[0])
    np.savez_compressed(cache, F=F, names=np.array(names))
    return F, names


def load(prefix, name):
    d = np.load(DATA / f"{prefix}_{name}.npz", allow_pickle=True)
    return d


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def score_entry(y, pred):
    return dict(rmse=rmse(pred, y), score=1 - rmse(pred, y),
                null_mean_pred=float(pred[y == 0].mean()) if (y == 0).any() else None)


def pressure_maps(d, pred):
    X, y, sites = d["X"], d["y"].astype(np.float64), d["site"]
    pm = {}
    bands = [(0.0, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
    by_band = {}
    for lo, hi in bands:
        m = (y == 0) if hi == 0.0 else ((y > lo) & (y <= hi))
        if m.sum() > 0:
            by_band[f"{lo}-{hi}"] = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
    pm["by_share_band"] = by_band
    by_site = {}
    for s in np.unique(sites):
        m = sites == s
        if m.sum() >= 30:
            by_site[str(s)] = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
    pm["by_site"] = by_site
    nmiss = np.isnan(X).any(axis=2).sum(axis=1)
    pm["by_n_missing_minutes"] = {
        str(k): dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
        for k, m in [(0, nmiss == 0), (1, nmiss == 1), (2, nmiss >= 2)] if m.sum() > 0}
    for key in ["mode", "order"]:
        ent = {}
        for v in np.unique(d[key]):
            m = d[key] == v
            e = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]))
            if (y[m] == 0).any():
                # Null behaviour on this background type: does the model
                # count pre-existing (natural) vessel-like energy toward
                # the label? It must not (brief point A).
                e["null_mean_pred"] = float(pred[m & (y == 0)].mean())
            ent[str(v)] = e
        pm[f"by_{key}"] = ent
    return pm


def train_model(F, y, names, tag):
    model_path = MODEL_DIR / f"lgbm_sim_v3{tag}.txt"
    iter_path = MODEL_DIR / f"lgbm_sim_v3{tag}_best_iter.txt"
    if model_path.exists() and iter_path.exists():
        m = lgb.Booster(model_file=str(model_path))
        print(f"[{tag}] loaded existing model, best iter "
              f"{iter_path.read_text().strip()}", flush=True)
        return m, {}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(F))
    cut = int(0.9 * len(F))
    tri, vai = perm[:cut], perm[cut:]
    dtrain = lgb.Dataset(F[tri], y[tri], feature_name=names)
    dval = lgb.Dataset(F[vai], y[vai], reference=dtrain)
    m = lgb.train(PARAMS, dtrain, num_boost_round=NUM_ROUNDS, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best = int(m.best_iteration)
    print(f"[{tag}] best iteration {best} of cap {NUM_ROUNDS}", flush=True)
    m.save_model(str(model_path), num_iteration=best)
    iter_path.write_text(str(best))
    imp = pd.Series(m.feature_importance("gain"), index=names)
    imp.sort_values(ascending=False).to_csv(
        MODEL_DIR / f"lgbm_sim_v3{tag}_importance.csv")
    return m, dict(num_iteration=best)


def main():
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, scipy=scipy.__version__,
                    lightgbm=lgb.__version__, sklearn=sklearn.__version__)
    print("versions:", versions, flush=True)
    results = dict(versions=versions, seed=SEED,
                   training=dict(num_boost_round_cap=NUM_ROUNDS,
                                 es_patience=ES_PATIENCE, params=PARAMS),
                   val_sets={}, diagnostics={})

    models = {}
    for tag, setname in [("", "train"), ("_A", "trainA"), ("_B", "trainB")]:
        d = load("sim_v3", setname)
        print(f"featurizing v3 {setname}...", flush=True)
        F, names = load_features(f"v3_{setname}", d["X"])
        m, kw = train_model(F, d["y"].astype(np.float64), names, tag)
        models[tag] = (m, kw)
        results["training"][f"model{tag or '_main'}"] = dict(
            set=setname, n=int(len(F)), kw=kw)
        print(f"model{tag or '_main'} ready", flush=True)

    # Ablation for brief point B: identical training examples minus the
    # mixture-background ones. Answers whether background mixtures
    # measurably change the realism diagnostics / validation scores.
    d_tr = load("sim_v3", "train")
    F_tr, names = load_features("v3_train", d_tr["X"])
    keep = d_tr["mode"] != "mixture"
    m, kw = train_model(F_tr[keep], d_tr["y"].astype(np.float64)[keep],
                        names, "_NM")
    models["_NM"] = (m, kw)
    results["training"]["model_NM"] = dict(
        set="train minus mixture-mode examples", n=int(keep.sum()), kw=kw)
    print("model_NM (no-mixture ablation) ready", flush=True)

    model_v2 = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_sim_v2.txt"))

    def predict(model_kw, F):
        m, kw = model_kw
        return np.clip(m.predict(F, **kw), 0, 1)

    preds = {}
    for name in VAL_SETS:
        d = load("sim_v3", name)
        y = d["y"].astype(np.float64)
        F, _ = load_features(f"v3_{name}", d["X"])
        b0 = np.full_like(y, 0.1)
        entry = dict(n=int(len(y)),
                     b0=dict(rmse=rmse(b0, y), score=1 - rmse(b0, y)),
                     v2_model=score_entry(y, np.clip(model_v2.predict(F), 0, 1)))
        pp = {}
        for tag in ["", "_A", "_B", "_NM"]:
            pp[tag] = predict(models[tag], F)
            entry[f"v3_model{tag or '_main'}"] = score_entry(y, pp[tag])
        entry["v3_model_NM"].update(pressure_maps(d, pp["_NM"]))
        entry["agreement_rmse"] = dict(
            A_B=rmse(pp["_A"], pp["_B"]),
            main_A=rmse(pp[""], pp["_A"]),
            main_B=rmse(pp[""], pp["_B"]))
        entry["v3_model_main"].update(pressure_maps(d, pp[""]))
        results["val_sets"][name] = entry
        preds[name] = pp
        print(f"v3 set {name}: B0 {entry['b0']['score']:.4f} | "
              f"v2 model {entry['v2_model']['score']:.4f} | "
              f"v3 main {entry['v3_model_main']['score']:.4f} | "
              f"A {entry['v3_model_A']['score']:.4f} | "
              f"B {entry['v3_model_B']['score']:.4f} | "
              f"A-B agree {entry['agreement_rmse']['A_B']:.4f}", flush=True)

    # Diagnostic 1: oracle ceiling of the v3 construction.
    d = load("sim_v3", "oracle")
    est = np.array([est_from_bg(d["X"][i].astype(np.float64),
                                d["B"][i].astype(np.float64))
                    for i in range(len(d["y"]))])
    yo = d["y"].astype(np.float64)
    results["diagnostics"]["oracle_ceiling"] = dict(
        n=int(len(yo)), rmse=rmse(est, yo), score=1 - rmse(est, yo))
    print("oracle ceiling:", results["diagnostics"]["oracle_ceiling"], flush=True)

    # Diagnostic 2: real test-window prediction distribution (inference
    # only; nothing fitted on Test.csv).
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft, _ = load_features("test", Xt)
    dist = {}
    main_pred = predict(models[""], Ft)
    for tag in ["", "_A", "_B", "_NM"]:
        s = predict(models[tag], Ft)
        dist[tag or "main"] = dict(
            mean=float(s.mean()), median=float(np.median(s)),
            frac_below_0p05=float((s < 0.05).mean()),
            deciles=np.percentile(s, [10, 25, 50, 75, 90]).round(4).tolist())
    dist["prior_band_mean"] = list(PRIOR_BAND)
    dist["main_mean_in_band"] = bool(PRIOR_BAND[0] <= dist["main"]["mean"] <= PRIOR_BAND[1])
    results["diagnostics"]["test_prediction_distribution"] = dist
    print("test prediction distribution:", json.dumps(dist, indent=1), flush=True)

    sub = pd.DataFrame(dict(ID=test["ID"], share_added_vessel=main_pred,
                            score_added_vessel=np.sqrt(main_pred)))
    sub.to_csv(MODEL_DIR / "sim_v3_test_predictions.csv", index=False)
    print("wrote models/sim_v3_test_predictions.csv (NOT a submission)", flush=True)

    (MODEL_DIR / "sim_v3_validation.json").write_text(json.dumps(results, indent=1))
    print("wrote models/sim_v3_validation.json", flush=True)


if __name__ == "__main__":
    main()
