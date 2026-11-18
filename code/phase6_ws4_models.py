"""Phase 6 workstream 4: model comparisons on identical windows,
frozen 549 features, trained on trainA, paired against baseline A.

Alternatives (justified, open source, no AutoML):
  xgb    — XGBoost 3.4.1, histogram tree method: the canonical
           second GBM implementation; different split finding and
           regularisation path from LightGBM.
  histgb — sklearn HistGradientBoostingRegressor: a third,
           independently implemented GBM (binning + early stopping
           on a validation fraction).
  huber  — LightGBM with the huber objective under the otherwise
           frozen protocol: tests whether the mid-band SSE
           concentration (workstream 2) is an L2-loss artefact.
  mlp    — sklearn MLPRegressor (128, 64) on standardised
           features: a smooth function class, the justified
           non-tree comparator from the PLAN ladder.
All use the frozen internal split (rng SEED permutation, 90/10)
where the library exposes it. Output: models/phase6_ws4.json.
"""
import json
import sys
import time
import numpy as np
from common import SEED
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        eval_holdouts, fmt_scores, rmse)

feats, ys = {}, {}
names = None
for s in ["trainA"] + VAL_SETS:
    feats[s], names = load_base_features(s)
    ys[s] = load_set(s)["y"].astype(np.float64)
Ft, _ = load_test_features()
baseA = baseline_model("_A")
out = dict(baseline_A=eval_holdouts(baseA, feats, ys),
           baseline_A_real_mean=float(predict(baseA, Ft).mean()),
           variants={})
_prior = MODEL_DIR / "phase6_ws4.json"
if _prior.exists():
    out["variants"] = json.loads(_prior.read_text()).get("variants", {})
print(f"baseline A: {fmt_scores(out['baseline_A'])}", flush=True)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(feats["trainA"]))
cut = int(0.9 * len(feats["trainA"]))
TRI, VAI = perm[:cut], perm[cut:]
Xtr, ytr = feats["trainA"][TRI], ys["trainA"][TRI]
Xva, yva = feats["trainA"][VAI], ys["trainA"][VAI]
Xall, yall = feats["trainA"], ys["trainA"]


def record(tag, pred_fn, extra=None):
    sc = {}
    for s in VAL_SETS:
        p = np.clip(pred_fn(feats[s]), 0, 1)
        sc[s] = dict(score=1 - rmse(p, ys[s]), rmse=rmse(p, ys[s]),
                     bias=float((p - ys[s]).mean()))
    real = float(np.clip(pred_fn(Ft), 0, 1).mean())
    out["variants"][tag] = dict(scores=sc, real_mean=real,
                                **(extra or {}))
    print(f"{tag}: {fmt_scores(sc)} | real mean {real:.4f}", flush=True)
    (MODEL_DIR / "phase6_ws4.json").write_text(json.dumps(out, indent=1))


which = sys.argv[1:] or ["xgb", "histgb", "huber", "mlp"]

if "xgb" in which:
    import xgboost as xgb
    t0 = time.time()
    m = xgb.XGBRegressor(
        n_estimators=6000, learning_rate=0.05, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        tree_method="hist", early_stopping_rounds=200,
        random_state=SEED, n_jobs=2)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"xgb best iter {m.best_iteration} ({time.time()-t0:.0f}s)",
          flush=True)
    record("xgb", lambda F: m.predict(F),
           dict(best_iteration=int(m.best_iteration)))

if "histgb" in which:
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    m = HistGradientBoostingRegressor(
        max_iter=6000, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=200, random_state=SEED)
    m.fit(Xall, yall)
    print(f"histgb iters {m.n_iter_} ({time.time()-t0:.0f}s)", flush=True)
    record("histgb", lambda F: m.predict(F),
           dict(best_iteration=int(m.n_iter_)))

if "huber" in which:
    import lightgbm as lgb
    from train_sim_v3 import PARAMS, NUM_ROUNDS, ES_PATIENCE
    t0 = time.time()
    params = dict(PARAMS, objective="huber", alpha=0.05)
    dtrain = lgb.Dataset(Xtr, ytr, feature_name=names)
    dval = lgb.Dataset(Xva, yva, reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=NUM_ROUNDS,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(ES_PATIENCE,
                                                 verbose=False)])
    print(f"huber best iter {m.best_iteration} ({time.time()-t0:.0f}s)",
          flush=True)
    record("huber",
           lambda F: m.predict(F, num_iteration=m.best_iteration),
           dict(best_iteration=int(m.best_iteration)))

if "mlp" in which:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    sc_ = StandardScaler().fit(Xall)
    m = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3,
                     learning_rate_init=1e-3, max_iter=200,
                     early_stopping=True, validation_fraction=0.1,
                     n_iter_no_change=20, random_state=SEED)
    m.fit(sc_.transform(Xall), yall)
    print(f"mlp iters {m.n_iter_} ({time.time()-t0:.0f}s)", flush=True)
    record("mlp", lambda F: m.predict(sc_.transform(F)),
           dict(best_iteration=int(m.n_iter_)))
print("done", flush=True)
