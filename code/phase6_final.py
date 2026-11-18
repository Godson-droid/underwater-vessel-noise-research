"""Phase 6 final full protocol: the best justified combination
from the screens — frozen bandext_maskfix splits, LightGBM frozen
protocol, features = 549 base + family B (temporal concentration)
+ family C (spectral shape residuals), the two families that were
independently non-negative on every holdout at screen scale
(workstream 3). Trains main / A / B, reports all four holdouts
per model, A-B agreement, share-band table for main, real-window
distribution and confidence categories, and the gate block.
Diagnostic test predictions are written to models/ only.
Output: models/phase6_final.json.
"""
import json
import numpy as np
import pandas as pd
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, share_band_table,
                        confidence_categories, rmse)
from phase6_features import load_extra

FAMS = ("B", "C")
SETS = ["train", "trainA", "trainB"] + VAL_SETS
feats, ys, Xs = {}, {}, {}
names = None
for s in SETS:
    Fb, names = load_base_features(s)
    d = load_set(s)
    ys[s] = d["y"].astype(np.float64)
    Xs[s] = d["X"]
    Fx, xn = load_extra(f"conv_bandext_maskfixf_{s}", Xs[s], FAMS)
    feats[s] = np.concatenate([Fb, Fx], axis=1)
    print(f"{s}: {feats[s].shape}", flush=True)
all_names = names + xn

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
Ft_b, _ = load_test_features()
Fx_t, _ = load_extra("test", Xt, FAMS)
Ft = np.concatenate([Ft_b, Fx_t], axis=1)

out = dict(config="frozen splits + LightGBM frozen protocol + "
                  "549 + famB + famC features",
           n_features=int(Ft.shape[1]), models={}, val_sets={},
           diagnostics={})
import lightgbm as lgb
import os
models = {}
_iters_path = MODEL_DIR / "phase6_final_iters.json"
_iters = json.loads(_iters_path.read_text()) if _iters_path.exists() \
    else {}
for suffix, setname in [("", "train"), ("_A", "trainA"),
                        ("_B", "trainB")]:
    key = suffix.replace("_", "") or "main"
    path = MODEL_DIR / f"phase6_final{suffix}.txt"
    if path.exists() and key in _iters:
        m = lgb.Booster(model_file=str(path))
        mkw = dict(num_iteration=_iters[key])
        models[suffix] = (m, mkw)
        out["models"][suffix or "main"] = dict(
            set=setname, best_iteration=_iters[key], resumed=True)
        print(f"model{suffix or '_main'} resumed "
              f"(iter {_iters[key]})", flush=True)
        continue
    m, mkw, info = train_lgbm(feats[setname], ys[setname], all_names)
    m.save_model(str(path), num_iteration=mkw["num_iteration"])
    _iters[key] = info["best_iteration"]
    _iters_path.write_text(json.dumps(_iters))
    models[suffix] = (m, mkw)
    out["models"][suffix or "main"] = dict(
        set=setname, best_iteration=info["best_iteration"])
    print(f"model{suffix or '_main'} done "
          f"(iter {info['best_iteration']})", flush=True)

base = {sfx: baseline_model(sfx) for sfx in ["", "_A", "_B"]}
preds, base_preds = {}, {}
for s in VAL_SETS:
    preds[s] = {sfx: predict(models[sfx], feats[s])
                for sfx in ["", "_A", "_B"]}
    Fb_s, _ = load_base_features(s)
    base_preds[s] = {sfx: predict(base[sfx], Fb_s)
                     for sfx in ["", "_A", "_B"]}
    e = dict()
    for sfx in ["", "_A", "_B"]:
        e[f"model{sfx or '_main'}"] = dict(
            score=1 - rmse(preds[s][sfx], ys[s]),
            rmse=rmse(preds[s][sfx], ys[s]),
            bias=float((preds[s][sfx] - ys[s]).mean()))
        e[f"baseline{sfx or '_main'}"] = dict(
            score=1 - rmse(base_preds[s][sfx], ys[s]))
    e["agreement_rmse_A_B"] = rmse(preds[s]["_A"], preds[s]["_B"])
    e["bands_main"] = share_band_table(ys[s], preds[s][""])
    e["confidence"] = confidence_categories(
        preds[s][""], preds[s]["_A"], preds[s]["_B"], ys[s])
    out["val_sets"][s] = e
    print(f"{s}: main {e['model_main']['score']:.4f} "
          f"(base {e['baseline_main']['score']:.4f}) | "
          f"A {e['model_A']['score']:.4f} | B {e['model_B']['score']:.4f}"
          f" | A-B {e['agreement_rmse_A_B']:.4f}", flush=True)

tp = {sfx: predict(models[sfx], Ft) for sfx in ["", "_A", "_B"]}
out["diagnostics"]["test"] = {
    sfx or "main": dict(mean=float(tp[sfx].mean()),
                        median=float(np.median(tp[sfx])),
                        frac_below_0p05=float((tp[sfx] < 0.05).mean()))
    for sfx in tp}
out["diagnostics"]["confidence_categories_test"] = \
    confidence_categories(tp[""], tp["_A"], tp["_B"])
out["diagnostics"]["construction_ceiling"] = 0.9981
print("test diag:", json.dumps(out["diagnostics"]["test"]), flush=True)
print("confidence (test):",
      json.dumps(out["diagnostics"]["confidence_categories_test"]),
      flush=True)

sub = pd.DataFrame(dict(ID=test["ID"], share_added_vessel=tp[""],
                        score_added_vessel=np.sqrt(tp[""])))
sub.to_csv(MODEL_DIR / "phase6_final_test_predictions.csv", index=False)
(MODEL_DIR / "phase6_final.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase6_final.json", flush=True)
