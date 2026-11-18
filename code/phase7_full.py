"""Phase 7 full protocol for the winning screen configuration:
character-synthesis training sets (sim_p7char_*) with either the
frozen 549 features (argv 'base') or 549 + family D (argv 'D'),
frozen LightGBM protocol, main / A / B models, evaluated on the
identical frozen bandext_maskfix holdouts against the Phase 5
baseline. Checkpointed per model. Diagnostic test predictions
go to models/ only — NOT a submission file.
Record: models/phase7_full.json.
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, share_band_table, confidence_categories,
                        rmse)
from phase7_features import load_extra_d
from train_conv import load_features

USE_D = len(sys.argv) > 1 and sys.argv[1] == "D"
TAG = "D" if USE_D else "base"
SETS = ["train", "trainA", "trainB"] + VAL_SETS

feats, ys = {}, {}
names = None
for s in ["train", "trainA", "trainB"]:
    d = np.load(DATA / f"sim_p7char_{s}.npz", allow_pickle=True)
    Fb, names = load_features(f"p7char_{s}", d["X"])
    ys[s] = d["y"].astype(np.float64)
    if USE_D:
        Fx, xn = load_extra_d(f"p7char_{s}", d["X"])
        feats[s] = np.concatenate([Fb, Fx], axis=1)
    else:
        feats[s] = Fb
    print(f"p7char {s}: {feats[s].shape}", flush=True)
for s in VAL_SETS:
    Fb, _ = load_base_features(s)
    d = load_set(s)
    ys[s] = d["y"].astype(np.float64)
    if USE_D:
        Fx, xn = load_extra_d(f"conv_bandext_maskfixf_{s}", d["X"])
        feats[s] = np.concatenate([Fb, Fx], axis=1)
    else:
        feats[s] = Fb
all_names = names + (xn if USE_D else [])

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
Ft_b, _ = load_test_features()
if USE_D:
    Fx_t, _ = load_extra_d("test", Xt)
    Ft = np.concatenate([Ft_b, Fx_t], axis=1)
else:
    Ft = Ft_b

out = dict(config=f"p7char training + {'549+D' if USE_D else '549'} "
                  f"features + frozen protocol",
           n_features=int(Ft.shape[1]), models={}, val_sets={},
           diagnostics={})
iters_path = MODEL_DIR / f"phase7_full_{TAG}_iters.json"
iters = json.loads(iters_path.read_text()) if iters_path.exists() else {}
models = {}
for suffix, setname in [("", "train"), ("_A", "trainA"), ("_B", "trainB")]:
    key = suffix.replace("_", "") or "main"
    path = MODEL_DIR / f"phase7_full_{TAG}{suffix}.txt"
    if path.exists() and key in iters:
        models[suffix] = (lgb.Booster(model_file=str(path)),
                          dict(num_iteration=iters[key]))
        out["models"][key] = dict(set=setname,
                                  best_iteration=iters[key], resumed=True)
        print(f"model {key} resumed (iter {iters[key]})", flush=True)
        continue
    m, mkw, info = train_lgbm(feats[setname], ys[setname], all_names)
    m.save_model(str(path), num_iteration=mkw["num_iteration"])
    iters[key] = info["best_iteration"]
    iters_path.write_text(json.dumps(iters))
    models[suffix] = (m, mkw)
    out["models"][key] = dict(set=setname,
                              best_iteration=info["best_iteration"])
    print(f"model {key} done (iter {info['best_iteration']})", flush=True)

base = {sfx: baseline_model(sfx) for sfx in ["", "_A", "_B"]}
preds, base_preds = {}, {}
for s in VAL_SETS:
    preds[s] = {sfx: predict(models[sfx], feats[s]) for sfx in ["", "_A", "_B"]}
    Fb_s, _ = load_base_features(s)
    base_preds[s] = {sfx: predict(base[sfx], Fb_s) for sfx in ["", "_A", "_B"]}
    e = {}
    for sfx in ["", "_A", "_B"]:
        e[f"model{sfx or '_main'}"] = dict(
            score=1 - rmse(preds[s][sfx], ys[s]),
            rmse=rmse(preds[s][sfx], ys[s]),
            bias=float((preds[s][sfx] - ys[s]).mean()))
        e[f"baseline{sfx or '_main'}"] = dict(
            score=1 - rmse(base_preds[s][sfx], ys[s]))
    e["agreement_rmse_A_B"] = rmse(preds[s]["_A"], preds[s]["_B"])
    e["bands_main"] = share_band_table(ys[s], preds[s][""])
    e["bands_baseline_main"] = share_band_table(ys[s], base_preds[s][""])
    e["confidence"] = confidence_categories(
        preds[s][""], preds[s]["_A"], preds[s]["_B"], ys[s])
    out["val_sets"][s] = e
    print(f"{s}: main {e['model_main']['score']:.4f} "
          f"(base {e['baseline_main']['score']:.4f}) | "
          f"A {e['model_A']['score']:.4f} | B {e['model_B']['score']:.4f} "
          f"| A-B {e['agreement_rmse_A_B']:.4f}", flush=True)

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
sub.to_csv(MODEL_DIR / f"phase7_full_{TAG}_test_predictions.csv",
           index=False)
(MODEL_DIR / "phase7_full.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase7_full.json", flush=True)
