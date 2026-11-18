"""Phase 7 workstream 3: leave-one-site-out ensembling.

Six training sites are present in the frozen trainA backgrounds
(Richards Bay, St Croix Island, Saldanha Bay, St_Francis, Durban,
Sea Point; Cape Recife and Riy Banks yielded no screened pool
windows). One LightGBM per held-out site is trained on trainA
minus that site's windows, under the frozen protocol; the
ensemble prediction is the mean of the six members' clipped
predictions. Paired against baseline A on the identical frozen
holdouts. Godson's requirement for this workstream: improvement
on ALL FOUR holdouts, not just the site holdout.

Caveat recorded: a generated window's site label is its
background's site (for mixture backgrounds, the first
component's site), so fold exclusion is by that label.
Record: models/phase7_ws3.json.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, share_band_table, rmse)

SETS = ["trainA"] + VAL_SETS
feats, ys = {}, {}
names = None
for s in SETS:
    feats[s], names = load_base_features(s)
    ys[s] = load_set(s)["y"].astype(np.float64)
sites_tr = load_set("trainA")["site"]
uniq = sorted(np.unique(sites_tr).tolist())
print("LOSO sites:", uniq, flush=True)

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
Ft, _ = load_test_features()

members, member_preds, member_test = {}, {}, {}
for s in uniq:
    tag = s.replace(" ", "_").replace("/", "_")
    path = MODEL_DIR / f"phase7_loso_{tag}.txt"
    mask = sites_tr != s
    if path.exists():
        m = lgb.Booster(model_file=str(path))
        # best iteration was saved into the model file's trees;
        # recover via the sidecar written at train time.
        it = int((MODEL_DIR / f"phase7_loso_{tag}_iter.txt").read_text())
        mkw = dict(num_iteration=it)
        print(f"member -{s}: resumed (iter {it})", flush=True)
    else:
        m, mkw, info = train_lgbm(feats["trainA"][mask], ys["trainA"][mask],
                                  names)
        m.save_model(str(path), num_iteration=mkw["num_iteration"])
        (MODEL_DIR / f"phase7_loso_{tag}_iter.txt").write_text(
            str(info["best_iteration"]))
        print(f"member -{s}: trained (iter {info['best_iteration']}, "
              f"n={int(mask.sum())})", flush=True)
    members[s] = (m, mkw)
    member_preds[s] = {v: predict((m, mkw), feats[v]) for v in VAL_SETS}
    member_test[s] = predict((m, mkw), Ft)

ens = {v: np.clip(np.mean([member_preds[s][v] for s in uniq], axis=0), 0, 1)
       for v in VAL_SETS}
ens_test = np.clip(np.mean([member_test[s] for s in uniq], axis=0), 0, 1)

baseA = baseline_model("_A")
out = dict(workstream="ws3_loso_ensemble", sites=uniq,
           member_holdouts={}, ensemble_holdouts={},
           baselineA_holdouts={}, site_bands_ensemble={},
           site_bands_baselineA={}, member_site_scores={}, real={})
for s in uniq:
    out["member_holdouts"][s] = {
        v: dict(score=1 - rmse(member_preds[s][v], ys[v])) for v in VAL_SETS}
    out["member_site_scores"][s] = out["member_holdouts"][s]["val_site"]["score"]
for v in VAL_SETS:
    out["ensemble_holdouts"][v] = dict(
        score=1 - rmse(ens[v], ys[v]), rmse=rmse(ens[v], ys[v]),
        bias=float((ens[v] - ys[v]).mean()))
    pb = predict(baseA, feats[v])
    out["baselineA_holdouts"][v] = dict(score=1 - rmse(pb, ys[v]))
out["site_bands_ensemble"] = share_band_table(ys["val_site"], ens["val_site"])
pb_site = predict(baseA, feats["val_site"])
out["site_bands_baselineA"] = share_band_table(ys["val_site"], pb_site)
ptb = predict(baseA, Ft)
out["real"] = dict(ensemble_mean=float(ens_test.mean()),
                   baselineA_mean=float(ptb.mean()),
                   member_means={s: float(member_test[s].mean())
                                 for s in uniq})
print("ensemble:", " / ".join(
    f"{out['ensemble_holdouts'][v]['score']:.4f}" for v in VAL_SETS),
    "vs baseA", " / ".join(
        f"{out['baselineA_holdouts'][v]['score']:.4f}" for v in VAL_SETS),
    flush=True)
print("site zero band: ens bias "
      f"{out['site_bands_ensemble']['0']['bias']:+.4f} rmse "
      f"{out['site_bands_ensemble']['0']['rmse']:.4f} | baseA bias "
      f"{out['site_bands_baselineA']['0']['bias']:+.4f} rmse "
      f"{out['site_bands_baselineA']['0']['rmse']:.4f}", flush=True)
print(f"real mean: ens {ens_test.mean():.4f} vs baseA {ptb.mean():.4f}",
      flush=True)
(MODEL_DIR / "phase7_ws3.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase7_ws3.json", flush=True)
