"""Phase 7 workstream 1 screen: frozen 549 + family D
(self-referenced background deviation), trained on the frozen
trainA file under the frozen protocol, paired against baseline A
on the identical frozen holdouts. Judged especially on the site
holdout's zero-band bias and RMSE (the phase's central target).
Record: models/phase7_ws1.json.
"""
import json
import numpy as np
import pandas as pd
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, share_band_table, rmse)
from phase7_features import load_extra_d

SETS = ["trainA"] + VAL_SETS
feats, ys, Xs = {}, {}, {}
names = None
for s in SETS:
    Fb, names = load_base_features(s)
    d = load_set(s)
    ys[s] = d["y"].astype(np.float64)
    Xs[s] = d["X"]
    Fx, xn = load_extra_d(f"conv_bandext_maskfixf_{s}", Xs[s])
    feats[s] = np.concatenate([Fb, Fx], axis=1)
    print(f"{s}: {feats[s].shape}", flush=True)
all_names = names + xn

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
Ft_b, _ = load_test_features()
Fx_t, _ = load_extra_d("test", Xt)
Ft = np.concatenate([Ft_b, Fx_t], axis=1)

m, mkw, info = train_lgbm(feats["trainA"], ys["trainA"], all_names)
m.save_model(str(MODEL_DIR / "phase7_ws1_scrA.txt"),
             num_iteration=mkw["num_iteration"])
print(f"ws1 model done (iter {info['best_iteration']})", flush=True)

baseA = baseline_model("_A")
out = dict(workstream="ws1_familyD_screen", n_features=int(Ft.shape[1]),
           feature_names_extra=xn, train_info=info, holdouts={},
           baselineA_holdouts={}, site_bands={}, site_bands_baselineA={},
           real={})
ev = eval_holdouts((m, mkw), feats, ys)
out["holdouts"] = ev
Fb_h = {s: load_base_features(s)[0] for s in VAL_SETS}
evb = eval_holdouts(baseA, Fb_h, ys)
out["baselineA_holdouts"] = evb
for s in VAL_SETS:
    p = predict((m, mkw), feats[s])
    out["site_bands" if s == "val_site" else f"bands_{s}"] = \
        share_band_table(ys[s], p)
    pb = predict(baseA, Fb_h[s])
    out["site_bands_baselineA" if s == "val_site"
        else f"bands_{s}_baselineA"] = share_band_table(ys[s], pb)
pt = predict((m, mkw), Ft)
Fb_t, _ = load_test_features()
ptb = predict(baseA, Fb_t)
out["real"] = dict(model_mean=float(pt.mean()),
                   model_median=float(np.median(pt)),
                   baselineA_mean=float(ptb.mean()))
print("ws1:", " / ".join(f"{ev[s]['score']:.4f}" for s in VAL_SETS),
      "vs baseA", " / ".join(f"{evb[s]['score']:.4f}" for s in VAL_SETS),
      flush=True)
print("site zero band: model bias "
      f"{out['site_bands']['0']['bias']:+.4f} rmse "
      f"{out['site_bands']['0']['rmse']:.4f} | baseA bias "
      f"{out['site_bands_baselineA']['0']['bias']:+.4f} rmse "
      f"{out['site_bands_baselineA']['0']['rmse']:.4f}", flush=True)
print(f"real mean: model {pt.mean():.4f} vs baseA {ptb.mean():.4f}",
      flush=True)
(MODEL_DIR / "phase7_ws1.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase7_ws1.json", flush=True)
