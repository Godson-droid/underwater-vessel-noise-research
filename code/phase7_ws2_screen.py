"""Phase 7 workstream 2 screen: model trained on the
character-synthesised trainA (sim_p7char_trainA, frozen 549
features, frozen protocol), paired against baseline A on the
identical frozen holdouts. val_site is additionally split by
site (Bird Island vs Fish Hoek) with zero-band bias/RMSE per
site, for this model and for baseline A.
Record: models/phase7_ws2.json.
"""
import json
import numpy as np
import pandas as pd
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, share_band_table, rmse)
from train_conv import load_features

d = np.load(DATA / "sim_p7char_trainA.npz", allow_pickle=True)
F, names = load_features("p7char_trainA", d["X"])
y_tr = d["y"].astype(np.float64)
print(f"trainA p7char: {F.shape}", flush=True)

m, mkw, info = train_lgbm(F, y_tr, names)
m.save_model(str(MODEL_DIR / "phase7_ws2_scrA.txt"),
             num_iteration=mkw["num_iteration"])
print(f"ws2 model done (iter {info['best_iteration']})", flush=True)

feats, ys = {}, {}
for s in VAL_SETS:
    feats[s], _ = load_base_features(s)
    ys[s] = load_set(s)["y"].astype(np.float64)
ev = eval_holdouts((m, mkw), feats, ys)
baseA = baseline_model("_A")
evb = eval_holdouts(baseA, feats, ys)

site_d = load_set("val_site")
site_arr = site_d["site"]
y_site = ys["val_site"]
p_new = predict((m, mkw), feats["val_site"])
p_base = predict(baseA, feats["val_site"])
per_site = {}
for s in sorted(np.unique(site_arr).tolist()):
    mm = site_arr == s
    per_site[s] = dict(
        n=int(mm.sum()),
        model=dict(score=1 - rmse(p_new[mm], y_site[mm]),
                   bands=share_band_table(y_site[mm], p_new[mm])),
        baselineA=dict(score=1 - rmse(p_base[mm], y_site[mm]),
                       bands=share_band_table(y_site[mm], p_base[mm])))

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
Ft, _ = load_test_features()
pt = predict((m, mkw), Ft)
ptb = predict(baseA, Ft)
out = dict(workstream="ws2_char_synth_screen", train_info=info,
           holdouts=ev, baselineA_holdouts=evb,
           site_bands=share_band_table(y_site, p_new),
           site_bands_baselineA=share_band_table(y_site, p_base),
           per_site=per_site,
           real=dict(model_mean=float(pt.mean()),
                     baselineA_mean=float(ptb.mean())))
print("ws2:", " / ".join(f"{ev[s]['score']:.4f}" for s in VAL_SETS),
      "vs baseA", " / ".join(f"{evb[s]['score']:.4f}" for s in VAL_SETS),
      flush=True)
for s, e in per_site.items():
    print(f"{s}: model {e['model']['score']:.4f} zero bias "
          f"{e['model']['bands']['0']['bias']:+.4f} | baseA "
          f"{e['baselineA']['score']:.4f} zero bias "
          f"{e['baselineA']['bands']['0']['bias']:+.4f}", flush=True)
print(f"real mean: model {pt.mean():.4f} vs baseA {ptb.mean():.4f}",
      flush=True)
(MODEL_DIR / "phase7_ws2.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase7_ws2.json", flush=True)
