"""Phase 7 interaction experiment (the ONE small combination,
run only after the independent screens): character-synthesis
training (workstream 2's sim_p7char_trainA) x family D features
(workstream 1's 549 + 56). Rationale recorded in the report:
family D's deviation-from-own-background features may defend
the high-share side that character training discounts on
unseen sites. Screen scale (trainA), paired against baseline A
and against both parent screens on identical frozen holdouts.
Record: models/phase7_interact.json.
"""
import json
import numpy as np
import pandas as pd
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, share_band_table)
from phase7_features import load_extra_d
from train_conv import load_features

d = np.load(DATA / "sim_p7char_trainA.npz", allow_pickle=True)
Fb_tr, names = load_features("p7char_trainA", d["X"])
Fx_tr, xn = load_extra_d("p7char_trainA", d["X"])
F_tr = np.concatenate([Fb_tr, Fx_tr], axis=1)
y_tr = d["y"].astype(np.float64)
all_names = names + xn
print(f"interaction trainA: {F_tr.shape}", flush=True)

m, mkw, info = train_lgbm(F_tr, y_tr, all_names)
m.save_model(str(MODEL_DIR / "phase7_interact_scrA.txt"),
             num_iteration=mkw["num_iteration"])
print(f"interaction model done (iter {info['best_iteration']})",
      flush=True)

feats, ys = {}, {}
for s in VAL_SETS:
    Fb, _ = load_base_features(s)
    Fx, _ = load_extra_d(f"conv_bandext_maskfixf_{s}", load_set(s)["X"])
    feats[s] = np.concatenate([Fb, Fx], axis=1)
    ys[s] = load_set(s)["y"].astype(np.float64)
ev = eval_holdouts((m, mkw), feats, ys)
p_site = predict((m, mkw), feats["val_site"])
bands = share_band_table(ys["val_site"], p_site)

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
Ft_b, _ = load_test_features()
Fx_t, _ = load_extra_d("test", Xt)
pt = predict((m, mkw), np.concatenate([Ft_b, Fx_t], axis=1))

out = dict(workstream="interaction_ws2_x_ws1D", train_info=info,
           holdouts=ev, site_bands=bands,
           real=dict(model_mean=float(pt.mean())))
print("interact:", " / ".join(f"{ev[s]['score']:.4f}" for s in VAL_SETS),
      "| ws2 alone: 0.9437 / 0.9141 / 0.9448 / 0.9372",
      "| ws1 alone: 0.9445 / 0.9042 / 0.9452 / 0.9364", flush=True)
print(f"site zero band bias {bands['0']['bias']:+.4f} rmse "
      f"{bands['0']['rmse']:.4f} | (0.3,0.6] bias "
      f"{bands['(0.3,0.6]']['bias']:+.4f} | real mean {pt.mean():.4f}",
      flush=True)
(MODEL_DIR / "phase7_interact.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase7_interact.json", flush=True)
