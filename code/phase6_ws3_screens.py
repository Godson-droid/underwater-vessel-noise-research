"""Phase 6 workstream 3: feature-family screens. One family at a
time added to the frozen 549, trained on trainA, paired against
baseline A on identical windows. Families per phase6_features.py:
A per-band excess, B temporal concentration, C spectral shape
residuals. Output: models/phase6_ws3_screens.json.
"""
import json
import sys
import numpy as np
import pandas as pd
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, fmt_scores)
from phase6_features import load_extra

SETS = ["trainA"] + VAL_SETS
base_feats, ys, names = {}, {}, None
for s in SETS:
    base_feats[s], names = load_base_features(s)
    ys[s] = load_set(s)["y"].astype(np.float64)
Xs = {s: load_set(s)["X"] for s in SETS}
test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
Ft_base, _ = load_test_features()

baseA = baseline_model("_A")
out = dict(baseline_A={k: v for k, v in
                       eval_holdouts(baseA, base_feats, ys).items()},
           baseline_A_real_mean=float(predict(baseA, Ft_base).mean()),
           variants={})
print(f"baseline A: {fmt_scores(out['baseline_A'])}", flush=True)

which = sys.argv[1:] or ["A", "B", "C"]
for fam in which:
    feats, feats_t = {}, None
    xn = None
    for s in SETS:
        Fx, xn = load_extra(f"conv_bandext_maskfixf_{s}", Xs[s], (fam,))
        feats[s] = np.concatenate([base_feats[s], Fx], axis=1)
    Fx_t, _ = load_extra("test", Xt, (fam,))
    feats_t = np.concatenate([Ft_base, Fx_t], axis=1)
    all_names = names + xn
    m, mkw, info = train_lgbm(feats["trainA"], ys["trainA"], all_names)
    sc = eval_holdouts((m, mkw), feats, ys)
    real = float(predict((m, mkw), feats_t).mean())
    out["variants"][f"fam{fam}"] = dict(
        n_extra=len(xn), scores=sc, real_mean=real,
        best_iteration=info["best_iteration"])
    print(f"fam{fam} (+{len(xn)} feats): {fmt_scores(sc)} | real mean "
          f"{real:.4f} | best iter {info['best_iteration']}", flush=True)
    (MODEL_DIR / "phase6_ws3_screens.json").write_text(
        json.dumps(out, indent=1))
print("done", flush=True)
