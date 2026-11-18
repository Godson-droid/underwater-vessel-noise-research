"""Phase 6 workstream 1, strategy screens. Each variant trains on
the frozen trainA file (40k) under the frozen protocol except for
the single strategy under test, and is paired against the baseline
A model (same file) on the identical frozen holdouts + real-window
mean (diagnostic only).

Variants:
  dropb0  — the 19 band-0-specific features removed.
  loso    — early stopping on a leave-site-out internal split:
            '5 St Croix Island' (second-largest trainA site, 6,473
            windows) is the internal validation set; all other
            windows train. Chosen a priori: large enough for a
            stable stopping signal, leaves 84% of data for training.
  bal     — site-balanced sample weights w_i = N / (n_sites *
            n_site(i)) countering the pool's 65% Richards Bay skew.
Output: models/phase6_ws1_screens.json.
"""
import json
import sys
import numpy as np
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, fmt_scores,
                        band0_feature_mask, rmse)

Ft, _ = load_test_features()
baseA = baseline_model("_A")
feats, ys = {}, {}
names = None
for s in ["trainA"] + VAL_SETS:
    feats[s], names = load_base_features(s)
    ys[s] = load_set(s)["y"].astype(np.float64)
sitesA = load_set("trainA")["site"]

base_scores = eval_holdouts(baseA, feats, ys)
base_real = float(predict(baseA, Ft).mean())
print(f"baseline A: {fmt_scores(base_scores)} | real mean "
      f"{base_real:.4f}", flush=True)

_prior = MODEL_DIR / "phase6_ws1_screens.json"
out = dict(baseline_A={k: base_scores[k] for k in VAL_SETS},
           baseline_A_real_mean=base_real, variants={})
if _prior.exists():
    out["variants"] = json.loads(_prior.read_text()).get("variants", {})


def run(tag, **kw):
    m, mkw, info = train_lgbm(feats["trainA"], ys["trainA"], names, **kw)
    m.save_model(str(MODEL_DIR / f"phase6_{tag}_scrA.txt"),
                 num_iteration=mkw["num_iteration"])
    mask = kw.get("feature_mask")
    if mask is not None:
        ev = {s: feats[s][:, mask] for s in feats}
        ev_t = Ft[:, mask]
    else:
        ev, ev_t = feats, Ft
    sc = eval_holdouts((m, mkw), ev, ys)
    real = float(predict((m, mkw), ev_t).mean())
    out["variants"][tag] = dict(scores=sc, real_mean=real, info={
        k: info[k] for k in ("best_iteration", "n_train", "n_val",
                             "split_mode")})
    print(f"{tag}: {fmt_scores(sc)} | real mean {real:.4f} | "
          f"best iter {info['best_iteration']}", flush=True)
    (MODEL_DIR / "phase6_ws1_screens.json").write_text(
        json.dumps(out, indent=1))


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "dropb0"):
    run("dropb0", feature_mask=~band0_feature_mask(names))
if which in ("all", "loso"):
    run("loso", split_mode="sites",
        split_sites=(["5 St Croix Island"], sitesA))
if which in ("all", "bal"):
    uniq, cnt = np.unique(sitesA, return_counts=True)
    wmap = {s: len(sitesA) / (len(uniq) * c) for s, c in zip(uniq, cnt)}
    w = np.array([wmap[s] for s in sitesA], dtype=np.float64)
    run("bal", sample_weight=w)
print("done", flush=True)
