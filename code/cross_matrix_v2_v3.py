"""V2 vs V3 cross-matrix on IDENTICAL windows (Godson's Phase 4
requirement): score the frozen V2 model and the V3 main model on the
same holdout window sets from BOTH generations — sim_v2 sets and
sim_v3 sets (val_random / val_site / val_period / val_long, each
generation's own splits and seed) — plus both models' prediction
distributions on the real test windows (inference only).

This isolates background realism from synthetic easiness: if V3's
advantage exists only on V3 windows, the V3 synthetic task merely got
easier; if V3 also wins (or loses less) on V2 windows and its real
test distribution moves toward the plausible prior band, the gain is
in the model, not the generator.

Frozen models: models/lgbm_sim_v2.txt (trimmed at save to its best
iteration), models/lgbm_sim_v3.txt (same). Feature caches are reused
(feat_v2_*.npz / feat_v3_*.npz / feat_test.npz); the transform is the
identical common.py featurisation for both generations.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA, ROOT
from train_sim_v3 import load_features, rmse, score_entry

SETS = ["val_random", "val_site", "val_period", "val_long"]
PRIOR_BAND = (0.13, 0.20)


def main():
    m2 = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_sim_v2.txt"))
    m3 = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_sim_v3.txt"))
    out = dict(generations={}, test_distribution={})
    for gen, prefix in [("v2", "sim_v2"), ("v3", "sim_v3")]:
        out["generations"][gen] = {}
        for name in SETS:
            d = np.load(DATA / f"{prefix}_{name}.npz", allow_pickle=True)
            y = d["y"].astype(np.float64)
            F, _ = load_features(f"{gen}_{name}", d["X"])
            p2 = np.clip(m2.predict(F), 0, 1)
            p3 = np.clip(m3.predict(F), 0, 1)
            e = dict(n=int(len(y)), label_mean=float(y.mean()),
                     v2_model=score_entry(y, p2),
                     v3_model=score_entry(y, p3))
            e["v3_minus_v2_score"] = e["v3_model"]["score"] - e["v2_model"]["score"]
            out["generations"][gen][name] = e
            print(f"{gen} {name}: v2 {e['v2_model']['score']:.4f} | "
                  f"v3 {e['v3_model']['score']:.4f} | "
                  f"delta {e['v3_minus_v2_score']:+.4f}", flush=True)
    # Real test windows: both models, inference only.
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft, _ = load_features("test", Xt)
    for tag, m in [("v2_model", m2), ("v3_model", m3)]:
        s = np.clip(m.predict(Ft), 0, 1)
        out["test_distribution"][tag] = dict(
            mean=float(s.mean()), median=float(np.median(s)),
            frac_below_0p05=float((s < 0.05).mean()),
            deciles=np.percentile(s, [10, 25, 50, 75, 90]).round(4).tolist(),
            mean_in_prior_band=bool(PRIOR_BAND[0] <= s.mean() <= PRIOR_BAND[1]))
        print(f"test {tag}: mean {s.mean():.4f} median {np.median(s):.4f} "
              f"frac<0.05 {(s < 0.05).mean():.3f}", flush=True)
    (ROOT / "models" / "cross_matrix_v2_v3.json").write_text(json.dumps(out, indent=1))
    print("wrote models/cross_matrix_v2_v3.json", flush=True)


if __name__ == "__main__":
    main()
