"""Phase 6 workstream 2: share-band error analysis of the frozen
bandext_maskfix baseline (main model) on the identical frozen
holdouts. Bands per Godson's brief: 0, (0,0.05], (0.05,0.1],
(0.1,0.3], (0.3,0.6], (0.6,1]. Also: per-site split inside
val_site, and band-0-feature grouped permutation importance of the
baseline on val_random vs val_site (feeds workstream 1).
Output: models/phase6_ws2.json + printed tables.
"""
import json
import numpy as np
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        baseline_model, predict, share_band_table, rmse,
                        band0_feature_mask)

rng = np.random.default_rng(20261117)
out = dict(baseline="bandext_maskfix main")
main = baseline_model("")

feats, ys, sites = {}, {}, {}
names = None
for s in VAL_SETS:
    feats[s], names = load_base_features(s)
    d = load_set(s)
    ys[s] = d["y"].astype(np.float64)
    sites[s] = d["site"]

out["bands"] = {}
for s in VAL_SETS:
    p = predict(main, feats[s])
    out["bands"][s] = share_band_table(ys[s], p)
    print(f"== {s} (score {1 - rmse(p, ys[s]):.4f}) ==")
    for lab, r in out["bands"][s].items():
        print(f"  {lab:>10} n={r['n']:5d} frac={r['frac']:.3f} "
              f"bias={r['bias']:+.4f} rmse={r['rmse']:.4f} "
              f"sse={r['sse_share']:.3f}")

# val_site per-site breakdown
p = predict(main, feats["val_site"])
out["val_site_by_site"] = {}
for site in sorted(set(sites["val_site"].tolist())):
    m = sites["val_site"] == site
    out["val_site_by_site"][site] = dict(
        n=int(m.sum()), score=1 - rmse(p[m], ys["val_site"][m]),
        bias=float((p[m] - ys["val_site"][m]).mean()),
        bands=share_band_table(ys["val_site"][m], p[m]))
    print(f"val_site {site}: n={int(m.sum())} "
          f"score {out['val_site_by_site'][site]['score']:.4f} "
          f"bias {out['val_site_by_site'][site]['bias']:+.4f}")

# Grouped permutation importance: band-0 features vs a control band.
b0 = band0_feature_mask(names)
names_arr = np.array(names)
# control group: band-12-specific features (same construction)
ctrl = np.zeros(len(names), dtype=bool)
for i, n in enumerate(names):
    if n.startswith("cell_share_"):
        if int(n.split("_")[2]) % 25 == 12:
            ctrl[i] = True
    elif n in ("band_share_12", "band_mean_db_rel_12",
               "band_range_db_12", "band_std_db_12"):
        ctrl[i] = True
print(f"band0 group: {int(b0.sum())} features; control band12: "
      f"{int(ctrl.sum())} features")
out["permutation"] = {}
for s in ["val_random", "val_site"]:
    F, y = feats[s], ys[s]
    base = 1 - rmse(predict(main, F), y)
    res = dict(base=base)
    for lab, mask in [("band0", b0), ("band12_control", ctrl)]:
        drops = []
        for rep in range(3):
            Fp = F.copy()
            Fp[:, mask] = Fp[rng.permutation(len(F))][:, mask]
            drops.append(base - (1 - rmse(predict(main, Fp), y)))
        res[lab] = float(np.mean(drops))
    out["permutation"][s] = res
    print(f"perm {s}: base {base:.4f} | band0 drop {res['band0']:.4f} "
          f"| band12 drop {res['band12_control']:.4f}")

(MODEL_DIR / "phase6_ws2.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase6_ws2.json")
