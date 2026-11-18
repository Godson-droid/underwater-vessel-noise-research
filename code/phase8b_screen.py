"""Phase 8B screen: one model per strength level, frozen
screen protocol (549 features, train_lgbm on the level's
trainA corpus), evaluated on the identical frozen holdouts,
paired against baseline A - mirroring phase7_ws2_screen.py.

Checkpointed per level: models/phase8b_screen_s{lv}.json plus
the booster models/phase8b_scr_s{lv}.txt; existing levels are
skipped. The s=1.0 level must replicate the Phase 7 workstream
2 screen (0.9437 / 0.9141 / 0.9448 / 0.9372; site zero bias
+0.028; (0.6,1] bias -0.088) - the harness validation.

Usage: python phase8b_screen.py [levels csv, default all]
"""
import json
import sys
import numpy as np
import pandas as pd
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, eval_holdouts, share_band_table,
                        rmse)
from train_conv import load_features

LEVELS = [0, 25, 50, 75, 100]


def run_level(lv, feats, ys, site_arr, Ft, baseA):
    rec_path = MODEL_DIR / "phase8b_screen_s{}.json".format(lv)
    if rec_path.exists():
        print("s{}: screen record exists, skipped".format(lv),
              flush=True)
        return json.loads(rec_path.read_text())
    d = np.load(DATA / "sim_p8b_s{}_trainA.npz".format(lv),
                allow_pickle=True)
    F, names = load_features("p8b_s{}_trainA".format(lv), d["X"])
    y_tr = d["y"].astype(np.float64)
    print("s{} trainA: {}".format(lv, F.shape), flush=True)
    m, mkw, info = train_lgbm(F, y_tr, names)
    m.save_model(str(MODEL_DIR / "phase8b_scr_s{}.txt".format(lv)),
                 num_iteration=mkw["num_iteration"])
    print("s{} model done (iter {})".format(lv,
                                            info["best_iteration"]),
          flush=True)
    ev = eval_holdouts((m, mkw), feats, ys)
    y_site = ys["val_site"]
    p_new = predict((m, mkw), feats["val_site"])
    per_site = {}
    for s in sorted(np.unique(site_arr).tolist()):
        mm = site_arr == s
        per_site[s] = dict(
            n=int(mm.sum()),
            score=1 - rmse(p_new[mm], y_site[mm]),
            bands=share_band_table(y_site[mm], p_new[mm]))
    pt = predict((m, mkw), Ft)
    out = dict(level_pct=lv, train_info=info, holdouts=ev,
               site_bands=share_band_table(y_site, p_new),
               per_site=per_site, real_mean=float(pt.mean()))
    rec_path.write_text(json.dumps(out, indent=1))
    print("s{}: {} | site bands 0 {:+.4f} / (0.6,1] {:+.4f} | "
          "real {:.4f}".format(
              lv, " / ".join("{:.4f}".format(ev[s]["score"])
                             for s in VAL_SETS),
              out["site_bands"]["0"]["bias"],
              out["site_bands"]["(0.6,1]"]["bias"],
              out["real_mean"]), flush=True)
    return out


def main():
    levels = ([int(v) for v in sys.argv[1].split(",")]
              if len(sys.argv) > 1 else LEVELS)
    feats, ys = {}, {}
    for s in VAL_SETS:
        feats[s], _ = load_base_features(s)
        ys[s] = load_set(s)["y"].astype(np.float64)
    site_arr = load_set("val_site")["site"]
    Ft, _ = load_test_features()
    baseA = baseline_model("_A")
    evb = eval_holdouts(baseA, feats, ys)
    print("baseline A: {}".format(
        " / ".join("{:.4f}".format(evb[s]["score"])
                   for s in VAL_SETS)), flush=True)
    results = {}
    for lv in levels:
        results[lv] = run_level(lv, feats, ys, site_arr, Ft, baseA)
    print("\nlevel | random / site / period / long | zero bias | "
          "(0.6,1] bias | real mean", flush=True)
    for lv in levels:
        o = results[lv]
        print("s{:>3} | {} | {:+.4f} | {:+.4f} | {:.4f}".format(
            lv, " / ".join("{:.4f}".format(o["holdouts"][s]["score"])
                           for s in VAL_SETS),
            o["site_bands"]["0"]["bias"],
            o["site_bands"]["(0.6,1]"]["bias"], o["real_mean"]),
            flush=True)


if __name__ == "__main__":
    main()
