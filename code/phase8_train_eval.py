"""Phase 8 exclusion test: training + three-way evaluation.

Trains the exclusion models (character-synthesis training sets
sim_p8excl_* with Bird Island removed from the transplant
targets) under the frozen protocol, in both Phase 7
configurations (549 features; 549 + family D), main / A / B,
checkpointed per model exactly as phase7_full.py.

Then evaluates THREE model families on the identical frozen
bandext_maskfix holdouts, all predictions recomputed here from
the saved boosters so the comparison is paired and exact:

  - Phase 5 baseline (models/lgbm_conv_bandext_maskfix*.txt)
  - Phase 7 full transplant, 549 and 549+D
    (models/phase7_full_base*.txt, phase7_full_D*.txt)
  - Phase 8 exclusion, 549 and 549+D (trained here)

Reported per family (main model unless noted): the four
holdout scores (main/A/B), the site-holdout share-band table,
Bird Island and Fish Hoek separately (score, bias, zero-band
bias/RMSE, (0.6,1] bias/RMSE), the top two bands' share of
site SSE, and the real-window mean as a diagnostic only.
Record: models/phase8_excl.json. Diagnostic test predictions
go to models/ only — NOT a submission file.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, share_band_table, rmse)
from phase7_features import load_extra_d
from train_conv import load_features

HOLDOUT_SHORT = {"val_random": "random", "val_site": "site",
                 "val_period": "period", "val_long": "long"}
TOP_BANDS = ["(0.3,0.6]", "(0.6,1]"]


def load_booster(path, iters, key):
    return (lgb.Booster(model_file=str(path)),
            dict(num_iteration=iters[key]))


def family_metrics(preds, ys, site_arr):
    """preds/ys keyed by VAL_SETS name; preds values are
    dicts suffix->prediction array."""
    out = {"holdouts": {}, "site_bands_main": None,
           "per_site_main": {}, "top_two_band_sse_share_main": None}
    for s in VAL_SETS:
        e = {}
        for sfx, lab in [("", "main"), ("_A", "A"), ("_B", "B")]:
            e[lab] = dict(score=1 - rmse(preds[s][sfx], ys[s]),
                          rmse=rmse(preds[s][sfx], ys[s]),
                          bias=float((preds[s][sfx] - ys[s]).mean()))
        out["holdouts"][HOLDOUT_SHORT[s]] = e
    y = ys["val_site"]
    p = preds["val_site"][""]
    bt = share_band_table(y, p)
    out["site_bands_main"] = bt
    out["top_two_band_sse_share_main"] = float(
        sum(bt[b]["sse_share"] for b in TOP_BANDS if b in bt))
    for sname in sorted(np.unique(site_arr).tolist()):
        mm = site_arr == sname
        bt_s = share_band_table(y[mm], p[mm])
        out["per_site_main"][sname] = dict(
            n=int(mm.sum()),
            score=1 - rmse(p[mm], y[mm]),
            bias=float((p[mm] - y[mm]).mean()),
            bands=bt_s,
            top_two_band_sse_share=float(
                sum(bt_s[b]["sse_share"] for b in TOP_BANDS
                    if b in bt_s)))
    return out


def main():
    # ---- features: exclusion training sets, both configs ----
    feats = {"base": {}, "D": {}}
    ys = {}
    names = None
    for s in ["train", "trainA", "trainB"]:
        d = np.load(DATA / f"sim_p8excl_{s}.npz", allow_pickle=True)
        Fb, names = load_features(f"p8excl_{s}", d["X"])
        Fx, xn = load_extra_d(f"p8excl_{s}", d["X"])
        feats["base"][s] = Fb
        feats["D"][s] = np.concatenate([Fb, Fx], axis=1)
        ys[s] = d["y"].astype(np.float64)
        print(f"p8excl {s}: base {Fb.shape}, D {feats['D'][s].shape}",
              flush=True)
    names_D = names + xn

    # ---- features: frozen holdouts + test, both configs ----
    hfeat = {"base": {}, "D": {}}
    for s in VAL_SETS:
        d = load_set(s)
        Fb, _ = load_base_features(s)
        Fx, _ = load_extra_d(f"conv_bandext_maskfixf_{s}", d["X"])
        hfeat["base"][s] = Fb
        hfeat["D"][s] = np.concatenate([Fb, Fx], axis=1)
        ys[s] = d["y"].astype(np.float64)
    site_arr = load_set("val_site")["site"]
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft_b, _ = load_test_features()
    Fx_t, _ = load_extra_d("test", Xt)
    tfeat = {"base": Ft_b, "D": np.concatenate([Ft_b, Fx_t], axis=1)}

    # ---- train exclusion models (checkpointed) ----
    excl_models = {}
    train_info = {}
    for tag in ["base", "D"]:
        iters_path = MODEL_DIR / f"phase8_excl_{tag}_iters.json"
        iters = (json.loads(iters_path.read_text())
                 if iters_path.exists() else {})
        excl_models[tag] = {}
        for suffix, setname in [("", "train"), ("_A", "trainA"),
                                ("_B", "trainB")]:
            key = suffix.replace("_", "") or "main"
            path = MODEL_DIR / f"phase8_excl_{tag}{suffix}.txt"
            if path.exists() and key in iters:
                excl_models[tag][suffix] = load_booster(path, iters,
                                                        key)
                train_info[f"{tag}/{key}"] = dict(
                    best_iteration=iters[key], resumed=True)
                print(f"excl {tag} {key} resumed "
                      f"(iter {iters[key]})", flush=True)
                continue
            m, mkw, info = train_lgbm(feats[tag][setname], ys[setname],
                                      names_D if tag == "D" else names)
            m.save_model(str(path), num_iteration=mkw["num_iteration"])
            iters[key] = info["best_iteration"]
            iters_path.write_text(json.dumps(iters))
            excl_models[tag][suffix] = (m, mkw)
            train_info[f"{tag}/{key}"] = dict(
                best_iteration=info["best_iteration"])
            print(f"excl {tag} {key} done "
                  f"(iter {info['best_iteration']})", flush=True)

    # ---- assemble all families ----
    p7_iters = {}
    for tag in ["base", "D"]:
        p7_iters[tag] = json.loads(
            (MODEL_DIR / f"phase7_full_{tag}_iters.json").read_text())
    families = {}   # label -> (config, {suffix: (booster, kw)})
    families["phase5_baseline"] = ("base", {
        sfx: baseline_model(sfx) for sfx in ["", "_A", "_B"]})
    for tag, lab in [("base", "phase7_char_549"),
                     ("D", "phase7_char_549D")]:
        families[lab] = (tag, {
            sfx: load_booster(MODEL_DIR / f"phase7_full_{tag}{sfx}.txt",
                              p7_iters[tag],
                              sfx.replace("_", "") or "main")
            for sfx in ["", "_A", "_B"]})
    for tag, lab in [("base", "phase8_excl_549"),
                     ("D", "phase8_excl_549D")]:
        families[lab] = (tag, excl_models[tag])

    out = dict(experiment="Phase 8 opening: Bird Island exclusion",
               train_info=train_info, families={})
    for lab, (cfg, models) in families.items():
        preds = {s: {sfx: predict(models[sfx], hfeat[cfg][s])
                     for sfx in ["", "_A", "_B"]} for s in VAL_SETS}
        fm = family_metrics(preds, ys, site_arr)
        tp = predict(models[""], tfeat[cfg])
        fm["real_window_mean_main"] = float(tp.mean())
        out["families"][lab] = fm
        h = fm["holdouts"]
        print(f"{lab}: random {h['random']['main']['score']:.4f} | "
              f"site {h['site']['main']['score']:.4f} | period "
              f"{h['period']['main']['score']:.4f} | long "
              f"{h['long']['main']['score']:.4f} | real mean "
              f"{fm['real_window_mean_main']:.4f}", flush=True)
        # diagnostic predictions only, NOT a submission file
        sub = pd.DataFrame(dict(ID=test["ID"],
                                share_added_vessel=tp,
                                score_added_vessel=np.sqrt(tp)))
        sub.to_csv(MODEL_DIR / f"phase8_diag_{lab}_test_predictions.csv",
                   index=False)

    (MODEL_DIR / "phase8_excl.json").write_text(json.dumps(out,
                                                           indent=1))
    print("wrote models/phase8_excl.json", flush=True)


if __name__ == "__main__":
    main()
