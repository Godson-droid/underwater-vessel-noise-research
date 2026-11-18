"""Phase 8B full protocol: graded-transplant survivors.

Trains the surviving strength levels' main / A / B models in
both Phase 7 configurations (549 features; 549 + family D)
under the frozen protocol, checkpointed per model exactly as
phase8_train_eval.py, then evaluates ALL families on the
identical frozen bandext_maskfix holdouts with predictions
recomputed from the saved boosters, so every comparison is
paired and exact:

  - Phase 5 baseline
  - Phase 7 char 549 / 549+D
  - Phase 8A excl 549 / 549+D
  - Phase 8B graded, per surviving level, 549 / 549+D

Per family: four holdout scores (main/A/B), A-B agreement,
the site-holdout share-band table, Bird Island and Fish Hoek
separately, the top two bands' share of site SSE, and the
real-window diagnostics (mean; clean / ambiguous / unresolved
categories from the main/A/B test predictions) - diagnostics
only, never a validation substitute.
Record: models/phase8b_graded.json. Diagnostic test
predictions go to models/ only - NOT a submission file.

Usage: python phase8b_train_eval.py <surviving levels csv>
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA
from phase6_lib import (MODEL_DIR, VAL_SETS, load_set, load_base_features,
                        load_test_features, baseline_model, predict,
                        train_lgbm, confidence_categories)
from phase7_features import load_extra_d
from phase8_train_eval import family_metrics, load_booster
from train_conv import load_features


def main():
    levels = [int(v) for v in sys.argv[1].split(",")]
    tags = {lv: "p8b_s{}".format(lv) for lv in levels}

    # ---- features: graded training sets, both configs ----
    feats = {lv: {"base": {}, "D": {}} for lv in levels}
    ys = {}
    names = None
    names_D = None
    for lv in levels:
        for s in ["train", "trainA", "trainB"]:
            d = np.load(DATA / "sim_{}_{}.npz".format(tags[lv], s),
                        allow_pickle=True)
            Fb, names = load_features("{}_{}".format(tags[lv], s),
                                      d["X"])
            Fx, xn = load_extra_d("{}_{}".format(tags[lv], s),
                                  d["X"])
            feats[lv]["base"][s] = Fb
            feats[lv]["D"][s] = np.concatenate([Fb, Fx], axis=1)
            ys[s] = d["y"].astype(np.float64)
            print("{} {}: base {}, D {}".format(
                tags[lv], s, Fb.shape, feats[lv]["D"][s].shape),
                flush=True)
        names_D = names + xn

    # ---- features: frozen holdouts + test, both configs ----
    hfeat = {"base": {}, "D": {}}
    for s in VAL_SETS:
        d = load_set(s)
        Fb, _ = load_base_features(s)
        Fx, _ = load_extra_d("conv_bandext_maskfixf_{}".format(s),
                             d["X"])
        hfeat["base"][s] = Fb
        hfeat["D"][s] = np.concatenate([Fb, Fx], axis=1)
        ys[s] = d["y"].astype(np.float64)
    site_arr = load_set("val_site")["site"]
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft_b, _ = load_test_features()
    Fx_t, _ = load_extra_d("test", Xt)
    tfeat = {"base": Ft_b, "D": np.concatenate([Ft_b, Fx_t],
                                               axis=1)}

    # ---- train graded models (checkpointed) ----
    graded_models = {}
    train_info = {}
    for lv in levels:
        graded_models[lv] = {}
        for tag in ["base", "D"]:
            fam = "phase8b_s{}_{}".format(lv, tag)
            iters_path = MODEL_DIR / (fam + "_iters.json")
            iters = (json.loads(iters_path.read_text())
                     if iters_path.exists() else {})
            graded_models[lv][tag] = {}
            for suffix, setname in [("", "train"), ("_A", "trainA"),
                                    ("_B", "trainB")]:
                key = suffix.replace("_", "") or "main"
                path = MODEL_DIR / "{}{}.txt".format(fam, suffix)
                if path.exists() and key in iters:
                    graded_models[lv][tag][suffix] = load_booster(
                        path, iters, key)
                    train_info["{}/{}".format(fam, key)] = dict(
                        best_iteration=iters[key], resumed=True)
                    print("{} {} resumed (iter {})".format(
                        fam, key, iters[key]), flush=True)
                    continue
                m, mkw, info = train_lgbm(
                    feats[lv][tag][setname], ys[setname],
                    names_D if tag == "D" else names)
                m.save_model(str(path),
                             num_iteration=mkw["num_iteration"])
                iters[key] = info["best_iteration"]
                iters_path.write_text(json.dumps(iters))
                graded_models[lv][tag][suffix] = (m, mkw)
                train_info["{}/{}".format(fam, key)] = dict(
                    best_iteration=info["best_iteration"])
                print("{} {} done (iter {})".format(
                    fam, key, info["best_iteration"]), flush=True)

    # ---- assemble all families ----
    p7_iters = {}
    p8_iters = {}
    for tag in ["base", "D"]:
        p7_iters[tag] = json.loads(
            (MODEL_DIR / "phase7_full_{}_iters.json".format(tag))
            .read_text())
        p8_iters[tag] = json.loads(
            (MODEL_DIR / "phase8_excl_{}_iters.json".format(tag))
            .read_text())
    families = {}
    families["phase5_baseline"] = ("base", {
        sfx: baseline_model(sfx) for sfx in ["", "_A", "_B"]})
    for tag, lab in [("base", "phase7_char_549"),
                     ("D", "phase7_char_549D")]:
        families[lab] = (tag, {
            sfx: load_booster(
                MODEL_DIR / "phase7_full_{}{}.txt".format(tag, sfx),
                p7_iters[tag], sfx.replace("_", "") or "main")
            for sfx in ["", "_A", "_B"]})
    for tag, lab in [("base", "phase8_excl_549"),
                     ("D", "phase8_excl_549D")]:
        families[lab] = (tag, {
            sfx: load_booster(
                MODEL_DIR / "phase8_excl_{}{}.txt".format(tag, sfx),
                p8_iters[tag], sfx.replace("_", "") or "main")
            for sfx in ["", "_A", "_B"]})
    for lv in levels:
        for tag, lab in [("base", "phase8b_s{}_549".format(lv)),
                         ("D", "phase8b_s{}_549D".format(lv))]:
            families[lab] = (tag, graded_models[lv][tag])

    out = dict(experiment="Phase 8B graded transplants, full "
                          "protocol",
               levels=levels, train_info=train_info, families={})
    for lab, (cfg, models) in families.items():
        preds = {s: {sfx: predict(models[sfx], hfeat[cfg][s])
                     for sfx in ["", "_A", "_B"]} for s in VAL_SETS}
        fm = family_metrics(preds, ys, site_arr)
        tp = {sfx: predict(models[sfx], tfeat[cfg])
              for sfx in ["", "_A", "_B"]}
        fm["real_window_mean_main"] = float(tp[""].mean())
        fm["real_confidence_main"] = confidence_categories(
            tp[""], tp["_A"], tp["_B"])
        fm["ab_agreement_rmse"] = {
            s: float(np.sqrt(np.mean(
                (preds[s]["_A"] - preds[s]["_B"]) ** 2)))
            for s in VAL_SETS}
        out["families"][lab] = fm
        h = fm["holdouts"]
        print("{}: random {:.4f} | site {:.4f} | period {:.4f} "
              "| long {:.4f} | real mean {:.4f}".format(
                  lab, h["random"]["main"]["score"],
                  h["site"]["main"]["score"],
                  h["period"]["main"]["score"],
                  h["long"]["main"]["score"],
                  fm["real_window_mean_main"]), flush=True)
        if lab.startswith("phase8b"):
            sub = pd.DataFrame(dict(
                ID=test["ID"], share_added_vessel=tp[""],
                score_added_vessel=np.sqrt(tp[""])))
            sub.to_csv(MODEL_DIR / "phase8b_diag_{}_test_"
                                   "predictions.csv".format(lab),
                       index=False)

    (MODEL_DIR / "phase8b_graded.json").write_text(json.dumps(
        out, indent=1))
    print("wrote models/phase8b_graded.json", flush=True)


if __name__ == "__main__":
    main()
