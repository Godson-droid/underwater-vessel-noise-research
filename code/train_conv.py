"""Phase 5 training and evaluation for passage-conversion variants.

Frozen protocol: identical LightGBM hyperparameters, round cap, early
stopping and internal split as train_sim_v3.py (the conversion is the
only axis under test). Identical 549-feature transform. Test.csv is
inference-only (real-window distribution diagnostic).

Modes:
  screen — one model on the variant's trainA (40k). Reports own-holdout
           scores (all four, separately), the paired cross-scores vs
           the v3 A model (variant windows scored by v3 A; v3 windows
           scored by the variant model), the variant construction's
           oracle ceiling, and the real-window distribution.
  full   — main / A / B models on the full v3-sized sets. Adds A-B
           agreement, pressure maps, an error decomposition by share
           band (each band's share of total squared error), the
           clean/ambiguous/unresolved confidence categories (fixed
           thresholds: max pairwise |main-A-B| <= 0.05 clean,
           <= 0.15 ambiguous, else unresolved), and the
           distance-to-aim block (aim 0.998; construction oracle
           ceiling) per holdout and cross-check.

Usage: python train_conv.py <tag> <screen|full>
"""
import json
import platform
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
import scipy
from common import DATA, ROOT, SEED, featurize
from phase3_gen import est_from_bg
from train_sim_v3 import PARAMS, NUM_ROUNDS, ES_PATIENCE, pressure_maps, \
    score_entry, rmse

MODEL_DIR = ROOT / "models"
VAL_SETS = ["val_random", "val_site", "val_period", "val_long"]
PRIOR_BAND = (0.13, 0.20)
AIM = 0.998
CONF_TIGHT, CONF_LOOSE = 0.05, 0.15


def _feat_one(x):
    return featurize(x)[0]


def featurize_mp(X, procs=4):
    with Pool(procs) as p:
        F = p.map(_feat_one, [X[i] for i in range(len(X))], chunksize=256)
    return np.asarray(F, dtype=np.float32)


def load_features(tag, X):
    cache = DATA / f"feat_{tag}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["F"], list(d["names"])
    F = featurize_mp(X)
    _, names = featurize(X[0])
    np.savez_compressed(cache, F=F, names=np.array(names))
    return F, names


def load(prefix, name):
    return np.load(DATA / f"{prefix}_{name}.npz", allow_pickle=True)


def train_model(F, y, names, filename):
    model_path = MODEL_DIR / filename
    iter_path = MODEL_DIR / (filename.replace(".txt", "") + "_best_iter.txt")
    if model_path.exists() and iter_path.exists():
        m = lgb.Booster(model_file=str(model_path))
        print(f"loaded existing {filename}, best iter "
              f"{iter_path.read_text().strip()}", flush=True)
        return m, dict(num_iteration=int(iter_path.read_text().strip()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(F))
    cut = int(0.9 * len(F))
    tri, vai = perm[:cut], perm[cut:]
    dtrain = lgb.Dataset(F[tri], y[tri], feature_name=names)
    dval = lgb.Dataset(F[vai], y[vai], reference=dtrain)
    m = lgb.train(PARAMS, dtrain, num_boost_round=NUM_ROUNDS,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best = int(m.best_iteration)
    print(f"{filename}: best iteration {best} of cap {NUM_ROUNDS}", flush=True)
    m.save_model(str(model_path), num_iteration=best)
    iter_path.write_text(str(best))
    imp = pd.Series(m.feature_importance("gain"), index=names)
    imp.sort_values(ascending=False).to_csv(
        MODEL_DIR / (filename.replace(".txt", "") + "_importance.csv"))
    return m, dict(num_iteration=best)


def predict(model_kw, F):
    m, kw = model_kw
    return np.clip(m.predict(F, **kw), 0, 1)


def dist_stats(s):
    return dict(mean=float(s.mean()), median=float(np.median(s)),
                frac_below_0p05=float((s < 0.05).mean()),
                deciles=np.percentile(s, [10, 25, 50, 75, 90]).round(4).tolist())


def oracle_ceiling(prefix):
    d = load(prefix, "oracle")
    est = np.array([est_from_bg(d["X"][i].astype(np.float64),
                                d["B"][i].astype(np.float64))
                    for i in range(len(d["y"]))])
    yo = d["y"].astype(np.float64)
    return dict(n=int(len(yo)), rmse=rmse(est, yo), score=1 - rmse(est, yo))


def sse_decomposition(y, pred):
    """Each share band's share of the total squared error."""
    bands = [(0.0, 0.0, "0"), (0.0, 0.1, "0-0.1"), (0.1, 0.3, "0.1-0.3"),
             (0.3, 0.6, "0.3-0.6"), (0.6, 1.0, "0.6-1.0")]
    tot = float(((pred - y) ** 2).sum())
    out = {}
    for lo, hi, lab in bands:
        m = (y == 0) if hi == 0.0 else ((y > lo) & (y <= hi))
        if m.sum() > 0:
            sse = float(((pred[m] - y[m]) ** 2).sum())
            out[lab] = dict(n=int(m.sum()), rmse=rmse(pred[m], y[m]),
                            sse_share=sse / tot if tot > 0 else 0.0,
                            bias=float((pred[m] - y[m]).mean()))
    return out


def confidence_categories(p_main, p_a, p_b, y=None):
    mx = np.maximum(np.maximum(np.abs(p_main - p_a), np.abs(p_main - p_b)),
                    np.abs(p_a - p_b))
    cats = {}
    for lab, m in [("clean", mx <= CONF_TIGHT),
                   ("ambiguous", (mx > CONF_TIGHT) & (mx <= CONF_LOOSE)),
                   ("unresolved", mx > CONF_LOOSE)]:
        e = dict(n=int(m.sum()), frac=float(m.mean()),
                 pred_mean=float(p_main[m].mean()) if m.sum() else None)
        if y is not None and m.sum() > 0:
            e["rmse"] = rmse(p_main[m], y[m])
        cats[lab] = e
    return cats


def main():
    tag, mode = sys.argv[1], sys.argv[2]
    assert mode in ("screen", "full")
    prefix = f"sim_c{'f' if mode == 'full' else 's'}_{tag}"
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, scipy=scipy.__version__,
                    lightgbm=lgb.__version__, sklearn=sklearn.__version__)
    results = dict(tag=tag, mode=mode, versions=versions, seed=SEED,
                   aim=AIM, training=dict(num_boost_round_cap=NUM_ROUNDS,
                                          es_patience=ES_PATIENCE,
                                          params=PARAMS),
                   val_sets={}, diagnostics={})
    print("versions:", versions, flush=True)

    # Test features (cached from the v1/v3 runs; inference only).
    test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
    fcols = [c for c in test.columns if c != "ID"]
    Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
    Ft, _ = load_features("test", Xt)

    v3A = (lgb.Booster(model_file=str(MODEL_DIR / "lgbm_sim_v3_A.txt")),
           dict(num_iteration=int(
               (MODEL_DIR / "lgbm_sim_v3_A_best_iter.txt").read_text())))
    v3main = (lgb.Booster(model_file=str(MODEL_DIR / "lgbm_sim_v3.txt")),
              dict(num_iteration=int(
                  (MODEL_DIR / "lgbm_sim_v3_best_iter.txt").read_text())))
    v3B = (lgb.Booster(model_file=str(MODEL_DIR / "lgbm_sim_v3_B.txt")),
           dict(num_iteration=int(
               (MODEL_DIR / "lgbm_sim_v3_B_best_iter.txt").read_text())))

    if mode == "screen":
        d = load(prefix, "trainA")
        F, names = load_features(f"conv_{tag}s_trainA", d["X"])
        model = train_model(F, d["y"].astype(np.float64), names,
                            f"lgbm_conv_{tag}_scrA.txt")
        results["training"]["model_scrA"] = dict(set="trainA", n=int(len(F)))
        for name in VAL_SETS:
            dv = load(prefix, name)
            y = dv["y"].astype(np.float64)
            Fv, _ = load_features(f"conv_{tag}s_{name}", dv["X"])
            p = predict(model, Fv)
            entry = dict(n=int(len(y)),
                         b0=dict(rmse=rmse(np.full_like(y, 0.1), y),
                                 score=1 - rmse(np.full_like(y, 0.1), y)),
                         model=score_entry(y, p),
                         v3A_cross=score_entry(y, predict(v3A, Fv)))
            entry["model"].update(pressure_maps(dv, p))
            # Paired direction: variant model on the identical v3 windows.
            d3 = load("sim_v3", name)
            y3 = d3["y"].astype(np.float64)
            F3, _ = load_features(f"v3_{name}", d3["X"])
            entry["paired_on_v3_windows"] = dict(
                variant_model=score_entry(y3, predict(model, F3)),
                v3A_model=score_entry(y3, predict(v3A, F3)),
                v3main_model=score_entry(y3, predict(v3main, F3)))
            results["val_sets"][name] = entry
            print(f"[{tag}] {name}: model {entry['model']['score']:.4f} | "
                  f"v3A cross {entry['v3A_cross']['score']:.4f} | "
                  f"on v3 windows: variant "
                  f"{entry['paired_on_v3_windows']['variant_model']['score']:.4f}"
                  f" vs v3A "
                  f"{entry['paired_on_v3_windows']['v3A_model']['score']:.4f}",
                  flush=True)
        results["diagnostics"]["oracle_ceiling"] = oracle_ceiling(prefix)
        print(f"[{tag}] oracle ceiling:",
              results["diagnostics"]["oracle_ceiling"], flush=True)
        p_test = predict(model, Ft)
        results["diagnostics"]["test_prediction_distribution"] = dict(
            model=dist_stats(p_test), prior_band_mean=list(PRIOR_BAND),
            mean_in_band=bool(PRIOR_BAND[0] <= float(p_test.mean())
                              <= PRIOR_BAND[1]))
        print(f"[{tag}] real-window dist:",
              json.dumps(results["diagnostics"]["test_prediction_distribution"]),
              flush=True)
        sub = pd.DataFrame(dict(ID=test["ID"], share_added_vessel=p_test,
                                score_added_vessel=np.sqrt(p_test)))
        sub.to_csv(MODEL_DIR / f"conv_{tag}_scrA_test_predictions.csv",
                   index=False)
        (MODEL_DIR / f"conv_{tag}_screen.json").write_text(
            json.dumps(results, indent=1))
        print(f"wrote models/conv_{tag}_screen.json", flush=True)
        return

    # ---------------- full mode ----------------
    models = {}
    for suffix, setname in [("", "train"), ("_A", "trainA"), ("_B", "trainB")]:
        d = load(prefix, setname)
        F, names = load_features(f"conv_{tag}f_{setname}", d["X"])
        models[suffix] = train_model(
            F, d["y"].astype(np.float64), names,
            f"lgbm_conv_{tag}{suffix}.txt")
        results["training"][f"model{suffix or '_main'}"] = dict(
            set=setname, n=int(len(F)))
        print(f"[{tag}] model{suffix or '_main'} ready", flush=True)

    ceil = oracle_ceiling(prefix)
    results["diagnostics"]["oracle_ceiling"] = ceil
    print(f"[{tag}] oracle ceiling:", ceil, flush=True)

    test_preds = {sfx: predict(models[sfx], Ft) for sfx in ["", "_A", "_B"]}
    results["diagnostics"]["test_prediction_distribution"] = {
        (sfx or "main"): dist_stats(test_preds[sfx]) for sfx in test_preds}
    results["diagnostics"]["test_prediction_distribution"]["prior_band_mean"] \
        = list(PRIOR_BAND)
    results["diagnostics"]["confidence_categories_test"] = \
        confidence_categories(test_preds[""], test_preds["_A"], test_preds["_B"])
    print(f"[{tag}] real dist main:",
          json.dumps(results["diagnostics"]
                     ["test_prediction_distribution"]["main"]), flush=True)
    print(f"[{tag}] confidence (test):",
          json.dumps(results["diagnostics"]["confidence_categories_test"]),
          flush=True)

    synth_preds, synth_y = [], []
    for name in VAL_SETS:
        dv = load(prefix, name)
        y = dv["y"].astype(np.float64)
        Fv, _ = load_features(f"conv_{tag}f_{name}", dv["X"])
        pp = {sfx: predict(models[sfx], Fv) for sfx in ["", "_A", "_B"]}
        entry = dict(
            n=int(len(y)),
            b0=dict(rmse=rmse(np.full_like(y, 0.1), y),
                    score=1 - rmse(np.full_like(y, 0.1), y)),
            v3main_cross=score_entry(y, predict(v3main, Fv)),
            v3A_cross=score_entry(y, predict(v3A, Fv)),
            v3B_cross=score_entry(y, predict(v3B, Fv)))
        for sfx in ["", "_A", "_B"]:
            entry[f"model{sfx or '_main'}"] = score_entry(y, pp[sfx])
        entry["model_main"].update(pressure_maps(dv, pp[""]))
        entry["error_decomposition_main"] = sse_decomposition(y, pp[""])
        entry["agreement_rmse"] = dict(
            A_B=rmse(pp["_A"], pp["_B"]), main_A=rmse(pp[""], pp["_A"]),
            main_B=rmse(pp[""], pp["_B"]))
        entry["confidence_categories"] = confidence_categories(
            pp[""], pp["_A"], pp["_B"], y)
        entry["distance_to_aim"] = dict(
            aim=AIM,
            main_gap=AIM - entry["model_main"]["score"],
            A_gap=AIM - entry["model_A"]["score"],
            B_gap=AIM - entry["model_B"]["score"],
            construction_ceiling=ceil["score"],
            main_gap_to_ceiling=ceil["score"] - entry["model_main"]["score"])
        # Paired direction on identical v3 windows.
        d3 = load("sim_v3", name)
        y3 = d3["y"].astype(np.float64)
        F3, _ = load_features(f"v3_{name}", d3["X"])
        entry["paired_on_v3_windows"] = dict(
            variant_main=score_entry(y3, predict(models[""], F3)),
            variant_A=score_entry(y3, predict(models["_A"], F3)),
            variant_B=score_entry(y3, predict(models["_B"], F3)),
            v3main=score_entry(y3, predict(v3main, F3)),
            v3A=score_entry(y3, predict(v3A, F3)),
            v3B=score_entry(y3, predict(v3B, F3)))
        results["val_sets"][name] = entry
        synth_preds.append(pp)
        synth_y.append(y)
        print(f"[{tag}] {name}: main {entry['model_main']['score']:.4f} | "
              f"A {entry['model_A']['score']:.4f} | "
              f"B {entry['model_B']['score']:.4f} | "
              f"A-B agree {entry['agreement_rmse']['A_B']:.4f}", flush=True)

    pm = np.concatenate([p[""] for p in synth_preds])
    pa = np.concatenate([p["_A"] for p in synth_preds])
    pb = np.concatenate([p["_B"] for p in synth_preds])
    yy = np.concatenate(synth_y)
    results["diagnostics"]["confidence_categories_synthetic_pooled"] = \
        confidence_categories(pm, pa, pb, yy)

    sub = pd.DataFrame(dict(ID=test["ID"],
                            share_added_vessel=test_preds[""],
                            score_added_vessel=np.sqrt(test_preds[""])))
    sub.to_csv(MODEL_DIR / f"conv_{tag}_test_predictions.csv", index=False)
    (MODEL_DIR / f"conv_{tag}_full.json").write_text(json.dumps(results,
                                                               indent=1))
    print(f"wrote models/conv_{tag}_full.json", flush=True)


if __name__ == "__main__":
    main()
