"""Phase 6 shared harness: frozen bandext_maskfix sets, baseline
models, training with strategy options, and evaluation helpers.

Everything trains/evaluates on the identical frozen generation
files data/sim_cf_bandext_maskfix_*.npz so comparisons against the
Phase 5 baseline are paired on identical windows.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA, ROOT, SEED
from train_sim_v3 import PARAMS, NUM_ROUNDS, ES_PATIENCE

MODEL_DIR = ROOT / "models"
PREFIX = "sim_cf_bandext_maskfix"
VAL_SETS = ["val_random", "val_site", "val_period", "val_long"]
AIM = 0.998


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_set(name, prefix=PREFIX):
    return np.load(DATA / f"{prefix}_{name}.npz", allow_pickle=True)


def load_base_features(setname):
    d = np.load(DATA / f"feat_conv_bandext_maskfixf_{setname}.npz",
                allow_pickle=True)
    return d["F"], list(d["names"])


def load_test_features():
    d = np.load(DATA / "feat_test.npz", allow_pickle=True)
    return d["F"], list(d["names"])


def baseline_model(suffix=""):
    name = f"lgbm_conv_bandext_maskfix{suffix}"
    m = lgb.Booster(model_file=str(MODEL_DIR / f"{name}.txt"))
    kw = dict(num_iteration=int(
        (MODEL_DIR / f"{name}_best_iter.txt").read_text()))
    return m, kw


def predict(model_kw, F):
    m, kw = model_kw
    return np.clip(m.predict(F, **kw), 0, 1)


def band0_feature_mask(names):
    """Boolean mask over the 549 features: True = the feature is
    specific to band 0 (cell shares of band 0 across minutes, and the
    per-band summaries for band 0). Minute-level aggregates that mix
    all bands are NOT band-0-specific and are excluded."""
    out = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        if n.startswith("cell_share_"):
            idx = int(n.split("_")[2])
            if idx % 25 == 0:
                out[i] = True
        elif n in ("band_share_0", "band_mean_db_rel_0",
                   "band_range_db_0", "band_std_db_0"):
            out[i] = True
    return out


def train_lgbm(F, y, names, seed=SEED, params_over=None,
               sample_weight=None, split_mode="random",
               split_sites=None, feature_mask=None, log=print):
    """Frozen-protocol LightGBM training with Phase 6 strategy knobs.

    split_mode 'random' reproduces train_conv.train_model's internal
    90/10 split exactly (rng seeded SEED, permutation). 'sites' uses
    all windows whose site is in split_sites as the early-stopping
    validation set (leave-site-out style stopping inside the pool).
    feature_mask: boolean over columns to KEEP.
    Returns (model, kw, info dict with best_iteration and split sizes).
    """
    if feature_mask is not None:
        F = F[:, feature_mask]
        names = [n for n, k in zip(names, feature_mask) if k]
    n = len(F)
    if split_mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(0.9 * n)
        tri, vai = perm[:cut], perm[cut:]
    elif split_mode == "sites":
        vai = np.where(np.isin(split_sites[1], list(split_sites[0])))[0]
        tri = np.where(~np.isin(split_sites[1], list(split_sites[0])))[0]
    else:
        raise ValueError(split_mode)
    params = dict(PARAMS)
    if params_over:
        params.update(params_over)
    dtrain = lgb.Dataset(F[tri], y[tri], feature_name=names,
                         weight=(None if sample_weight is None
                                 else sample_weight[tri]))
    dval = lgb.Dataset(F[vai], y[vai], reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=NUM_ROUNDS,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best = int(m.best_iteration)
    log(f"trained: best iter {best} of cap {NUM_ROUNDS} "
        f"(train {len(tri)}, internal val {len(vai)})")
    return m, dict(num_iteration=best), dict(
        best_iteration=best, n_train=int(len(tri)), n_val=int(len(vai)),
        split_mode=split_mode, params=params)


def eval_holdouts(model_kw, feats_by_set, ys_by_set):
    out = {}
    for name in VAL_SETS:
        p = predict(model_kw, feats_by_set[name])
        out[name] = dict(score=1 - rmse(p, ys_by_set[name]),
                         rmse=rmse(p, ys_by_set[name]),
                         bias=float((p - ys_by_set[name]).mean()))
    return out


SHARE_BANDS = [(0.0, 0.0, "0"), (0.0, 0.05, "(0,0.05]"),
               (0.05, 0.1, "(0.05,0.1]"), (0.1, 0.3, "(0.1,0.3]"),
               (0.3, 0.6, "(0.3,0.6]"), (0.6, 1.0, "(0.6,1]")]


def share_band_table(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    tot = float(((p - y) ** 2).sum())
    rows = {}
    for lo, hi, lab in SHARE_BANDS:
        m = (y == 0) if hi == 0.0 else ((y > lo) & (y <= hi))
        if m.sum() > 0:
            rows[lab] = dict(
                n=int(m.sum()), frac=float(m.mean()),
                bias=float((p[m] - y[m]).mean()),
                rmse=rmse(p[m], y[m]),
                sse_share=float(((p[m] - y[m]) ** 2).sum() / tot)
                if tot > 0 else 0.0)
    return rows


def confidence_categories(p_main, p_a, p_b, y=None):
    mx = np.maximum(np.maximum(np.abs(p_main - p_a),
                               np.abs(p_main - p_b)),
                    np.abs(p_a - p_b))
    cats = {}
    for lab, m in [("clean", mx <= 0.05),
                   ("ambiguous", (mx > 0.05) & (mx <= 0.15)),
                   ("unresolved", mx > 0.15)]:
        e = dict(n=int(m.sum()), frac=float(m.mean()),
                 pred_mean=float(p_main[m].mean()) if m.sum() else None)
        if y is not None and m.sum() > 0:
            e["rmse"] = rmse(p_main[m], y[m])
        cats[lab] = e
    return cats


def fmt_scores(d):
    return " / ".join(f"{d[v]['score']:.4f}" for v in VAL_SETS)


def load_all_base():
    """Convenience: base features + labels for every frozen set."""
    feats, ys = {}, {}
    names = None
    for s in ["train", "trainA", "trainB"] + VAL_SETS:
        feats[s], names = load_base_features(s)
        ys[s] = load_set(s)["y"].astype(np.float64)
    return feats, ys, names
