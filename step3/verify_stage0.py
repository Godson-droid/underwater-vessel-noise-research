"""Step 3, Stage 0 verification: disjointness evidence, harness hand-check,
oracle check, and the frozen Phase 5 baseline check on I1 material.

The frozen baseline is RUN read-only, as the proposal requires: its feature
transform (code/common.py) and its booster (models/lgbm_conv_bandext_
maskfix.txt) are loaded unmodified. Nothing in the archive is touched.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))  # frozen feature transform, read-only


def score(y, p):
    return float(1.0 - np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def disjointness():
    sides = pd.read_csv(HERE / "clip_sides.csv")
    fam = sides[sides["side"] == "family"]
    i1s = sides[sides["side"] == "i1"]
    out = {
        "shared_hulls": int(len(set(fam["MMSI"]) & set(i1s["MMSI"]))),
        "shared_dates": int(len(set(fam["date"]) & set(i1s["date"]))),
        "family_clips": int(len(fam)), "i1_clips": int(len(i1s)),
    }
    r = np.load(HERE / "bg_roles.npz")
    role, site = np.asarray(r["role"]), np.asarray(r["site"])
    for s in ("4 Bird Island", "9 Fish Hoek"):
        roles_s = set(role[site == s].tolist())
        out[f"roles_at_{s}"] = sorted(roles_s)
        assert "fit" not in roles_s and "eval_seen" not in roles_s
    assert out["shared_hulls"] == 0 and out["shared_dates"] == 0
    return out


def hand_check(n_sample=12, seed=5):
    """Recompute labels for a sample of main windows from the stored
    component energies with an independent, explicit-loop implementation,
    and check the exported summaries against the component sums."""
    X = np.load(HERE / "corpus" / "main_X.npz")
    T = np.load(HERE / "corpus" / "main_truth.npz")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T["y"]), size=n_sample, replace=False)
    worst_label, worst_x = 0.0, 0.0
    for i in idx:
        add_obs = 0.0
        tot_obs = 0.0
        for m in range(15):
            if not T["obs"][i][m]:
                continue
            for b in range(25):
                a = float(T["E_add"][i][m][b])
                g = float(T["E_bg"][i][m][b])
                add_obs += a
                tot_obs += a + g
                x_expect = round(10.0 * np.log10(max(a + g, 1e-30)) * 10) / 10
                worst_x = max(worst_x, abs(x_expect - float(X["X"][i][m][b])))
        lab = add_obs / tot_obs if tot_obs > 0 else 0.0
        worst_label = max(worst_label, abs(lab - float(T["y"][i])))
    return {"n_sample": int(n_sample), "worst_label_abs_diff": float(worst_label),
            "worst_summary_abs_diff_db": float(worst_x)}


def oracle_check():
    X = np.load(HERE / "corpus" / "main_X.npz")
    T = np.load(HERE / "corpus" / "main_truth.npz")
    E_mix = np.power(10.0, np.nan_to_num(X["X"], nan=-300.0) / 10.0)
    E_mix = np.where(T["obs"][:, :, None], E_mix, 0.0)
    bg_obs = np.where(T["obs"][:, :, None], T["E_bg"], 0.0)
    oracle = 1.0 - bg_obs.sum(axis=(1, 2)) / E_mix.sum(axis=(1, 2))
    return {"oracle_score_main": score(T["y"], oracle),
            "oracle_mean": float(oracle.mean()), "label_mean": float(T["y"].mean())}


def baseline_check():
    import lightgbm as lgb
    import common as frozen_common
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_conv_bandext_maskfix.txt"))
    out = {}
    for tag in ("main", "null"):
        X = np.load(HERE / "corpus" / f"{tag}_X.npz")
        F, _names = frozen_common.featurize_batch(X["X"])
        pred = np.clip(booster.predict(F), 0.0, 1.0)
        if tag == "main":
            T = np.load(HERE / "corpus" / "main_truth.npz")
            y = T["y"]
            roles = np.array([t.split(":")[0] for t in X["bg_tag"]])
            out["main_overall"] = score(y, pred)
            out["main_unseen_site"] = score(y[roles == "eval_unseen"],
                                            pred[roles == "eval_unseen"])
            out["main_seen"] = score(y[roles == "eval_seen"], pred[roles == "eval_seen"])
            out["main_period_holdout"] = score(y[roles == "eval_period_holdout"],
                                               pred[roles == "eval_period_holdout"])
            zero_unseen = (roles == "eval_unseen") & (y == 0)
            out["main_unseen_zero_band_bias"] = float(pred[zero_unseen].mean())
            out["main_pred_mean"] = float(pred.mean())
        else:
            out["null_pred_mean"] = float(pred.mean())
            out["null_pred_median"] = float(np.median(pred))
    return out


def main():
    res = {}
    out_path = HERE / "stage0_verification.json"
    for key, fn in (("disjointness", disjointness), ("hand_check", hand_check),
                    ("oracle", oracle_check), ("baseline_frozen_phase5", baseline_check)):
        res[key] = fn()
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{key}] done", flush=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
