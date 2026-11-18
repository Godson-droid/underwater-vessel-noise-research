"""Step 3, Stage 1: the screen. Order is binding: the null check runs
first, for both minimal estimators; an estimator that fails the null does
not proceed. Survivors are scored on the I1 main set against the
predeclared lines of proposal Section 6.4.

Line application, stated explicitly: the zero-band and (0.6, 1] bias
lines are applied to the UNSEEN-SITE subset, matching the frozen record's
usage (the reference numbers +0.030 / -0.051 / -0.055 / -0.070 are all
site-holdout statistics from Phases 6 to 8B). Overall figures are also
reported. Share bands follow the frozen banding: [0], (0, 0.05],
(0.05, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 1].
"""
import json
from pathlib import Path
import numpy as np

import estimators

HERE = Path(__file__).resolve().parent
BANDS = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]


def score(y, p):
    return float(1.0 - np.sqrt(np.mean((y - p) ** 2)))


def band_bias(y, p):
    out = {}
    for lo, hi in BANDS:
        if lo == hi:
            m = y == 0.0
            key = "0"
        else:
            m = (y > lo) & (y <= hi)
            key = f"({lo}, {hi}]"
        out[key] = {"n": int(m.sum()),
                    "bias": float((p[m] - y[m]).mean()) if m.sum() else None}
    return out


def main():
    results = {}
    # ---- Null check first, both estimators.
    survivors = []
    for which in ("s1", "s2"):
        preds = estimators.predict_set("null", which)
        np.save(HERE / f"null_pred_{which}.npy", preds)
        mean, med = float(preds.mean()), float(np.median(preds))
        passed = mean <= 0.03 and med <= 0.02
        results[which] = {"null": {"mean": mean, "median": med, "passed": bool(passed)}}
        print(f"[{which}] null mean {mean:.4f} median {med:.4f} -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
        if passed:
            survivors.append(which)
    # ---- Main screen for survivors.
    X = np.load(HERE / "corpus" / "main_X.npz")
    T = np.load(HERE / "corpus" / "main_truth.npz")
    y = T["y"]
    roles = np.array([t.split(":")[0] for t in X["bg_tag"]])
    unseen = roles == "eval_unseen"
    for which in survivors:
        p = estimators.predict_set("main", which)
        np.save(HERE / f"main_pred_{which}.npy", p)
        zb_site = float(p[unseen & (y == 0)].mean()) if (unseen & (y == 0)).any() else None
        hi_site_mask = unseen & (y > 0.6)
        hi_site = float((p[hi_site_mask] - y[hi_site_mask]).mean()) if hi_site_mask.any() else None
        zb_all = float(p[y == 0].mean()) if (y == 0).any() else None
        hi_all_mask = y > 0.6
        hi_all = float((p[hi_all_mask] - y[hi_all_mask]).mean()) if hi_all_mask.any() else None
        entry = {
            "overall_score": score(y, p),
            "unseen_site_score": score(y[unseen], p[unseen]),
            "seen_score": score(y[roles == "eval_seen"], p[roles == "eval_seen"]),
            "period_holdout_score": score(y[roles == "eval_period_holdout"],
                                          p[roles == "eval_period_holdout"]),
            "site_zero_band_bias": zb_site,
            "site_high_band_bias": hi_site,
            "overall_zero_band_bias": zb_all,
            "overall_high_band_bias": hi_all,
            "bands_unseen_site": band_bias(y[unseen], p[unseen]),
            "bands_overall": band_bias(y, p),
            "pred_mean": float(p.mean()),
        }
        # Verdict against the predeclared lines.
        kill = []
        if entry["overall_score"] < 0.93:
            kill.append(f"overall {entry['overall_score']:.4f} < 0.93")
        if zb_site is not None and zb_site > 0.040:
            kill.append(f"site zero-band bias {zb_site:+.4f} > +0.040")
        if hi_site is not None and hi_site < -0.070:
            kill.append(f"site (0.6,1] bias {hi_site:+.4f} < -0.070")
        if kill:
            entry["verdict"] = "killed"
            entry["kill_reasons"] = kill
        elif (entry["overall_score"] >= 0.96 and entry["unseen_site_score"] >= 0.93
              and zb_site is not None and zb_site <= 0.030
              and hi_site is not None and hi_site >= -0.055):
            entry["verdict"] = "promise"
        else:
            entry["verdict"] = "intermediate"
        results[which].update(entry)
        print(f"[{which}] overall {entry['overall_score']:.4f} unseen "
              f"{entry['unseen_site_score']:.4f} zb_site {zb_site:+.4f} "
              f"hi_site {hi_site:+.4f} -> {entry['verdict']}", flush=True)
    for which in ("s1", "s2"):
        if which not in survivors:
            results[which]["verdict"] = "killed"
            results[which]["kill_reasons"] = ["null check failed"]
    with open(HERE / "stage1_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v.get("verdict") for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
