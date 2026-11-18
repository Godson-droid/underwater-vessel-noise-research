"""Phase 7 perturbation test — PREDECLARED set, inference only.

Restrictions (Godson's ruling, binding): predeclared physically
defensible perturbations only; no fitting, calibration,
threshold selection, training or pseudo-labelling from test
inputs; outcomes never modify any model; stability and
disagreement recorded as diagnostics only; the stored test
inputs are never modified (perturbations apply to in-memory
copies) and the submission pipeline is untouched.

Set (predeclared in RESULTS_phase7_calibration.md before this
run): G+ / G- global gain +-2 dB; T+ / T- spectral tilt -+0.5 dB
at band 0 to +-0.5 dB at band 24 (linear in band index); B+ / B-
band 0 only +-1 dB; M8 minute 8 set missing; RV minute order
reversed.

Models: Phase 5 baseline main (549 features) and Phase 6 final
main (549 + families B + C) — the two families whose real-window
readings diverge (0.2061 vs 0.1718).
Record: models/phase7_perturb.json.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import DATA
from phase6_lib import MODEL_DIR, baseline_model, predict, rmse
from phase6_features import featurize_extra_mp
from train_conv import featurize_mp

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
X0 = test[fcols].to_numpy(dtype=np.float64).reshape(-1, 15, 25)
print(f"test windows: {X0.shape}", flush=True)

tilt = -0.5 + np.arange(25) * (1.0 / 24.0)


def pert(pid, X):
    X = X.copy()
    if pid == "G+":
        X = X + 2.0
    elif pid == "G-":
        X = X - 2.0
    elif pid == "T+":
        X = X + tilt[None, None, :]
    elif pid == "T-":
        X = X - tilt[None, None, :]
    elif pid == "B+":
        X[:, :, 0] = X[:, :, 0] + 1.0
    elif pid == "B-":
        X[:, :, 0] = X[:, :, 0] - 1.0
    elif pid == "M8":
        X[:, 7, :] = np.nan
    elif pid == "RV":
        X = X[:, ::-1, :].copy()
    else:
        raise ValueError(pid)
    return X


PIDS = ["G+", "G-", "T+", "T-", "B+", "B-", "M8", "RV"]

base_main = baseline_model("")
iters = json.loads((MODEL_DIR / "phase6_final_iters.json").read_text())
p6_main = (lgb.Booster(model_file=str(MODEL_DIR / "phase6_final.txt")),
           dict(num_iteration=iters["main"]))


def predict_both(X):
    Fb = featurize_mp(X.astype(np.float32))
    Fx = featurize_extra_mp(X, ("B", "C"))
    F6 = np.concatenate([Fb, Fx], axis=1)
    return predict(base_main, Fb), predict(p6_main, F6)


p0_base, p0_p6 = predict_both(X0)
print(f"unperturbed: baseline mean {p0_base.mean():.4f} "
      f"(expect 0.2061), phase6 mean {p0_p6.mean():.4f} "
      f"(expect 0.1718), disagreement RMSE "
      f"{rmse(p0_base, p0_p6):.4f}", flush=True)

out = dict(test="phase7_perturbation_predeclared",
           unperturbed=dict(baseline_mean=float(p0_base.mean()),
                            phase6_mean=float(p0_p6.mean()),
                            disagreement_rmse=rmse(p0_base, p0_p6)),
           perturbations={})
for pid in PIDS:
    pb, p6 = predict_both(pert(pid, X0))
    row = {}
    for lab, p, p0 in [("baseline", pb, p0_base), ("phase6", p6, p0_p6)]:
        d = p - p0
        row[lab] = dict(mean_pred=float(p.mean()),
                        mean_signed_delta=float(d.mean()),
                        mean_abs_delta=float(np.abs(d).mean()),
                        p95_abs_delta=float(np.percentile(np.abs(d), 95)),
                        frac_abs_gt_0p05=float((np.abs(d) > 0.05).mean()))
    row["disagreement_rmse"] = rmse(pb, p6)
    out["perturbations"][pid] = row
    print(f"{pid}: base |d| {row['baseline']['mean_abs_delta']:.4f} "
          f"p95 {row['baseline']['p95_abs_delta']:.4f} | "
          f"p6 |d| {row['phase6']['mean_abs_delta']:.4f} "
          f"p95 {row['phase6']['p95_abs_delta']:.4f} | "
          f"disagree {row['disagreement_rmse']:.4f}", flush=True)
(MODEL_DIR / "phase7_perturb.json").write_text(json.dumps(out, indent=1))
print("wrote models/phase7_perturb.json", flush=True)
