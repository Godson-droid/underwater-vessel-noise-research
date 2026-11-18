"""Step 3, Stage 1: reference families and the S1 correction, fitted only on
family-side material.

Background family: from FIT-partition pool windows only (family sites,
non-holdout periods). Contents: pooled median spectrum shape, per-site
median shapes, and the top log-shape principal components across windows.

Passage family: from FAMILY-side clips only, converted fresh from the
waveforms with the Step 3 bandext implementation (no variation applied).
Contents: prototype minute-spectrum shapes (energy shares): the pooled
vessel median, per-vessel-class medians, and the ambient median.

S1 correction: fitted on the calibration corpus (family-side mixtures
whose truth is known to the FITTING code; the calibration corpus is a
training-side artefact, disjoint in material from the I1 main set).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import i1_common as i1

HERE = Path(__file__).resolve().parent
K_COMP = 4


def fit_background_family(max_windows=6000, seed=3):
    z = np.load(i1.ROOT / "data" / "bg_windows.npz")
    X, site = z["X"], np.asarray(z["site"])
    r = np.load(HERE / "bg_roles.npz")
    role = np.asarray(r["role"])
    idx = np.where(role == "fit")[0]
    rng = np.random.default_rng(seed)
    if len(idx) > max_windows:
        idx = rng.choice(idx, size=max_windows, replace=False)
    E = i1.db_to_lin(X[idx])                       # (n,15,25)
    win_spec = E.mean(axis=1)                      # (n,25)
    shapes = win_spec / win_spec.sum(axis=1, keepdims=True)
    pooled_median = np.median(shapes, axis=0)
    pooled_median /= pooled_median.sum()
    per_site = {}
    for s in sorted(set(site[idx].tolist())):
        sh = shapes[site[idx] == s]
        med = np.median(sh, axis=0)
        per_site[s] = (med / med.sum()).tolist()
    log_shapes = np.log(np.maximum(shapes, 1e-12))
    mu = np.log(np.maximum(pooled_median, 1e-12))
    centred = log_shapes - mu[None, :]
    _u, _s, vt = np.linalg.svd(centred - centred.mean(axis=0), full_matrices=False)
    comps = vt[:K_COMP]
    # per-minute log-spectrum variability (for S2 weighting): sd across
    # minutes within windows of the log spectrum, pooled median per band.
    minute_log = np.log(np.maximum(E, 1e-30))
    within_sd = np.median(minute_log.std(axis=1), axis=0)
    return {
        "mu_log_shape": mu.tolist(),
        "components": comps.tolist(),
        "component_centre": centred.mean(axis=0).tolist(),
        "per_site_median_shapes": per_site,
        "within_window_log_sd": within_sd.tolist(),
        "n_windows": int(len(idx)),
    }


def fit_passage_family():
    sides = pd.read_csv(HERE / "clip_sides.csv")
    fam = sides[sides["side"] == "family"]
    minute_shapes = {"vessel": [], "ambient": []}
    class_shapes = {}
    for _, row in fam.iterrows():
        wav = i1.load_wav(row["clip_key"])
        P = i1.convert_waveform_to_minutes(wav)
        if len(P) == 0:
            continue
        sh = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-30)
        kind = "ambient" if row["is_ambient_label"] else "vessel"
        minute_shapes[kind].append(sh)
        if kind == "vessel":
            class_shapes.setdefault(str(row["label"]), []).append(sh)
    protos = {}
    for kind in ("vessel", "ambient"):
        allsh = np.concatenate(minute_shapes[kind], axis=0)
        med = np.median(allsh, axis=0)
        protos[kind] = (med / med.sum()).tolist()
        protos[kind + "_n_minutes"] = int(len(allsh))
    class_protos = {}
    for lab, lst in sorted(class_shapes.items()):
        allsh = np.concatenate(lst, axis=0)
        if len(allsh) >= 60:
            med = np.median(allsh, axis=0)
            class_protos[lab] = (med / med.sum()).tolist()
    return {"prototypes": protos, "class_prototypes": class_protos}


def fit_s1(q_grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)):
    """Fit S1's quantile level and per-band correction on the calib corpus.

    For each candidate quantile q: U_b(w) = q-quantile over observed
    minutes of the band energies, times the number of observed minutes;
    kappa_b is the through-origin least-squares factor mapping U_b to the
    true per-band background total over observed cells. The q minimising
    calibration share RMSE is selected. All constants are then frozen.
    """
    X = np.load(HERE / "corpus" / "calib_X.npz")
    T = np.load(HERE / "corpus" / "calib_truth.npz")
    E = i1.db_to_lin(np.nan_to_num(X["X"], nan=-300.0))
    E = np.where(T["obs"][:, :, None], E, np.nan)
    E_bg_true = np.where(T["obs"][:, :, None], T["E_bg"], 0.0)
    T_b = E_bg_true.sum(axis=1)                    # (n,25)
    E_tot = np.nansum(np.where(T["obs"][:, :, None], E, 0.0), axis=(1, 2))
    n_obs = T["obs"].sum(axis=1).astype(float)
    y = T["y"]
    trials = {}
    best = None
    for q in q_grid:
        with np.errstate(all="ignore"):
            U = np.nanquantile(E, q, axis=1) * n_obs[:, None]   # (n,25)
        U = np.nan_to_num(U)
        kappa = (U * T_b).sum(axis=0) / np.maximum((U * U).sum(axis=0), 1e-30)
        est_bg = (U * kappa[None, :]).sum(axis=1)
        pred = np.clip(1.0 - est_bg / np.maximum(E_tot, 1e-30), 0.0, 1.0)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        trials[str(q)] = {"rmse": rmse, "kappa": kappa.tolist()}
        if best is None or rmse < best[1]:
            best = (q, rmse, kappa)
    q, rmse, kappa = best
    return {"q": float(q), "kappa": kappa.tolist(), "calib_rmse": float(rmse),
            "trials": {k: {"rmse": v["rmse"]} for k, v in trials.items()}}


def main():
    fam = {
        "background": fit_background_family(),
        "passage": fit_passage_family(),
        "s1": fit_s1(),
    }
    with open(HERE / "families.json", "w") as f:
        json.dump(fam, f)
    print("background family windows:", fam["background"]["n_windows"])
    print("passage prototypes:", list(fam["passage"]["prototypes"].keys()))
    print("class prototypes:", list(fam["passage"]["class_prototypes"].keys()))
    print("S1: q =", fam["s1"]["q"], "calib RMSE =", round(fam["s1"]["calib_rmse"], 4),
          "trials:", {k: round(v["rmse"], 4) for k, v in fam["s1"]["trials"].items()})


if __name__ == "__main__":
    main()
