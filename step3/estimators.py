"""Step 3, Stage 1: the minimal decomposition estimators S1 and S2.

Both estimators see only a window's summary (15x25 dB, NaN for missing
minutes) and the frozen reference families in families.json. Neither sees
any truth array. Both return a share in [0, 1] computed by the settled
arithmetic: share = 1 - E_bg_hat / E_tot over observed cells.

S1 (temporal envelope): per band, the background level is a low quantile
of the window's minute energies, corrected by the per-band factor kappa
fitted on the calibration corpus (families.fit_s1). The background is
taken as constant across the window's minutes within each band.

S2 (spectral family decomposition): each observed minute's spectrum is
fitted as a background component (a shape from the background family's
log-shape space, level free per minute, shape coefficients smoothed
across minutes) plus a passage component (one prototype shape from the
passage family, amplitude non-negative and smoothed across minutes).
Fitting alternates between the two components. E_bg_hat is the sum of
the fitted background levels over observed minutes.
"""
import json
from pathlib import Path
import numpy as np

import i1_common as i1

HERE = Path(__file__).resolve().parent
_fam = None


def families():
    global _fam
    if _fam is None:
        with open(HERE / "families.json") as f:
            _fam = json.load(f)
    return _fam


def _energies(win_db):
    cell_obs = np.isfinite(win_db)
    E = np.where(cell_obs, i1.db_to_lin(np.where(cell_obs, win_db, 0.0)), 0.0)
    obs = cell_obs.any(axis=1)  # minutes are wholly present or wholly missing
    return E, obs, cell_obs


def s1_predict(win_db):
    fam = families()["s1"]
    q, kappa = fam["q"], np.asarray(fam["kappa"])
    E, obs, cell_obs = _energies(win_db)
    E_tot = E.sum()
    if E_tot <= 0:
        return 0.0
    n_obs = int(obs.sum())
    Em = np.where(cell_obs, E, np.nan)
    with np.errstate(all="ignore"):
        U = np.nanquantile(Em, q, axis=0) * n_obs
    est_bg = float((np.nan_to_num(U) * kappa).sum())
    return float(np.clip(1.0 - est_bg / E_tot, 0.0, 1.0))


def _fit_bg_minutes(T, mu, C, centre, w_init=None, smooth_passes=2):
    """Fit per-minute background shapes/levels to target spectra T (m,25).

    Model: log shape = mu + centre + C w_m; level beta_m free. Weighted
    least squares in log space with weights sqrt(energy). Returns
    (beta (m,), W (m,K)).
    """
    m = T.shape[0]
    logT = np.log(np.maximum(T, 1e-30))
    Wgt = np.sqrt(np.maximum(T, 0.0))
    Wgt = Wgt / np.maximum(Wgt.mean(axis=1, keepdims=True), 1e-30)
    base = mu + centre
    W = np.zeros((m, C.shape[0])) if w_init is None else w_init.copy()
    beta = np.maximum(T.sum(axis=1), 1e-30)
    for _ in range(3):
        # given W: level per minute
        shape_log = base[None, :] + W @ C
        resid = logT - shape_log
        log_beta = (Wgt * resid).sum(axis=1) / np.maximum(Wgt.sum(axis=1), 1e-30)
        beta = np.exp(log_beta) * np.exp(shape_log).sum(axis=1) / np.exp(shape_log).sum(axis=1)
        # beta is defined so that beta * normalised shape matches scale:
        beta = np.exp(log_beta)
        # given beta: coefficients per minute (weighted ridge)
        r = logT - np.log(np.maximum(beta, 1e-30))[:, None] - base[None, :]
        for i in range(m):
            A = C.T * Wgt[i][:, None]
            lhs = A.T @ C.T + 1e-3 * np.eye(C.shape[0])
            rhs = A.T @ r[i]
            W[i] = np.linalg.solve(lhs, rhs)
    for _ in range(smooth_passes):
        Ws = W.copy()
        for i in range(m):
            lo, hi = max(0, i - 1), min(m, i + 2)
            Ws[i] = W[lo:hi].mean(axis=0)
        W = Ws
        shape_log = base[None, :] + W @ C
        resid = logT - shape_log
        log_beta = (Wgt * resid).sum(axis=1) / np.maximum(Wgt.sum(axis=1), 1e-30)
        beta = np.exp(log_beta)
    shapes = np.exp(base[None, :] + W @ C)
    shapes = shapes / shapes.sum(axis=1, keepdims=True)
    # Final level: project target onto shape in energy space.
    beta = np.maximum((T * shapes).sum(axis=1) / np.maximum((shapes * shapes).sum(axis=1), 1e-30), 0.0)
    return beta, W, shapes


def s2_predict(win_db):
    fam = families()
    bg = fam["background"]
    mu = np.asarray(bg["mu_log_shape"])
    C = np.asarray(bg["components"])
    centre = np.asarray(bg["component_centre"])
    protos = []
    pp = fam["passage"]["prototypes"]
    protos.append(np.asarray(pp["vessel"]))
    for v in fam["passage"]["class_prototypes"].values():
        protos.append(np.asarray(v))
    protos.append(np.asarray(pp["ambient"]))
    E, obs, _cell_obs = _energies(win_db)
    E_tot = E.sum()
    if E_tot <= 0:
        return 0.0
    T = E[obs]                                    # (m,25) observed minutes
    m = T.shape[0]
    A = np.zeros(m)                               # passage amplitudes
    p = protos[0]
    W = None
    for _round in range(3):
        beta, W, shapes = _fit_bg_minutes(np.maximum(T - A[:, None] * p[None, :], 1e-30),
                                          mu, C, centre, w_init=W)
        R = T - beta[:, None] * shapes
        # choose the prototype with the best total fit
        best = None
        for cand in protos:
            a = np.maximum((R * cand[None, :]).sum(axis=1) /
                           max(float(cand @ cand), 1e-30), 0.0)
            # smooth amplitudes over minutes (envelope regularisation)
            a_s = a.copy()
            for i in range(m):
                lo, hi = max(0, i - 1), min(m, i + 2)
                a_s[i] = np.median(a[lo:hi])
            sse = float(((R - a_s[:, None] * cand[None, :]) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, cand, a_s)
        _sse, p, A = best
    beta, W, shapes = _fit_bg_minutes(np.maximum(T - A[:, None] * p[None, :], 1e-30),
                                      mu, C, centre, w_init=W)
    est_bg = float(beta.sum())
    return float(np.clip(1.0 - est_bg / E_tot, 0.0, 1.0))


def predict_set(tag, which):
    X = np.load(HERE / "corpus" / f"{tag}_X.npz")
    fn = {"s1": s1_predict, "s2": s2_predict}[which]
    preds = np.empty(len(X["X"]))
    for i in range(len(X["X"])):
        preds[i] = fn(X["X"][i])
        if (i + 1) % 250 == 0:
            print(f"[{which} on {tag}] {i + 1}/{len(X['X'])}", flush=True)
    return preds
