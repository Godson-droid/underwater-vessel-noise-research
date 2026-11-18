"""Constraint-1 experiment for Simulator v3: which label / missingness
construction is consistent with the real top score of 0.9943?

Phase 3 measured that under OUR v1/v2 construction an oracle subtracting
the TRUE background caps at 0.9528, because the label is the realised
share over the full 15x25 window computed BEFORE the missingness pattern
drops whole minutes: energy in dropped minutes is in the label but not
in any observable. The organisers' process must define the label so
that near-perfect recovery is possible. Candidates:

  A  (v1/v2 status quo) mix at target share over the full window, round
     to the 0.1 dB grid, drop minutes, label = full-window share
     (pre-missingness).
  B  (drop-first) draw the missingness mask first; scale the addition
     against the OBSERVED background cells only; label = realised share
     over observed cells (= target by construction).
  C  (mix-first, observed label) mix at target share over the full
     window exactly as A, round, drop minutes, but label = realised
     share over the OBSERVED cells only.

B and C share the same label *function* of (observed window, true
background) — share over observed cells — and differ only in the
generative order. A differs in the label function itself.

Method: paired draws. For each rep, one background, one assembled event
(v2 machinery), one target share, one missingness mask are drawn from a
per-rep seed; all three variants are then built from those same draws.
The estimator scored is the ORACLE: subtraction with the true background
over observed cells (phase3_gen.est_from_bg) — the ceiling of any
method under that construction. Nothing is fitted; Test.csv untouched.
"""
import numpy as np
from common import DATA, SEED, db_to_lin, lin_to_db
from simulate_v2 import (StratPool, build_event_lin, draw_duration,
                         warp_time, colour, gain_envelope, MISS_P)
from phase3_gen import est_from_bg

LEVELS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
REPS = 30


def draw_mask(r):
    drops = [m for m, p in MISS_P.items() if r.random() < p]
    if len(drops) > 3:
        drops = list(r.choice(drops, size=3, replace=False))
    mask = np.ones(15, dtype=bool)
    mask[drops] = False
    return mask


def build_event(clips, r):
    """v2-style assembled event, linear energy, unscaled, placed."""
    D = draw_duration(r)
    w = r.uniform(0.5, 2.0)
    src_len = int(np.clip(round(D / w), 1, 40))
    seg = build_event_lin(clips, src_len, r)
    if r.random() < 0.30:
        seg = seg[::-1].copy()
    new_len = int(np.clip(round(seg.shape[0] * w), 1, 15))
    seg = warp_time(seg, new_len, r)
    seg = colour(seg, r)
    seg = gain_envelope(seg, r)
    off = int(r.integers(0, 15 - new_len + 1))
    placed = np.zeros((15, 25))
    placed[off:off + new_len] = seg
    return placed


def to_window(E_lin, mask):
    win = np.round(lin_to_db(E_lin) * 10.0) / 10.0
    win[~mask, :] = np.nan
    return win


def run_variant(E_bg, placed, s, mask):
    """Returns dict variant -> (label, oracle_estimate)."""
    out = {}
    tot_full = E_bg.sum()
    p_full = placed.sum()
    obs = mask[:, None]
    bg_obs = np.where(obs, E_bg, 0.0).sum()
    p_obs = np.where(obs, placed, 0.0).sum()

    # A: scale on full window; label full share.
    if s > 0:
        aA = s * tot_full / ((1.0 - s) * p_full)
    else:
        aA = 0.0
    E_A = E_bg + aA * placed
    lab_A = float(aA * p_full / (tot_full + aA * p_full)) if s > 0 else 0.0
    win_A = to_window(E_A, mask)
    out["A"] = (lab_A, win_A)

    # B: scale on observed cells only; label = observed share.
    if s > 0:
        aB = s * bg_obs / ((1.0 - s) * p_obs)
    else:
        aB = 0.0
    E_B = E_bg + aB * placed
    lab_B = float(aB * p_obs / (bg_obs + aB * p_obs)) if s > 0 else 0.0
    win_B = to_window(E_B, mask)
    out["B"] = (lab_B, win_B)

    # C: same mixed window as A; label recomputed over observed cells.
    lab_C = float(aA * p_obs / (bg_obs + aA * p_obs)) if s > 0 else 0.0
    out["C"] = (lab_C, win_A)
    return out


def main():
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    BX, BS = bg["X"], bg["site"]
    pas = np.load(DATA / "passages.npz", allow_pickle=True)
    rel, kinds = list(pas["rel_db"]), pas["kinds"]
    vessel = [rel[i] for i in range(len(rel)) if kinds[i] == "vessel"]
    pool = StratPool(BX, BS)

    errs = {v: [] for v in "ABC"}
    errs_lvl = {v: {L: [] for L in LEVELS} for v in "ABC"}
    n = 0
    for rep in range(REPS):
        for L in LEVELS:
            r = np.random.default_rng(SEED + 1000 * rep + int(L * 100))
            j = pool.draw(r)
            bg_db = BX[j].astype(np.float64)
            E_bg = db_to_lin(bg_db)
            placed = build_event(vessel, r) if L > 0 else np.zeros((15, 25))
            mask = draw_mask(r)
            res = run_variant(E_bg, placed, L, mask)
            for v, (lab, win) in res.items():
                est = est_from_bg(win, bg_db)
                errs[v].append(est - lab)
                errs_lvl[v][L].append(est - lab)
            n += 1
    print(f"paired windows per variant: {n}")
    for v in "ABC":
        e = np.array(errs[v])
        rmse = float(np.sqrt(np.mean(e ** 2)))
        print(f"variant {v}: oracle RMSE {rmse:.4f}  oracle score {1 - rmse:.4f}")
        row = []
        for L in LEVELS:
            el = np.array(errs_lvl[v][L])
            row.append(f"{L}: {np.sqrt(np.mean(el ** 2)):.4f}")
        print("   by level:", "  ".join(row))


if __name__ == "__main__":
    main()
