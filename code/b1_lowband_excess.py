"""B1 — low-band excess-energy heuristic for the Zindi Underwater Vessel Noise Challenge.

A fixed per-window transform of Test.csv features only. Nothing is fitted —
not on test inputs (forbidden, and there are no labels), and not on the
unlabelled release either. The two design constants below are fixed a priori
from the acoustic reasoning stated here, not tuned.

Reasoning
---------
The organisers built each test window as background + a controlled vessel
passage added in the energy domain (starter notebook, cells 5 and 9). The
target, share_added_vessel, is the added passage's fraction of the window's
total energy. Two physical facts drive the estimator:

1. Vessel radiated noise is concentrated at low frequencies (propeller and
   machinery energy, mostly below ~1 kHz), while the window's total energy
   spans all 25 bands. So the added energy shows up overwhelmingly as excess
   energy in the low bands.
2. Within a 15-minute window the background is comparatively stationary,
   while a passage is a transient that occupies only part of the window.
   The band's own lower-quartile level across the window's minutes therefore
   tracks the background, and energy above that level, aggregated over the
   window, is attributable to the addition (plus ambient fluctuation, which
   is why this is a heuristic, not a model).

Estimator (per window)
----------------------
  E(m, b)      = 10^(L(m, b) / 10)              dB -> linear energy, NaN kept
  B_b          = 25th percentile of E(:, b) over the window's valid minutes
  Excess_b     = max( sum_m E(m, b) - n_b * B_b, 0 )   for low bands only
  share_hat    = sum_{low b} Excess_b / sum_{all m, b} E(m, b), clipped [0,1]

Low bands = the 16 bands with centres <= 1000 Hz (31.6 ... 1000 Hz).
Aggregating before subtracting (rather than summing per-minute positive
excesses) lets ordinary ambient fluctuation cancel instead of accumulating
as false added energy. Missing minutes are simply absent from every sum;
because share is a ratio of energies over the same observed cells, whole
missing minutes (the only missingness pattern in Test.csv: minute 1 in
41.6% of rows, minute 15 in 6.7%) cancel to first order.

Known failure mode (documented, accepted for a baseline): a passage that
fills nearly all 15 minutes at a steady level raises the lower quartile
itself, so its excess is underestimated.

score_added_vessel is set equal to share_hat: the evidence for a controlled
addition grows with the estimated added share. (Per the starter notebook the
score column is diagnostic only, ROC AUC, platform weight zero.)

Deterministic; no randomness, no seed needed.

Usage: python3 b1_lowband_excess.py
Writes: ../submissions/submission_b1_lowband.csv
Prints: summary statistics of the predictions for the run log.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
BACKGROUND_QUANTILE = 0.25     # fixed design constant, see docstring
LOW_BAND_MAX_HZ = 1000.0       # fixed design constant, see docstring


def estimate_share(window_db: np.ndarray, band_centres_hz: np.ndarray) -> float:
    """window_db: (15, 25) dB levels with NaN for missing. Returns share in [0,1]."""
    energy = 10.0 ** (window_db / 10.0)                    # NaN propagates
    total = np.nansum(energy)
    if not np.isfinite(total) or total <= 0:
        return 0.0
    low = band_centres_hz <= LOW_BAND_MAX_HZ
    background = np.nanpercentile(energy, BACKGROUND_QUANTILE * 100, axis=0)  # (25,)
    n_valid = np.sum(np.isfinite(energy), axis=0)                             # (25,)
    excess = np.maximum(np.nansum(energy, axis=0) - n_valid * background, 0.0)
    share = float(np.sum(excess[low]) / total)
    return min(max(share, 0.0), 1.0)


def main() -> None:
    test = pd.read_csv(BASE / "data" / "Test.csv", dtype={"ID": str})
    feature_columns = [c for c in test.columns if c != "ID"]
    assert len(feature_columns) == 15 * 25
    band_centres = np.array(
        [float(c.rsplit("_", 1)[1]) for c in feature_columns[:25]], dtype=float
    )
    acoustics = test[feature_columns].to_numpy(dtype=np.float64).reshape(-1, 15, 25)

    shares = np.array([estimate_share(w, band_centres) for w in acoustics])
    submission = pd.DataFrame(
        {
            "ID": test["ID"],
            "share_added_vessel": shares,
            "score_added_vessel": shares,
        }
    )
    out = BASE / "submissions" / "submission_b1_lowband.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print(f"Wrote {out}: {len(submission)} rows")
    print(
        "share_hat: min %.4f  p25 %.4f  median %.4f  mean %.4f  p75 %.4f  max %.4f"
        % (
            shares.min(),
            np.percentile(shares, 25),
            np.median(shares),
            shares.mean(),
            np.percentile(shares, 75),
            shares.max(),
        )
    )
    print(f"share_hat == 0 for {(shares == 0).sum()} windows; > 0.5 for {(shares > 0.5).sum()}")


if __name__ == "__main__":
    main()
