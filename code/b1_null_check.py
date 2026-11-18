"""Null diagnostic for B1: run the B1 estimator on ambient-only windows.

Samples clean 15-minute windows (15 consecutive rows, 60 s apart, dur == 60,
all common-25 values finite and within (0, 165) dB to screen sentinel rows)
from the unlabelled South African release, per recorder, and applies
b1_lowband_excess.estimate_share to each. These windows contain NO controlled
addition, so the true share_added_vessel is 0 for the controlled component
(they may contain real vessels — part of the point: the estimator cannot
tell). The resulting distribution measures B1's upward null bias.

Fixed seed 20261116; up to 150 windows per recorder.

Result on 2026-11-16 (2,100 windows, 8 of 10 sites; Cape Recife and Riy Banks
yielded no clean windows under the strict screen):
    median 0.344, mean 0.398, p25 0.196, p75 0.573, max 0.999; never 0.
Compare B1 on Test.csv: median 0.526, mean 0.528. Conclusion recorded in
PLAN.md section 4.2: B1 raw is dominated by null bias and must not be
submitted uncalibrated; its statistic is a feature, not a predictor.

Usage: python3 b1_null_check.py   (uses the project venv's pyarrow)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_lowband_excess import estimate_share  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
COMMON = [
    "31.6", "39.8", "50.1", "63.1", "79.4", "100", "125.9", "158.5", "199.5",
    "251.2", "316.2", "398.1", "501.2", "631", "794.3", "1000", "1258.9",
    "1584.9", "1995.3", "2511.9", "3162.3", "3981.1", "5011.9", "6309.6",
    "7943.3",
]
TOL_COLS = ["TOL_" + b for b in COMMON]
CENTRES = np.array([float(b) for b in COMMON])
SEED = 20261116
MAX_PER_RECORDER = 150


def to_seconds(series) -> np.ndarray:
    raw = series.astype("int64").to_numpy()
    unit = str(series.dtype)
    if "ns" in unit:
        return raw // 10**9
    if "us" in unit:
        return raw // 10**6
    return raw // 10**3  # datetime64[ms]


def window_starts(tsec, dur, x) -> np.ndarray:
    n = len(tsec)
    if n < 15:
        return np.array([], dtype=int)
    step_ok = (np.diff(tsec) == 60).astype(np.int32)
    row_ok = (
        (dur == 60)
        & np.isfinite(x).all(axis=1)
        & (x.min(axis=1) > 0)
        & (x.max(axis=1) < 165)
    ).astype(np.int32)
    cs_step = np.concatenate([[0], np.cumsum(step_ok)])
    cs_row = np.concatenate([[0], np.cumsum(row_ok)])
    idx = np.arange(n - 14)
    good = (cs_step[idx + 14] - cs_step[idx] == 14) & (cs_row[idx + 15] - cs_row[idx] == 15)
    return idx[good]


def main() -> None:
    rng = np.random.default_rng(SEED)
    all_null = []
    for path in sorted(glob.glob(str(BASE / "data" / "unzipped" / "acoustics" / "*.parquet"))):
        df = pq.read_table(path, columns=["recorder", "t", "dur"] + TOL_COLS).to_pandas()
        df = df.sort_values(["recorder", "t"])
        ests = []
        for _, group in df.groupby("recorder"):
            group = group.reset_index(drop=True)
            x = group[TOL_COLS].to_numpy(dtype=np.float64)
            starts = window_starts(to_seconds(group["t"]), group["dur"].to_numpy(), x)
            if len(starts) == 0:
                continue
            picks = rng.choice(starts, size=min(MAX_PER_RECORDER, len(starts)), replace=False)
            ests.extend(estimate_share(x[i : i + 15], CENTRES) for i in picks)
        if ests:
            ests = np.array(ests)
            all_null.append(ests)
            print(
                f"{Path(path).name:28s} null n={len(ests):4d}  median {np.median(ests):.3f}  "
                f"p25 {np.percentile(ests, 25):.3f}  p75 {np.percentile(ests, 75):.3f}  mean {ests.mean():.3f}"
            )
        else:
            print(f"{Path(path).name:28s} null n=0 (no clean 15x60s windows)")
    all_null = np.concatenate(all_null)
    print(
        f"\nALL NULL windows: n={len(all_null)}  min {all_null.min():.3f}  "
        f"p25 {np.percentile(all_null, 25):.3f}  median {np.median(all_null):.3f}  "
        f"mean {all_null.mean():.3f}  p75 {np.percentile(all_null, 75):.3f}  max {all_null.max():.3f}"
    )


if __name__ == "__main__":
    main()
