"""Build the background pool: screened, contiguous 15-minute windows from the
released South African acoustics, restricted to the 25 common bands.

Screening (PLAN.md section 4.3 step 1, Phase 0 section 3 findings):
  - drop file_kind == 'MinRes_ERROR' rows (8,684, all Richards Bay)
  - require dur >= 55 s
  - require all 25 common bands finite
  - require every common-band value within [25, 160] dB (sentinel screen:
    Richards Bay reaches -156.5 dB; every other site's minimum is >= 32.8 dB
    and maxima are <= 156.9 dB apart from one Bird Island outlier at 174.2)
Windows are non-overlapping blocks of 15 consecutive screened rows whose
timestamps step by exactly 60 s. Each window records site and start time for
split discipline (site holdout / period holdout in simulate.py).
"""
import glob
import numpy as np
import pandas as pd
from common import DATA, BAND_COLS

LO_DB, HI_DB = 25.0, 160.0

rows_out, counts = [], {}
for path in sorted(glob.glob(str(DATA / "unzipped" / "acoustics" / "*.parquet"))):
    cols = ["site", "t", "dur", "file_kind"] + BAND_COLS
    df = pd.read_parquet(path, columns=cols)
    n0 = len(df)
    df = df.sort_values("t").reset_index(drop=True)
    vals = df[BAND_COLS].to_numpy(dtype=np.float64)
    ok = (
        (df["file_kind"].to_numpy() == "MinRes")
        & (df["dur"].to_numpy() >= 55)
        & np.isfinite(vals).all(axis=1)
        & (vals >= LO_DB).all(axis=1)
        & (vals <= HI_DB).all(axis=1)
    )
    site = str(df["site"].iloc[0])
    tsec = df["t"].astype("datetime64[s]").astype("int64").to_numpy()
    # contiguous runs of screened rows with exact 60 s steps
    good = ok.copy()
    step_ok = np.zeros(len(df), dtype=bool)
    step_ok[1:] = (np.diff(tsec) == 60)
    wins = []
    i = 0
    idx = np.arange(len(df))
    while i + 15 <= len(df):
        block = idx[i:i + 15]
        if good[block].all() and step_ok[block[1:]].all():
            wins.append(block)
            i += 15
        else:
            i += 1
    counts[site] = dict(rows=n0, screened=int(ok.sum()), windows=len(wins))
    for w in wins:
        rows_out.append((site, int(tsec[w[0]]), vals[w].astype(np.float32)))
    print(site, counts[site], flush=True)

sites = np.array([r[0] for r in rows_out])
starts = np.array([r[1] for r in rows_out], dtype=np.int64)
X = np.stack([r[2] for r in rows_out])
np.savez_compressed(DATA / "bg_windows.npz", X=X, site=sites, start=starts)
print("TOTAL windows:", len(rows_out), "shape:", X.shape)
for s, c in counts.items():
    print("SCREEN", s, c)
