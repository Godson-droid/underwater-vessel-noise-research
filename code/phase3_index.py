"""Phase 3 diagnostic, step 1: minute-level index of the released SA acoustics.

Builds a searchable array of every screened one-minute record (same screen as
build_bg_pool.py: MinRes only, dur >= 55 s, all 25 common bands finite and
within [25, 160] dB), sorted by (site, t), plus per-minute metadata. Also
measures whether released values are quantised like Test.csv (0.1 dB).

Output: data/phase3_minutes.npz with
  V    (N,25) float32 dB values
  site (N,)   site label per minute
  t    (N,)   int64 export-clock seconds per minute
  runok(N-14,) bool: minutes k..k+14 form a contiguous 15-min window
               (same site, exact 60 s steps)
"""
import glob
import numpy as np
import pandas as pd
from common import DATA, BAND_COLS

frames = []
for path in sorted(glob.glob(str(DATA / "unzipped" / "acoustics" / "*.parquet"))):
    cols = ["site", "t", "dur", "file_kind"] + BAND_COLS
    df = pd.read_parquet(path, columns=cols)
    df = df.sort_values("t").reset_index(drop=True)
    vals = df[BAND_COLS].to_numpy(dtype=np.float64)
    ok = (
        (df["file_kind"].to_numpy() == "MinRes")
        & (df["dur"].to_numpy() >= 55)
        & np.isfinite(vals).all(axis=1)
        & (vals >= 25.0).all(axis=1)
        & (vals <= 160.0).all(axis=1)
    )
    site = str(df["site"].iloc[0])
    tsec = df["t"].astype("datetime64[s]").astype("int64").to_numpy()
    sub = pd.DataFrame({"site": site, "t": tsec[ok]})
    for j, c in enumerate(BAND_COLS):
        sub[c] = vals[ok, j]
    frames.append(sub)
    print(site, "screened minutes:", int(ok.sum()), flush=True)

allm = pd.concat(frames, ignore_index=True)
V = allm[BAND_COLS].to_numpy(dtype=np.float32)
site = allm["site"].to_numpy()
t = allm["t"].to_numpy(dtype=np.int64)
N = len(V)
print("TOTAL screened minutes:", N)

# Quantisation check: distance of released values to the 0.1 dB grid.
samp = V[::997].astype(np.float64)
frac = np.abs(samp * 10 - np.round(samp * 10))
print(f"released values: median |10v - round(10v)| = {np.median(frac):.4f}, "
      f"p95 = {np.percentile(frac, 95):.4f} (0 => on the 0.1 dB grid)")

# Contiguity: window starting at k is valid if minutes k..k+14 share a site
# and step by exactly 60 s.
same_next = np.zeros(N, dtype=bool)
same_next[:-1] = (site[1:] == site[:-1]) & (np.diff(t) == 60)
runok = np.ones(N - 14, dtype=bool)
for d in range(14):
    runok &= same_next[d:d + N - 14]
print("contiguous 15-min window starts:", int(runok.sum()))

np.savez(DATA / "phase3_minutes.npz", V=V, site=site, t=t, runok=runok)
print("saved data/phase3_minutes.npz")
