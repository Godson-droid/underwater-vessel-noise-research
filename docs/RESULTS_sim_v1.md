# Simulator v1 + first model — results (2026-11-16)

Status: **candidate only. NOT submitted to Zindi.** Godson reviews before any submission.

## What was built

- **Background pool** (`code/build_bg_pool.py` → `data/bg_windows.npz`): 87,653 screened
  contiguous 15-min windows from the SA release (25 common bands). Screening: dropped all
  8,684 `MinRes_ERROR` rows, required dur ≥ 55 s, all 25 bands finite, and every value within
  [25, 160] dB (sentinel screen). Per-site screened rows / windows: Richards Bay 977,063 /
  52,089; St Croix 250,587 / 16,655; Saldanha Bay 134,704 / 8,863; St Francis 169,487 / 3,371;
  Durban 44,640 / 2,973; Bird Island 47,493 / 3,163; Fish Hoek 52,753 / 296; Sea Point 3,660 /
  243; Cape Recife 5,928 / 0 and Riy Banks 4,121 / 0 (rows exist but form no contiguous
  15-min run of screened rows).
- **Passage pool** (`code/build_passage_pool.py` → `data/passages.npz`): VTUAD v1.3.2 from
  Zenodo (DOI 10.5281/zenodo.23037895, CC BY 4.0). Converted with the starter notebook's own
  `minute_band_power` (cell 11, verbatim), stored as relative dB (absolute gain discarded, as
  the organisers did). **526 vessel clips + 16 ambient clips**; clip lengths min/median/max
  1/3/37 minutes; 32 kHz sample rate. Files used (exact bytes verified against the Zenodo
  record): `vtuad_strict_wav_vessel.zip` reassembled from parts 01–03 (3,712,337,699 B,
  **SHA-256 96564721a917d…ac6 matches the official checksum**), `vtuad_strict_wav_background.zip`
  (926,037,631 B, the 16 ambient clips), and the per-class archives `vtuad_wav_pilotvessel.zip`
  (69,120,288 B), `vtuad_wav_rescue.zip` (215,040,558 B), `vtuad_wav_tanker.zip`
  (253,440,688 B), `vtuad_wav_sailing.zip` (261,120,826 B) — 17 of the 526 vessel clips.
  Total downloaded ≈ 5.5 GB including record metadata/manifests.
- **Simulator** (`code/simulate.py`, seed 20261116): linear-energy mixing to an exact target
  share; share prior = 25% exact zeros, else 30% U(0,0.05) / 35% U(0.05,0.2) / 20% U(0.2,0.5)
  / 15% U(0.5,0.95) (mean ≈ 0.17, matching the low concentration the 0.1-constant benchmark
  implies); 30% of additions use an ambient clip segment as the added component (ambient-
  carrying references); variation = time reversal (p=0.3), time-warp ×0.7–1.4, smooth random
  per-band colouring (±4 dB), random placement; test missingness synthesised at the measured
  rates (minute 1: 41.6%, minute 15: 6.7%, minutes 2/14: 0.27%, 8/9/13: 0.07%, cap 3).
  Splits are disjoint in background windows AND passage clips: train 60,000 mixes (65,838 bg
  windows, 8 sites), val_random 6,000, val_site 6,000 (Bird Island + Fish Hoek held out),
  val_period 6,000 (per-site latest 15% of periods).
- **Model** (`code/train_sim_v1.py` → `models/lgbm_sim_v1.txt`, 9.2 MB): LightGBM on 549
  fixed per-window features (cell/band/minute energy shares, low-band fraction trajectory,
  per-band dB stats, B1 statistic as one feature, missingness indicators). Manual
  hyperparameters, fixed seed, no AutoML, nothing fitted on Test.csv.
  `score_added_vessel = sqrt(share prediction)` — a fixed monotone companion (zero-weight
  diagnostic column).

## Validation (synthetic data only; B0 = constant 0.1 on identical sets)

| set | model RMSE | model 1−RMSE | B0 RMSE | B0 1−RMSE | null mean pred (true share 0) |
|---|---|---|---|---|---|
| val_random | 0.0831 | **0.9169** | 0.2391 | 0.7609 | 0.018 |
| val_site (2 unseen sites) | 0.0951 | **0.9049** | 0.2471 | 0.7529 | 0.051 |
| val_period (late periods) | 0.0885 | **0.9115** | 0.2472 | 0.7528 | 0.022 |

Pressure map (RMSE by true share band, val_random / val_site / val_period):
0 → 0.039 / 0.070 / 0.050; 0–0.1 → 0.065 / 0.083 / 0.080; 0.1–0.3 → 0.086 / 0.091 / 0.087;
0.3–0.6 → 0.103 / 0.107 / 0.101; 0.6–1.0 → 0.163 / 0.166 / 0.159.
By missingness (val_random): 0 missing minutes 0.064, 1 missing 0.100, ≥2 missing 0.119.
Weakest site: Fish Hoek (val_site RMSE 0.146, n=500); Bird Island 0.089.
Top features (gain): low-band energy fraction, 1258.9 Hz band share, per-band temporal
std (esp. ~200 Hz and ~800 Hz), top-3-minute energy share — the expected acoustic story.
Full tables: `models/sim_v1_validation.json`.

## Test predictions — and the honest warning

`submissions/submission_sim_v1.csv` (sha256 2cb9937c22eb60d0…) **passes**
`code/validate_submission.py` (1,500 rows, IDs exact, values finite in [0,1]).
Predicted share on Test.csv: mean 0.330, median 0.286, only 14.3% below 0.05.

**That distribution does not match the simulator's world** (synthetic share mean 0.17;
~56% of synthetic examples have true share ≤ 0.1, and the benchmark arithmetic implies the
real test shares are also concentrated low). The model sees substantially more
"added-like" energy in real test windows than in synthetic mixes. Plausible causes, all
unresolved: (a) the organisers' passage components are longer or spectrally stronger than
my pool's clips (VTUAD strict clips are short — median 3 min); (b) their withheld test
backgrounds are quieter, raising relative shares; (c) their share prior is higher than the
one the benchmark led me to choose. **Consequence: the synthetic 0.90–0.92 scores will very
likely NOT transfer to the real leaderboard.** This file is a smoke-test candidate for one
of the 2/day submission slots if Godson approves — its public score would measure the
simulator–test gap directly — but it should not be treated as a 0.90 entry.

## Failures, shortcuts, caveats (stated plainly)

- Zenodo was reachable and used as planned; the documented synthetic-passage fallback was
  NOT needed. Downloads were interrupted once by a service restart; all files were resumed
  and size/checksum verified afterwards.
- LightGBM hit the 1,500-round cap (best iteration 1499; early stopping never fired), so
  the model may be slightly undertrained. One training run was lost to a LightGBM
  feature-name error (`[i]` suffixes) — fixed; featurisation is now cached on disk.
- The simulator's passage events are short (pool median 3 min, warped ×0.7–1.4) and placed
  once per window; if the organisers used longer or repeated passages, that is a first-order
  mismatch (see warning above). Ambient clips number only 16, split 13/3 train/val.
- Two sites (Cape Recife, Riy Banks) contribute no background windows; Fish Hoek only 296,
  which is why its holdout RMSE rests on 500 mixes drawn from few windows.
- Versions (venv, for the code review): python 3.12.3, numpy 2.5.5, pandas 3.1.1,
  pyarrow 26.0.1, scipy 1.18.1, lightgbm 4.7.0, scikit-learn 1.9.1, soundfile 0.14.0.
  Seed 20261116 everywhere. Nothing was submitted or posted anywhere.
