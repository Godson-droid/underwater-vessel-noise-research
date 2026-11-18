# Phase 0 audit — measured values (2026-11-16)

All values below were measured directly from the downloaded files in this
directory. Where they differ from the organisers' Data-tab documentation,
the measured value is the one to trust. Tooling: system Python 3 + pandas
2.1.4 / NumPy 1.26.4 for the CSVs; project venv (`.venv/`, pyarrow 26.0.1,
pandas 3.1.1) for the Parquet files.

## Files on disk

| File | Bytes | Notes |
|---|---|---|
| Test.csv | 2,829,614 | 1,500 rows × 376 cols |
| sample_submission.csv | 49,541 | on-disk name is lowercase `sample_submission.csv` (docs/notebook say `SampleSubmission.csv`) |
| maritime_starter_notebook.ipynb | 10,782 | 12 cells, nbformat 4.5 |
| South_African_Real_Acoustics.zip | 94,916,719 | unzips to `unzipped/` (14 files, incl. SHA256SUMS.txt) |

## Test.csv — measured

- Shape 1,500 × 376 = `ID` + 375 features, all features float64.
- Column order: minute-major, band ascending within each minute
  (`m01_TOL_31.6` … `m15_TOL_7943.3`). All 375 names match `m<MM>_TOL_<f>`.
- Bands (25, deci-decade): 31.6, 39.8, 50.1, 63.1, 79.4, 100, 125.9, 158.5,
  199.5, 251.2, 316.2, 398.1, 501.2, 631.0, 794.3, 1000, 1258.9, 1584.9,
  1995.3, 2511.9, 3162.3, 3981.1, 5011.9, 6309.6, 7943.3 Hz — exactly the
  starter notebook's `10^(1.5 + 0.1k)`, k = 0..24. Confirms the documented
  ≈31.6 Hz–7.94 kHz grid.
- IDs: 24-character lowercase hex strings, unique, set-identical to
  sample_submission.csv.
- Values: finite range 29.3–152.9 dB, quantised to 0.1 dB (1,149 distinct
  finite values), no infinities. Per-band medians decline smoothly from
  89.9 dB (31.6 Hz) to 69.2 dB (7943.3 Hz); per-band p5–p95 spans ≈40 dB.
- Missingness: 18,350 of 562,500 cells are NaN (3.2622%), and **every NaN
  belongs to a whole missing minute** (734 missing minutes × 25 bands =
  18,350 exactly). There is no partial-minute and no whole-band
  missingness. Per-minute missing rows:

  | Minute | Rows missing (of 1,500) | % |
  |---|---|---|
  | 1 | 624 | 41.60 |
  | 15 | 100 | 6.67 |
  | 14 | 4 | 0.27 |
  | 2 | 3 | 0.20 |
  | 8, 9, 13 | 1 each | 0.07 |
  | all others | 0 | 0 |

  775 rows are complete; the worst row misses 3 minutes. Per-band NaN rate
  is a uniform 3.26% for all 25 bands (a consequence of whole-minute
  missingness).

## sample_submission.csv — measured

- 1,500 rows, header exactly `ID,share_added_vessel,score_added_vessel`.
- Placeholder values are **share = 0.0 and score = 0.5 for every row**
  (not blanks). Note the starter notebook's constant baseline uses
  share = 0.1 instead — the sample's 0.0 is a placeholder, not a baseline.

## South_African_Real_Acoustics.zip — measured inventory

- 10 Parquet files under `unzipped/acoustics/`, one per site, plus
  `stations.csv`, `recorders.csv`, `DATA_DICTIONARY.md`, `README.md`,
  `SHA256SUMS.txt`.
- **Total rows: 1,715,779 — exactly the documented count.** Per-site
  measured row counts equal `stations.csv`'s `minutes` column exactly.
- All 10 files share one 74-column schema: 8 metadata columns (`site`,
  `recorder`, `station_token`, `sample_rate_hz`, `file_kind`, `src_file`,
  `file_date`, `t`), **62 TOL band columns spanning 0.8 Hz–288 kHz** (the
  common 25 bands are a subset; extra bands are recorder-specific export
  columns), and 3 boolean QC flags (`qc_duration_ge55`,
  `qc_25band_finite`, `qc_25band_spread_ge6`).
- `t` is datetime64[ms], export clock, timezone unverified.

| Site | Rows | Recorders | sample_rate_hz tokens | Coverage (export clock) | dur == 60 s | Rows dur < 55 s | Common-25 range (dB) |
|---|---|---|---|---|---|---|---|
| 1 Richards Bay | 1,001,174 | 3 | 24000 | 2017-05-09 → 2024-05-25 | 97.3% | 15,691 | −156.5 … 154.7 |
| 2 Durban | 44,640 | 1 | 48000 | 2025-10-28 → 2025-12-06 | 100% | 0 | 39.0 … 144.2 |
| 3 Cape Recife | 5,928 | 1 | 48000 | 2024-03-05 → 2024-03-23 | 100% | 0 | 50.8 … 125.7 |
| 4 Bird Island | 47,552 | 2 | 24000, 12000 | 2022-03-21 → 2024-06-29 | 99.9% | 32 | 38.0 … 174.2 |
| 5 St Croix Island | 250,717 | 3 | 12000, 24000 | 2022-04-05 → 2025-09-20 | 99.9% | 130 | 38.1 … 152.1 |
| 6 Riy Banks | 4,121 | 1 | 48000 | 2024-03-05 → 2024-04-05 | 100% | 0 | 53.9 … 125.3 |
| 7 St Francis | 169,488 | 1 | 48000 | 2015-04-21 → 2018-02-28 | ~100% | 1 | 59.4 … 156.9 |
| 9 Fish Hoek | 53,767 | 2 | 96000 | 2016-09-13 → 2021-09-28 | 84.6% | 1,014 | 32.8 … 144.3 |
| 10 Sea Point | 3,660 | 1 | 96000 | 2018-10-11 → 2018-10-13 | 99.9% | 0 | 36.1 … 105.5 |
| 13 Saldanha Bay | 134,732 | 2 | 48000, 96000 | 2022-03-22 → 2023-05-29 | 99.9% | 28 | 37.4 … 139.1 |

- Total rows with dur < 55 s: 16,896 — matches DATA_DICTIONARY.md exactly.
  `file_kind` is `MinRes` everywhere except 8,684 `MinRes_ERROR` rows, all
  at Richards Bay.
- **The common 25 bands are 100% finite in every row of every file**
  (`qc_25band_finite` true rate 1.000 throughout): the release contains no
  missing minutes at all. All test missingness was introduced downstream
  of this release.
- Sentinel warning: Richards Bay (58% of all records) reaches −156.5 dB,
  far below every other site's minimum (≥ 32.8 dB), and holds all the
  `MinRes_ERROR` rows. Bird Island's 174.2 dB maximum is likewise an
  outlier versus other sites (≤ 156.9 dB). Screen before pooling.
- Durban is the only site reaching into the competition period
  (Oct–Dec 2025). Per the archive README, test-background periods are
  withheld from this release, so no released window is a test background.

## B1 null diagnostic (see PLAN.md §4.2 and code/b1_null_check.py)

B1's estimator applied to 2,100 ambient-only 15-minute windows sampled from
the release (seed 20261116, ≤150 per recorder, strict cleanliness screen;
Cape Recife and Riy Banks yielded none): median 0.344, mean 0.398,
p25 0.196, p75 0.573, max 0.999, never 0 — versus B1 on Test.csv:
median 0.526, mean 0.528. The estimator's output is mostly null bias.
