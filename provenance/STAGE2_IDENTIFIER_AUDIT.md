# Stage 2 Identifier Audit — MMSI and other identifiers

**Date:** 18 November 2026 (Africa/Lagos)
**Scope:** Stage 2 Area C.
**Original file:** `step3/clip_sides.csv` — preserved unchanged.
**Stage 2 working file:** `stage2_work/clip_sides_sanitised.csv`.
No MMSI value is printed in this report; identifiers are referred to
by count and by masked prefix (first three digits + `***`) only.

---

## 1. The MMSI-bearing file: `step3/clip_sides.csv`

**Structure.** 525 data rows (plus header), one per metadata-bearing
clip in the Step 3 inventory (509 vessel + 16 ambient, per
DATA_SOURCES.md §1). Columns, in order:

`clip_key, label, date, MMSI, duration_sec, is_ambient_label, side`

**The identifier column.** `MMSI` is the only identifier column. Its
values are stored in float form (a 9-digit maritime mobile service
identity followed by `.0`). Findings:

- **179 distinct values** in the column, of which one is the
  placeholder `0.0`, carried by the 16 strict-background (ambient)
  rows, which have no vessel identity.
- **178 distinct real MMSI values** spread over the 509 vessel rows.
- **83 of the real values appear on more than one row** — expected:
  the MMSI is effectively the hull identifier, and the split was made
  per hull, so one vessel contributes several clips. (This is also
  why the column matters scientifically; see §4.)
- Masked-prefix distribution (distinct values per first-three-digits
  group, largest groups): `316***` — 101 distinct values; `367***` —
  14; `366***` — 7; `538***` — 7; `357***` — 5; `311***` — 4;
  `477***` — 4; the remaining 36 distinct values fall in 24 further
  prefix groups of 1–3 values each.

**Other columns.** `clip_key` is a file name within the VTUAD archives
(e.g. a vessel/background key); `label` is the vessel class;
`date` is the recording date (YYYYMMDD); `duration_sec`,
`is_ambient_label` and `side` (family / i1 / excluded_disagreement;
212 / 211 / 102 rows) complete the assignment record. None of these is
a personal or vessel identifier.

**Second known location (outside the scan scope below).**
`data/vtuad/metadata.csv`, part of the VTUAD download in `data/`, also
carries an `MMSI` column (confirmed by header inspection only). It is
not a release candidate — the whole of `data/` is excluded from any
release (see the data hygiene report) — and no values were extracted
from it.

## 2. Cross-scan for the same identifiers elsewhere

**Scope actually scanned (193 text files):** all `*.py`, `*.sh`,
`*.log`, `*.txt` at the top level of `code/`; all `*.py`, `*.json`,
`*.log`, `*.txt` at the top level of `step3/` (plus `step3/` CSVs
other than `clip_sides.csv` itself); all root-level `*.md`;
`submissions/*.csv`; `release_docs_stage1/*.md` and `*.txt`.
(Subdirectory contents — `step3/corpus/`, `step3/wav_cache/` — were
not text-scanned; both are excluded generated/derived data.)

**Method.** Each of the 178 distinct real MMSI digit strings was
searched as an exact substring in every file in scope (a scripted
scan; counts only were recorded, no matched content was reproduced).

**Result: no file in scope contains any of the 178 MMSI values.** The
identifiers exist in the working tree's text files only in
`step3/clip_sides.csv` itself (and, per §1, in the excluded
`data/vtuad/metadata.csv`).

**Scope limits, stated plainly.** Binary artefacts (`.npz` corpora,
LightGBM boosters, `.npy` predictions) were not content-scanned: their
formats do not store text identifiers in searchable form, and every
one of them is excluded from any release on data-hygiene grounds
regardless. The frozen `archive/` was not scanned — it is the
immutable evidence vault and never a release source; its copy of the
Step 3 record does not include `clip_sides.csv` as a release item.
This scan is therefore exhaustive for the text scope named above, and
is not claimed beyond it.

## 3. Sanitised derivative — method and verification

**Method.** `stage2_work/clip_sides_sanitised.csv` was produced by a
column drop only: the `MMSI` column was removed and every other column
and row was carried over unchanged, in the original order. **No
replacement identifiers were invented** — no pseudonyms, hashes or
renumbering were substituted for the MMSI values. Retained columns:

`clip_key, label, date, duration_sec, is_ambient_label, side`

**Verification performed (scripted, on the derivative):**

| Check | Result |
|---|---|
| Row count, derivative vs original | 525 vs 525 — identical |
| Retained-field values, row by row | 0 rows differ on any retained field |
| Row order (by `clip_key`) | Preserved exactly |
| Any of the 178 original MMSI digit strings present in the derivative | 0 found |
| Any 9-digit number pattern (`\b\d{9}(\.0)?\b`) in the derivative | 0 occurrences |
| Header | Exactly the six retained columns, original order |

The original `step3/clip_sides.csv` was not modified (it remains in
the research tree, and the frozen record is untouched; see the final
checksum verification).

## 4. Consequence of removing the MMSI field — documented

Removing `MMSI` removes the file's only join key to vessel identity:

- The sanitised file can no longer be joined to
  `data/vtuad/metadata.csv` (or to any AIS-derived source) to recover
  which clips belong to the same vessel.
- Consequently, the hull-disjointness of the Step 3 split — the
  property that family-side and I1-side clips share no hull — cannot
  be *re-verified from the sanitised file alone*. Within the file,
  clips of one vessel are no longer recognisable as such.

What is **not** lost: the assignment itself remains fully
interpretable. Every clip's side, class label, recording date and
duration are retained, so the split's composition (212 / 211 / 102)
and any per-clip question about the assignment can still be answered
from the sanitised file, and `step3/splits.json` independently records
the split counts and clip keys. The disjointness *verification* rests,
as it already does in the research record, on
`step3/stage0_verification.json` and the original file, which remain
preserved in the research tree for any authorised re-check. The
sanitised derivative is a release-facing copy; it does not replace
the original as evidence.

## 5. Other identifier-like content in scope

- **Email addresses:** none found anywhere in the scan scope of §2
  (email-pattern search over the same 193 files).
- **Zindi account username** (the owner's competition account name,
  `Godson22`): present in `PLAN.md` (1 occurrence), `SCOPE.md` (1),
  `PROPOSAL_step3_decomposition.md` (3), `WRITEUP_consolidated.md` (1)
  and `WRITEUP_phase0_8b.md` (1). This is the owner's own username in
  his own research record. The audit already recommends excluding
  `PLAN.md` from any release; the write-ups and proposal are release
  candidates, so whether the username stays in them is an owner
  authorship decision, flagged here and in the audit — not a leak and
  not sanitised in Stage 2.
- **Owner's first name in code docstrings:** exactly 10 files under
  `code/` mention it, in phrases recording the owner's direction
  (`cross_matrix_v2_v3.py`, `phase6_ws2_bands.py`, `phase7_gen_char.py`,
  `phase7_perturb.py`, `phase7_ws3_loso.py`, `phase8_gen_excl.py`,
  `phase8b_gen_graded.py`, `phase8b_verify_indep.py`, `simulate_v3.py`,
  `train_sim_v3.py`) — confirming the audit's "~10 files" finding.
  Authorship attribution; owner's decision, per the audit.
- **Credentials, tokens, wallet addresses:** none — see the Stage 2
  path hygiene report §3 for the credential scan and its one schematic
  (non-credential) finding in `PLAN.md`.

---

## Addendum — 18 November 2026, Stage 2 completion: the clip-key linkage issue

This addendum refines §§1, 3 and 4 above. The original text is
preserved unchanged; nothing in it is withdrawn except the implied
privacy strength of the §3 derivative, corrected in A.5 below.

### A.1 The linkage finding — a join from `clip_key` to vessel identity exists

The §3 derivative removed the `MMSI` column but retained `clip_key`.
That retention preserves a working join to vessel identity, verified
in this completion round by direct inspection:

- **Join path.** `clip_sides.csv.clip_key` → `data/vtuad/metadata.csv`
  `path` column, where the metadata value is the clip key with a
  leading `./` (clip keys have the form `vessel/<index>.wav` and
  `background/<index>.wav`; metadata paths are the same strings
  prefixed `./`). The metadata row reached by that join carries an
  `MMSI` column (its header also includes vessel dimensions and
  timestamps).
- **Coverage.** All 525 of 525 clip keys join to exactly one metadata
  row (525 distinct paths on each side). Of the joined rows, **509
  carry a real MMSI** in the metadata; the remaining 16 are the
  ambient/background rows whose metadata MMSI is the placeholder.
- **Consistency.** The metadata MMSI agrees with the `clip_sides.csv`
  MMSI on 525 of 525 compared rows — the two files encode the same
  identity assignment, so removing the column from one file while
  keeping the join key re-exposes exactly what was removed.
- **Consequence.** Anyone holding the VTUAD archive and its metadata
  (a public, CC BY 4.0 dataset) can join any retained clip key in the
  §3 derivative back to a vessel identity. The §3 derivative is
  therefore **MMSI-stripped only. It is not anonymised and is not
  resistant to re-identification**, and must not be described as such.
  Three distinct things must not be conflated: (i) direct MMSI
  removal (done in §3); (ii) removal of joinable source-clip
  references (not done in §3; done in A.3 below); (iii) the
  release-facing derivative versus the original evidence — the
  original `clip_sides.csv`, with keys and MMSI, remains the
  verification record in the private research tree.

### A.2 Every release-candidate record containing the clip keys

A programmatic scan of the Step 3 text records, root-level Step 3
documents, the Stage 1 release documents and `stage2_work/` (52 files)
for all 525 key strings found the keys in exactly three files:

| File | Keys present | How they appear |
|---|---|---|
| `step3/clip_sides.csv` (original) | 525 | The `clip_key` column itself. Private evidence; unchanged. |
| `step3/splits.json` | 423 distinct | Structured lists: `clip_split.family.clip_keys` (212) and `clip_split.i1.clip_keys` (211). The 102 excluded_disagreement clips appear as a count only, not by key. |
| `stage2_work/clip_sides_sanitised.csv` (§3 derivative) | 525 | The retained `clip_key` column — the linkage addressed here. |

No other text record contains any key: `step3/families.json`,
`step3/stage0_verification.json` and `step3/stage1_results.json`
contain none (site names, counts and scores only); the Step 3 code
files construct keys programmatically but embed no literal key values;
`RESULTS_step3_stage01.md` mentions only the script name
`extract_wav.py` and the `wav_cache/` directory in prose. Binary
derived artefacts (`step3/wav_cache/` file names, corpus `.npz` name
arrays) also embody the keys, but they are excluded generated data,
not release candidates (data hygiene report §1.5).

### A.3 Treatment implemented — opaque identifiers in release-facing derivatives

**Reproducibility assessment.** Public reproducibility does not
genuinely require the original clip keys in release-facing records.
The derivative's purpose is to show the assignment structure — sides,
class labels, recording dates, durations, ambient flags — and every
one of those fields is non-identifying and is preserved verbatim by
opaque substitution. The hull/date disjointness evidence does not
live in the release derivative: it rests on the preserved originals
(`step3/clip_sides.csv`, `step3/splits.py`,
`step3/stage0_verification.json`, which records shared hulls 0 and
shared dates 0), all retained in the private research tree. The one
capability given up is fetching the exact VTUAD WAV files by name
from the release records alone; VTUAD is in any case an external
prerequisite, and the private mapping (A.4) preserves the route for
any authorised re-check. Opaque substitution therefore does not
materially impair the scientific content of the release-facing
records, and it was implemented rather than stopping for a
retain-or-exclude decision.

**New derivatives (both in `stage2_work/`):**

- `clip_sides_release.csv` — columns
  `clip_id, label, date, duration_sec, is_ambient_label, side`;
  525 rows in the original row order. No `clip_key`, no `MMSI`.
- `splits_release.json` — `step3/splits.json` with the two
  `clip_keys` lists replaced by the corresponding opaque IDs under
  the **same** mapping; verified identical to the original in every
  other field.

Opaque IDs are `clip_0001` … `clip_0525`, assigned in the original
`clip_sides.csv` row order — stable, arbitrary, and carrying no
vessel, date or archive information.

**Verification (scripted, on the derivatives):**

| Check | Result |
|---|---|
| Row count, `clip_sides_release.csv` vs original | 525 vs 525 |
| Opaque IDs distinct | 525 of 525 |
| Non-identifying fields row-identical to the original | All 525 rows, all five fields |
| Side counts preserved | family 212 / i1 211 / excluded_disagreement 102 |
| `splits_release.json` key lists replaced | 423 entries (212 + 211), all mapped |
| Cross-file consistency: splits_release ID sets equal the release CSV's per-side ID sets | Yes, both sides |
| `splits_release.json` apart from the two key lists | Identical to `splits.json` |
| Any of the 525 original clip-key strings in either derivative | 0 |
| Any of the 178 real MMSI digit strings in either derivative | 0 |
| Any 9-digit number pattern in either derivative | 0 |

**What the opaque derivative does not do.** It does **not**
independently prove hull-disjointness: with keys and MMSI both absent,
same-vessel clips are unrecognisable within it, exactly as §4 already
stated for the MMSI-stripped file. The disjointness evidence remains
the preserved original verification record
(`step3/stage0_verification.json` plus the original
`clip_sides.csv`), and any re-verification is an authorised private
re-check using the mapping, not a property of the release derivative.

### A.4 The mapping — private only

The mapping from original clip keys to opaque IDs (525 rows) was
written **only** to `step3/opaque_id_mapping_PRIVATE.csv`, a new file
in the private research tree. It is not in `stage2_work/`, must not
enter any public-facing material, and its contents are not reproduced
in any Stage 2 report. Its absence from `stage2_work/` and its
presence in `step3/` were both verified in the completion round's
final consistency checks (see the Stage 2 completion report).

### A.5 Terminology correction and residual risk

- The §3 file, `clip_sides_sanitised.csv`, is to be described as
  **MMSI-stripped** only. It remains in `stage2_work/` as the record
  of the §3 step, but it retains joinable source-clip references and
  is superseded for release purposes by `clip_sides_release.csv`.
- Residual re-identification risk after opaque substitution: the
  release derivatives contain no direct identifier and no joinable
  source reference. Residual risk is limited to the non-identifying
  fields themselves (class label, recording date, duration, side) —
  quasi-identifiers that could in principle narrow a clip within the
  upstream metadata, though they cannot select a vessel without the
  private mapping or the original files, and dates/durations are
  shared by many clips. This residual risk is recorded, not claimed
  to be zero.
- The original research files — `step3/clip_sides.csv`,
  `step3/splits.json` and all historical verification records — are
  unchanged by this addendum's work.
