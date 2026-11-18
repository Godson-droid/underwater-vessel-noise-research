# Underwater Vessel Noise — Research Programme Record

**Status of this document:** proposed public-facing README, prepared as
Stage 1 release documentation (18 November 2026). The research programme
is closed. **No public release has been authorised yet**, and no licence
has been chosen for the code (see `LICENSE_DECISION.md`). This README
describes the programme and its record as they stand.

---

## 1. Project overview and competition objective

This repository records a complete, closed research programme on the
**Underwater Vessel Noise Challenge by the French Embassy in South
Africa**, hosted on Zindi. The programme was a solo entry. (The
consolidated research report was finalised, and the release-readiness
work that produced this documentation was carried out, between 16 and
18 November 2026.)

The competition task: for each of 1,500 fifteen-minute underwater
acoustic test windows, predict what fraction of the window's total
acoustic energy came from a vessel passage the organisers added to a real
recorded background.

The difficulty is that there is no labelled training set. The only
training material supplied is 1,715,779 unlabelled one-minute band-energy
records from ten South African sites. All labelled data therefore has to
be manufactured by simulating the organisers' construction: real recorded
backgrounds plus controlled vessel-passage additions drawn from the
VTUAD v1.3.2 passage corpus that the competition starter notebook names.

The programme built that simulator, diagnosed why its models failed to
transfer to the real windows, corrected the largest causes it could
measure, and stopped when its pre-set quality gate was not met. It then
tested one physically different alternative approach under a separate
authorisation; that approach also failed its predeclared test and was
closed. Exactly one competition submission was ever made (an early
diagnostic, which scored below the constant benchmark). The value of this
record is the diagnosis and the disciplined negative results, not a
winning model.

## 2. Target definition and scoring

Each test window is a 15 × 25 grid: fifteen one-minute observations
across 25 frequency bands, given as band-energy summaries in decibels.
There is no raw audio for the test windows.

- **Target:** `share_added_vessel` — the fraction of the window's total
  acoustic energy attributable to the controlled added passage, in
  [0, 1].
- **Score:** `1 − RMSE` computed on `share_added_vessel` only. Higher is
  better; a perfect predictor scores 1.
- **Companion column:** `score_added_vessel` is a zero-weight diagnostic.
  Throughout the programme it was predicted as a fixed monotone companion
  of the share, √share, exactly as the competition format requires, and
  it plays no part in scoring.
- **Leaderboard split:** the public leaderboard reflects about 30% of the
  test windows; the private leaderboard covers the remaining ~70%. A
  public score does not establish private performance.
- **Benchmark:** the constant predictor 0.1 scores 0.753867825 on the
  public board. The public board's top score was 0.994250562. Both
  figures are leaderboard records, not artefacts reproducible from this
  repository.

The programme's own label definition, settled by a target-arithmetic
audit in Phase 5, is the realised share over the window's **observed
cells** (after missingness is applied). Every other audited definition
(full-window share, cell-mean, band-restricted and amplitude-domain
variants) has an oracle ceiling too low to be consistent with the
observed top score and was rejected on measurement.

## 3. The research programme in brief

**Phases 0–4 — building the simulator, finding the transfer failure.**
Phase 0 audited the released data and set two baselines: a constant
predictor (B0) and a fixed low-band excess statistic (B1). B1 failed its
null check — on 2,100 ambient-only windows whose true share is zero, it
read a median of 0.344 — establishing the programme's first rule: an
estimator that has not passed a null check on pure backgrounds is not yet
an estimator. Simulator generations v1 and v2 (Phases 1–2) scored
0.90–0.92 on their own synthetic holdouts but read the real test windows
at a mean of about 0.33 against a plausible true mean of 0.13–0.20. The
v2 model was submitted once, as a deliberate smoke test, and scored
0.687868143 on the public board, below the constant benchmark. The
result demonstrated that strong synthetic validation scores alone were
insufficient evidence of transfer to real competition windows. Phase
3 tested whether the test backgrounds could be matched verbatim in the
released acoustics and the added component recovered by subtraction: the
matching machinery worked on synthetic windows, but on real windows an
exact-minute fingerprint fired zero times in 60 sampled windows, and the
hypothesis was rejected and the direction closed permanently. Phase 4
(simulator v3) fixed the label definition, raising the construction's
oracle ceiling from 0.9510 to 0.9983; synthetic scores rose, the
real-window over-reading did not move, and the diagnosis pointed at the
passage conversion.

**Phase 5 — the conversion was the principal measured source of transfer failure.** Ten conversion variants were
screened under an identical protocol, changing one axis at a time. Nine
left the real-window mean essentially unchanged. One — **bandext**,
which integrates band 0 from 0 Hz instead of its deci-decade lower edge
(≈56 Hz) and band 24 up to Nyquist (16 kHz) instead of ≈8.9 kHz — cut
the real-window mean from 0.337 to 0.2042 and produced the best synthetic
holdouts. The released background windows are the organisers' own
conversion and already contain full-range edge bands; the clipped
conversion had made synthetic additions carry far too little edge-band
energy, so models calibrated on it systematically over-read real windows.
Combined with a mask correction (maskfix), the resulting construction,
**bandext_maskfix**, became the programme baseline: holdouts 0.9482
random / 0.9030 site / 0.9469 period / 0.9422 long, construction ceiling
0.9981, real-window mean 0.2061.

**Phase 6 — the model was not the bottleneck.** On the frozen baseline,
error split into ordinary shrinkage on seen sites and a distinct
site-holdout failure: at true share zero, bias +0.065, with the zero
band carrying 21.4% of the site holdout's squared error. Band 0 is both
the dominant vessel channel and site-confounded; the failure is
calibration, not identity leakage. Deleting band-0 features, leave-site-
out stopping and site-balanced weights were all rejected or null;
XGBoost, histogram gradient boosting, an MLP and a Huber-loss variant
all tied LightGBM. Model class, loss, training volume and variation
breadth were each shown not to be the constraint.

**Phase 7 — background character is a major source of zero-share
over-reading, and character training reduces it.** Each site's character was summarised as its
median per-band energy-share vector, and drawn backgrounds were
transplanted onto other sites' characters with total energy preserved
(Bird Island's natural band-0 share is 1.70× the all-site median — its
backgrounds are band-0-heavy in exactly the way the baseline reads as
vessel energy). Character training improved the site holdout to 0.9232
(549 features) and 0.9250 (with background-relative family D features),
cut the zero-share bias to +0.028 / +0.025 and halved its error — at the
cost of under-reading genuinely large additions on unseen sites. Two
other levers failed for identified reasons: background-relative features
did not move the zero bias (the whole window carries the site character),
and leave-one-site-out ensembling made it worse. A blindness caveat was
recorded: the transplant family included the holdout sites' aggregate
median shapes.

**Phase 8A — how much of the correction transfers.** The Phase 7
generation was rerun with Bird Island's shape removed from the
transplant targets. The zero-share correction survived at 89–91% of its
bias correction: the correction **partially survives** — transferable
learning dominant, with a real, bounded Bird-Island-specific term.

**Phase 8B — the trade-off, mapped.** Character strength was varied on a
predeclared graded scale (0, 0.25, 0.5, 0.75, 1.0), every strength level
crossed with the same drawn-share distribution — independent drawn-share
assignment with small measured realised-label variation (labels exactly
identical for 90.62% of windows; mean absolute drift 9.1×10⁻⁴). The
screen showed a smooth dose–response: most of the zero-share correction
is present by quarter strength, while most of the high-share penalty
arrives between quarter and three-quarter strength. Only quarter
strength advanced to the full protocol (site 0.9163 / 0.9218): the
zero-share correction retained, the high-share recovery partial and
bounded.

**Step 3 — a physically different alternative, tested and rejected.**
Under a separate authorisation, the programme tested physical energy
decomposition: estimate the background share of a window's observed-cell
energy and derive the share as one minus that ratio. Stage 0 built a
separately implemented synthetic validation (I1) — passage clips disjoint
by hull and recording date from the fitting material, an unseen-site
evaluation drawn from two sites excluded entirely from background-family
fitting — and passed its verification checks (oracle 0.9989; the frozen
Phase 5 baseline scored 0.8854 on it as a reference point). Stage 1 ran
the predeclared null test first: on 550 pure backgrounds from held-out
sites, an estimator passed only if its mean estimated share was at most
0.03 and its median at most 0.02. Both minimal estimators failed by wide
margins — S1 (temporal envelope) mean 0.5747, median 0.6285; S2
(spectral family decomposition) mean 0.0999, median 0.0920 — and the work
stopped there, exactly where the protocol required. Because the null
test failed, **no overall or unseen-site scores for S1 or S2 exist**;
none were produced, and none should be inferred. Step 3 is formally
closed; Stage 2 is not authorised.

## 4. The gate that was not met, and the two standards

Two different numbers governed this programme, and they must not be
confused:

- **The 0.96 development gate.** A configuration could only be proposed
  as a submission candidate if it reached 0.96 on the fixed validation
  *and* both independent cross-checks (models trained on disjoint
  splits), with all four holdouts — random, site, period and long —
  reported separately. **This gate was not met by any configuration in
  any phase.** The best holdout score recorded in Phases 0–8B is 0.9488
  and the best site-holdout score is 0.9250.
- **The ≈0.998 submission standard.** The eventual standard for actually
  spending a submission slot was approximately 0.998 under genuinely
  independent validation — the level the construction's oracle ceiling
  (≈0.998) shows is arithmetically recoverable under a correct mirror of
  the organisers' process. **That ceiling is a property of the
  construction, not demonstrated model performance.** No model in this
  programme approached the standard, and no candidate was produced after
  Submission 1.

## 5. Synthetic validation is not real-world validation

Every validation in this programme is synthetic. There are no labelled
real windows outside the organisers' process, so no result here is, or
can be presented as, real-world validation.

The Step 3 construction (I1) strengthens the synthetic evidence — it is
independent in implementation and in held-out material — but it carries
a binding qualification, recorded when the work was authorised: **it
still uses the same underlying public data sources and synthetic
energy-additive assumptions as the frozen programme. It must never be
represented as independent real-world validation.**

Likewise, real-window prediction means reported in the phase reports
(for example the baseline's 0.2061) are unvalidated diagnostics used for
consistency checking. They are not estimates of the true test
distribution, and no phase treated them as validation scores.

## 6. Reproduction roadmap

Reproduction is a partial reconstruction sequence: the release ships
code, records and reports, and the commands below regenerate the
research inputs, selected diagnostics and intermediate artefacts that
can be reproduced without creating competition submissions. Some
historical training commands are deliberately excluded because they
write files using the exact competition submission schema, even when
their output directory is `models/`. Consequently, this roadmap does
not provide a complete end-to-end regeneration of every reported
model score or trained booster. Those results remain documented in
the frozen records, with their configurations, seeds and provenance.
Commands below are drawn from the scripts' own usage lines,
the two shell drivers in `code/`, and the code lists in
`ARCHIVE_MANIFEST.md` §5. Steps whose exact invocation could not be
verified from a usage line or driver are marked **requires
verification**.

**This roadmap is a proposed reconstruction sequence, not a verified
one-command procedure.** It has not been executed end to end as written;
treat every step as requiring confirmation against the phase reports
before reliance, and the steps marked **requires verification** as
unconfirmed. Run any reproduction in a **separate environment** with
fresh output locations: running the historical scripts regenerates
historical experimental artefacts (corpora, boosters, diagnostics),
and those outputs belong in the scratch environment, never in the
record. Reproduction must **never modify the frozen `archive/`**, and
it must **never create competition submissions**: every script known
to write a submission file, or a submission-shaped output, is
deliberately excluded from the sequence below (see the omission notes
in Steps 2 and 4).

**Environment.** Python 3.12.3; install the pinned dependencies:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Working directory convention.** The original drivers run from inside
`code/` with the project venv at `../.venv/bin/python`, and the scripts
resolve all data paths absolutely from their own location
(`code/common.py`), so they can also be invoked from the repository
root as shown below. The two shell drivers contain machine-specific
paths from the original machine and need adaptation before use
(**requires verification** on any other machine).

**Step 0 — obtain the external data** (see `DATA_SOURCES.md`). Place the
competition files (the South African acoustic package and the starter
notebook) and the VTUAD v1.3.2
archives under `data/` in the layout the scripts expect. Full
reproduction requires competition access to the South African acoustic
package through Zindi; see §8.

**Step 1 — build the pools (Phase 0–1).**

```
python code/build_bg_pool.py        # screened background pool: 87,653 windows
python code/build_passage_pool.py   # VTUAD conversion; reassembles and
                                    # checksum-verifies the strict vessel
                                    # archive on first run (see DATA_SOURCES.md)
python code/b1_null_check.py        # B1 null check (seed 20261116)
```

(Invocation order within this step follows `ARCHIVE_MANIFEST.md` §5;
**requires verification** for argument details — these scripts carry no
usage lines.)

**Step 2 — simulator generations v1–v3 (Phases 1, 2 and 4).**

```
python code/simulate.py             # v1 generation
python code/simulate_v2.py          # v2 generation
python code/label_variants.py       # Phase 4 label-definition audit
python code/simulate_v3.py          # v3 generation (observed-cells label)
python code/cross_matrix_v2_v3.py   # cross-generation comparison
```

(**Requires verification**: exact sequencing and any arguments; order
follows the manifest's code lists and the phase reports.)

**Omitted from this sequence deliberately:** `code/b0_constant.py` and
`code/b1_lowband_excess.py` (Step 1) and `code/train_sim_v1.py` and
`code/train_sim_v2.py` (Step 2) wrote competition submission files as
part of their historical runs, so they are not part of the
reproduction sequence.
This roadmap covers research-result reproduction only; it is not a
route to competition inference or submission generation, and no
submission file is needed to reproduce the research findings. The
scores those scripts produced are preserved in the frozen records.

**Additional commands deliberately excluded:** several later-phase
training scripts — `code/train_sim_v3.py`, `code/train_conv.py`
(screen and full variants), `code/phase6_final.py`,
`code/phase7_full.py` (both configurations), `code/phase8_train_eval.py`
and `code/phase8b_train_eval.py` — write test-prediction CSVs using
the exact competition submission schema (`ID`, `share_added_vessel`,
`score_added_vessel`), including the prescribed square-root companion
column, one row for every test ID and no additional diagnostic
columns. Although their historical output paths are under `models/`,
rather than `submissions/`, these files are functionally
submission-shaped outputs. The scripts are therefore not included as
executable commands in this reproduction roadmap. Their existing
machine-readable model records and reported results remain part of
the research record. This exclusion does not alter or delete the
historical source code or experimental evidence.

**Step 3 — Phase 3 background-matching diagnostic.**

```
python code/phase3_index.py
python code/phase3_gen.py
python code/phase3_match.py
python code/phase3_synth.py
python code/phase3_analyze.py
python code/phase3_prior.py
python code/phase3_test.py
```

(**Requires verification**: order as listed in `ARCHIVE_MANIFEST.md` §5;
the phase report, `RESULTS_phase3_diag.md`, is the controlling account.)

**Step 4 — Phase 5 conversion investigation and the baseline.**

The screen and full sequences were historically driven by two shell
scripts, `code/run_screens.sh` and `code/run_full_pipeline.sh`, which
remain the historical command record for this phase. They are not
executable steps in this roadmap: both exist to run `train_conv.py`
sequences, and `train_conv.py` is excluded (see Step 2's omission
notes) because it writes submission-shaped outputs.

The underlying commands that remain in the sequence, from the drivers
and the scripts' usage lines, are:

```
python code/convert_variants.py                 # conversion variant harvest
python code/convert_variants2.py
python code/simulate_conv.py <tag> screen       # e.g. <tag> = bandext
python code/simulate_conv.py bandext full
python code/convert_bandextdb.py
python code/simulate_interact.py bandext maskfix
python code/audit_labels.py                     # target-arithmetic audit
python code/audit_missing.py                    # missingness audit
python code/audit_missing_bandext.py
```

**Step 5 — Phase 6 model and site-generalisation experiments** (all on
the frozen Phase 5 generation files):

```
python code/phase6_ws1_diag.py
python code/phase6_ws1_screens.py
python code/phase6_ws2_bands.py
python code/phase6_ws3_screens.py
python code/phase6_ws4_models.py    # XGBoost / HistGB / MLP comparisons
python code/phase6_gen_coverage.py
python code/phase6_ws5_train.py
```

(**Requires verification**: workstream order follows
`ARCHIVE_MANIFEST.md` §5 and `RESULTS_phase6_model.md`.)

**Step 6 — Phase 7 site calibration.**

```
python code/phase7_gen_char.py      # character-transplant generation (screen)
python code/phase7_ws1_screen.py    # family D screen
python code/phase7_ws2_screen.py    # character screen
python code/phase7_ws3_loso.py      # leave-one-site-out ensemble (rejected)
python code/phase7_interact.py
python code/phase7_perturb.py       # predeclared perturbation diagnostic
python code/phase7_gen_char_full.py # full-protocol generation
```

**Step 7 — Phase 8A exclusion.**

```
python code/phase8_gen_excl.py
```

**Step 8 — Phase 8B graded strengths.**

```
python code/phase8b_gen_graded.py
python code/phase8b_verify_indep.py  # strength × share independence check
python code/phase8b_screen.py
```

(The graded-strength training and evaluation script,
`phase8b_train_eval.py`, is excluded from this sequence; see Step 2's
omission notes. Its records are preserved in the frozen archive.)

**Step 9 — Step 3 (run from inside `step3/`).**

```
cd step3
python splits.py          # hull/date side assignment; writes splits.json
                          # and clip_sides.csv
python extract_wav.py     # extracts kept-side waveforms to a local cache
python families.py        # fits background and passage families,
                          # family-side material only
python gen_corpus.py      # generates calibration, main and null corpora
python verify_stage0.py   # disjointness, hand-check, oracle and baseline
                          # checks; writes stage0_verification.json
python run_stage1.py      # the null test, run first and binding;
                          # writes stage1_results.json
```

Note: `verify_stage0.py`'s baseline check loads the Phase 5 baseline
booster (`models/lgbm_conv_bandext_maskfix.txt`) read-only. That
booster is produced by `train_conv.py`, which is excluded from this
roadmap because running it also writes a submission-shaped file, so
the roadmap as listed cannot regenerate the booster. The booster must
already be available locally from the historical research environment
for this check to be reproduced; regenerating it by any other route
would require a separately authorised, non-submission-producing
method.

**Identifier warning.** `splits.py` writes `clip_sides.csv`, which
contains MMSI vessel identifiers alongside the clip assignments. Any
locally regenerated copy of that metadata contains those identifiers,
must remain local, and must not be copied into the curated public
release.

Master seed for Phases 0–8B is 20261116 (`code/common.py` and every
model record); Step 3's seed is 20261118 (`step3/splits.json`).
Per-phase stream seeds are recorded in `ARCHIVE_MANIFEST.md` §5 and in
the generation metadata files.

## 7. What is deliberately not in the release, and how it is regenerated

The following material is excluded from the proposed release by design:

- **The South African source data, and derived data that could
  reproduce it** (the screened background pool and the feature corpora
  built from it). The competition package carries a no-redistribution
  notice; see `DATA_SOURCES.md`. This material cannot be redistributed
  and can only be re-obtained through the competition.
- **VTUAD audio** (about 3.7 GB of archives, and the ~15 GB of extracted
  waveforms derived from it). VTUAD is CC BY 4.0 and may be
  redistributed with attribution, but it is not bundled: fetch it from
  the Zenodo record cited in `DATA_SOURCES.md`. The extracted waveform
  cache is regenerated by `step3/extract_wav.py`; the converted passage
  pool is regenerated by `code/build_passage_pool.py`.
- **Generated corpora** (the simulator generations and the Step 3
  corpora, as `.npz` files). These are regenerated by the generation
  scripts in §6 from the recorded seeds; the generation metadata files,
  which carry the realised seeds, are part of the record.
- **Trained LightGBM boosters** (about 2 GB of model files). These were
  deliberately excluded from the frozen archive as well. Only some are
  regenerated by the training and screen commands that remain in §6:
  the boosters belonging to the excluded training runs (among them the
  Phase 5 baseline, the Phase 7 full runs and the Phase 8A and 8B
  training and evaluation runs) are deliberately not regenerated by
  the proposed sequence. Every model record JSON carries the
  configuration, seed and best-iteration count needed to identify what
  produced each reported score.
- **Submission-shaped test-prediction CSVs** (the
  `*_test_predictions.csv` files written by the six scripts excluded
  in §6). These use the exact competition submission schema and are
  functionally submission outputs, not diagnostics. They are
  deliberately not regenerated by the roadmap in §6.
- **Feature-importance and other genuine diagnostic CSVs.** These are
  regenerated only where the relevant non-excluded commands in §6
  produce them; no blanket regeneration is claimed for them.

Nothing in this list blocks checking the reported numbers: the great
majority of reported scores trace to machine-readable JSON records in
the frozen archive (see §9 for the three recorded exceptions). The
code plus seeds regenerate the artefacts listed above where §6
retains the producing commands; the outputs of the deliberately
excluded training runs are not regenerated by the proposed sequence
and are verified against their frozen records instead.

## 8. External-data prerequisites

Full reproduction requires two external data sources, neither of which
is bundled:

1. **The competition package** — the released South African acoustics
   (1,715,779 records, ten sites) and the starter notebook. These are
   obtainable only through authorised
   Zindi competition access. **Full reproduction therefore requires
   competition access to the South African acoustic package**; without
   it, the background pool, the simulator generations and every
   downstream score cannot be rebuilt. The starter notebook is the
   organisers' material and is not redistributed here.
2. **VTUAD v1.3.2** — the passage corpus the starter notebook names,
   from Zenodo (DOI 10.5281/zenodo.23037895), CC BY 4.0. See
   `DATA_SOURCES.md` for the exact release, the attribution
   requirements, and the three-part archive reassembly and checksum
   verification procedure.

## 9. Score provenance: the archive manifest and the evidence vault

`ARCHIVE_MANIFEST.md` is the score-provenance index for the whole
programme. For the programme's model scores it records the model record
file, the configuration, the seed set and the data construction behind
it, and it lists SHA-256 checksums for all 242 frozen files. On
18 November 2026, all 242 files in the frozen archive were checked
against their manifest-listed SHA-256 checksums: 242 of 242 matched,
with zero mismatches.

Three exceptions to full machine-readable traceability are recorded
plainly. (a) The B1 baseline's null-check results (median 0.344, mean
0.398 on pure backgrounds) rest on the preserved code — the result is
recorded in the docstring of `code/b1_null_check.py` — and on the prose
record; no output artefact from that run was preserved. (b) Leaderboard
figures (the constant benchmark 0.753867825, the public top score
0.994250562 and Submission 1's own 0.687868143) are externally sourced
competition results, recorded in `SUBMISSIONS.md` and the phase
reports; no leaderboard export exists in the record. (c) The Phase 0
data-audit figures are recorded in `data/AUDIT.md`, which sits in the
working tree and was deliberately not frozen into the archive.

The frozen `archive/` is an **evidence vault, not a complete executable
reproduction bundle**. It freezes the reports, scripts and model records
exactly as they were when the research closed, so that every number in
the write-up can be checked against its source. It deliberately does
not contain the trained boosters, the generated corpora or the bulky
diagnostics (§7), so the archive alone cannot regenerate the scores —
the working code, the recorded seeds and the external data are also
required. A release is a curated copy drawn from this record; the
archive itself is never modified to build one.

The full research narrative, with every finding classified as Measured,
Supported, Unresolved or Rejected, is in `WRITEUP_consolidated.md`. The
phase reports (`RESULTS_*.md`), `STEP3_CLOSURE.md` and
`RESULTS_step3_stage01.md` are the detailed record behind it.

## 10. Authorship

This programme was researched and directed by its owner as a solo Zindi
entry, and the record is released under the owner's name at the owner's
decision. *(Authorship line in its final public form is subject to owner
review before any release.)*
