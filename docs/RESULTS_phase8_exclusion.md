# Phase 8A — The Bird Island exclusion test: results (2026-11-17)

The opening experiment of Phase 8, and the only one run.
Godson directed it with a clear stop/go gate: run this one
experiment, report, and stop. No broad Phase 8 feature search,
no graded transplants, no further experiments were run or are
authorised by this record.

**The question.** Phase 7's site-character transplant family
included Bird Island's own aggregate median shape. Its
site-holdout correction could therefore reflect either (a)
genuine learning of a transferable class of
low-frequency-heavy natural backgrounds, or (b) partial
adaptation to the particular Bird Island distribution. This
test separates the two: it is a test of whether the correction
generalises without access to Bird Island's own aggregate
shape, not a score optimisation.

**Standards (unchanged).** Development minimum **0.96** on the
fixed validation and both cross-checks, all four holdouts
reported separately; submission standard ≈0.998 under
genuinely independent validation. No candidate file produced,
no Zindi slot spent (still 1 of 200), the Zindi site not
visited. The Phase 5 baseline remains the historical
reference; both Phase 7 configurations (the simpler
549-feature version and the 549+D version) are the
experimental references. The exclusion was run in **both**
configurations, each mirroring its Phase 7 counterpart:
excl-549 against Phase 7 char-549, excl-D against Phase 7
char-549D.

## Method — one change, everything else frozen

The Phase 7 generation (`code/phase7_gen_char.py`,
`code/phase7_gen_char_full.py`) was rerun with identical
seeds (trainA 20358118, train 20358119, trainB 20358120),
identical pools and split discipline
(`simulate_conv.build_pools('bandext')`), identical
P_TRANSPLANT (0.6), and the frozen bandext_maskfix
construction throughout — conversion, target arithmetic and
missingness not reopened. The single change
(`code/phase8_gen_excl.py`): **Bird Island's aggregate shape
is removed from the transplant targets.** A background whose
transplant draw selects Bird Island keeps its natural
character; Bird Island's shape is never applied to any
training window.

Controlled-comparison design: the draw keeps the Phase 7
target list's length and order, so the random stream is
consumed draw-for-draw identically to Phase 7; the exclusion
acts only on whether the drawn target is applied. Training
(frozen LightGBM protocol via `phase6_lib.train_lgbm`) and
evaluation (identical frozen holdouts) follow
`code/phase8_train_eval.py`, which also recomputes the Phase 5
baseline's and both Phase 7 families' predictions from their
saved boosters, so all five families are compared paired on
identical windows from one harness. Record:
`models/phase8_excl.json`.

**Verification of the control (Measured).** The exclusion
corpora differ from the Phase 7 corpora in exactly the Bird
Island windows and nothing else:

| set | rows bit-identical to Phase 7 corpus | rows differing | Phase 7 Bird Island transplants |
|---|---|---|---|
| trainA | 37,084 / 40,000 | 2,916 | 2,916 |
| train | 73,864 / 80,000 | 6,136 | (6,136 of 42,433 total) |
| trainB | 37,043 / 40,000 | 2,957 | (2,957 of 20,807 total) |

(NaN-aware row equality; labels identical on identical rows.)
Per-target transplant counts for the seven retained sites
match Phase 7's recorded counts exactly on trainA (Richards
Bay 1,044; Sea Point 2,998; Saldanha Bay 2,731; Durban 2,866;
St Croix Island 2,478; St_Francis 2,889; Fish Hoek 3,082), and
on train/trainB the applied-plus-excluded totals reproduce
Phase 7's recorded transplant totals exactly (42,433 and
20,807). 12,009 training windows across the three sets drew
Bird Island in Phase 7 and keep their natural character here.
**The only difference between the Phase 7 and Phase 8 training
corpora is the treatment of those windows.**

**Harness validation (Measured).** The recomputed Phase 5
baseline (0.9482 / 0.9030 / 0.9469 / 0.9422, real mean 0.2061)
and both Phase 7 families reproduce their recorded Phase 7
numbers exactly at four decimals on every holdout, band
statistic and per-site statistic cross-checked, including the
baseline's Bird Island score (0.9011) and zero bias (+0.0676).

## Results — holdout scores (main / A / B)

| family | random | site | period | long |
|---|---|---|---|---|
| Phase 5 baseline (historical ref) | 0.9482 / 0.9434 / 0.9436 | 0.9030 / 0.9002 / 0.9051 | 0.9469 / 0.9439 / 0.9439 | 0.9422 / 0.9381 / 0.9383 |
| Phase 7 char, 549 | 0.9476 / 0.9437 / 0.9445 | 0.9232 / 0.9141 / 0.9186 | 0.9488 / 0.9448 / 0.9452 | 0.9417 / 0.9372 / 0.9370 |
| Phase 7 char, 549+D | 0.9470 / 0.9435 / 0.9439 | 0.9250 / 0.9154 / 0.9214 | 0.9474 / 0.9445 / 0.9446 | 0.9396 / 0.9353 / 0.9355 |
| **Phase 8 excl, 549** | 0.9475 / 0.9429 / 0.9442 | **0.9159** / 0.9080 / 0.9149 | 0.9481 / 0.9457 / 0.9453 | 0.9412 / 0.9368 / 0.9374 |
| **Phase 8 excl, 549+D** | 0.9467 / 0.9440 / 0.9435 | **0.9205** / 0.9134 / 0.9171 | 0.9483 / 0.9451 / 0.9447 | 0.9402 / 0.9356 / 0.9356 |

Best iterations, exclusion models: 549 config 6,000 / 3,866 /
4,309; D config 5,997 / 5,918 / 4,240 (Phase 7 counterparts:
5,999 / 5,288 / 4,315 and 5,991 / 5,466 / 2,662 — the same
training regime). On the seen holdouts the exclusion models
are within ±0.001 of their Phase 7 counterparts on every main
model; nothing was sacrificed there, and nothing was gained.
The A and B replications reproduce the main models' ordering
on the site holdout in both configurations (exclusion below
full transplant, both above baseline).

## Results — the site holdout in detail (main models)

Share bands: bias / RMSE / SSE share.

| band | Phase 5 baseline | Phase 7 char 549 | Phase 7 char 549+D | Excl 549 | Excl 549+D |
|---|---|---|---|---|---|
| 0 | +0.065 / 0.090 / 21.4% | +0.028 / 0.045 / 8.5% | +0.025 / 0.040 / 7.1% | **+0.032 / 0.053 / 9.8%** | **+0.028 / 0.046 / 8.3%** |
| (0,0.05] | +0.053 / 0.084 / 16.9% | +0.021 / 0.044 / 7.3% | +0.020 / 0.039 / 6.3% | +0.023 / 0.052 / 8.5% | +0.020 / 0.043 / 6.8% |
| (0.05,0.1] | +0.032 / 0.074 / 5.0% | +0.001 / 0.049 / 3.5% | +0.004 / 0.045 / 3.2% | +0.003 / 0.052 / 3.3% | +0.004 / 0.047 / 3.1% |
| (0.1,0.3] | +0.008 / 0.091 / 19.6% | −0.019 / 0.075 / 21.3% | −0.021 / 0.071 / 20.2% | −0.019 / 0.081 / 20.9% | −0.021 / 0.077 / 21.2% |
| (0.3,0.6] | −0.034 / 0.131 / 23.1% | −0.062 / 0.125 / 33.2% | −0.066 / 0.125 / 35.2% | −0.069 / 0.138 / 33.8% | −0.069 / 0.133 / 35.3% |
| (0.6,1] | −0.051 / 0.122 / 14.0% | −0.070 / 0.132 / 26.1% | −0.070 / 0.134 / 28.1% | **−0.078 / 0.138 / 23.7%** | **−0.073 / 0.135 / 25.4%** |
| top two bands' SSE share | 37.1% | 59.3% | 63.3% | 57.5% | 60.6% |

Per site (main models): score / overall bias / zero-band
bias / zero-band RMSE / (0.6,1] bias.

| site | Phase 5 baseline | Phase 7 char 549 | Phase 7 char 549+D | Excl 549 | Excl 549+D |
|---|---|---|---|---|---|
| Bird Island (n=5,540) | 0.9011 / +0.026 / +0.068 / 0.093 / −0.051 | 0.9226 / −0.007 / +0.028 / 0.045 / −0.070 | 0.9246 / −0.008 / +0.025 / 0.041 / −0.071 | **0.9150 / −0.006 / +0.033 / 0.054 / −0.078** | **0.9198 / −0.008 / +0.029 / 0.047 / −0.074** |
| Fish Hoek (n=460) | 0.9284 / +0.003 / +0.032 / 0.055 / −0.050 | 0.9305 / −0.003 / +0.029 / 0.043 / −0.065 | 0.9301 / −0.007 / +0.019 / 0.032 / −0.059 | **0.9283 / −0.005 / +0.030 / 0.043 / −0.071** | **0.9290 / −0.006 / +0.020 / 0.030 / −0.067** |

Real-window diagnostics (consistency only, never a
validation substitute): main-model means — baseline 0.2061;
Phase 7 0.1893 (549) / 0.1823 (D); exclusion 0.1943 (549) /
0.1849 (D). Confidence categories for the exclusion models:
549 — clean 715 (47.7%) at 0.098, ambiguous 618 (41.2%) at
0.252, unresolved 167 (11.1%) at 0.391; D — clean 755
(50.3%) at 0.094, ambiguous 608 (40.5%) at 0.255, unresolved
137 (9.1%) at 0.376.

## The joint criterion — verdict

Godson's criterion, stated before the run: the result that
matters is not simply whether the Bird Island score stays
high; it is whether the zero-share correction **survives**
while high-share under-reading **does not become worse**.

**First half — the zero-share correction survives
(Measured).** Without Bird Island's shape anywhere in
training, the pooled site zero band reads bias **+0.032**
(549) / **+0.028** (D) and RMSE **0.053** / **0.046**, against
the baseline's +0.065 / 0.090 and the full transplant's
+0.028 / 0.045 (549) and +0.025 / 0.040 (D). The exclusion
retains **89% (549) / 91% (D)** of Phase 7's zero-band bias
correction and **83% / 88%** of its RMSE correction. On Bird
Island itself — the site whose shape was removed — the zero
bias is +0.033 / +0.029 against the full transplant's +0.028
/ +0.025 and the baseline's +0.068: the correction on the
excluded site survives nearly intact. The correction does
not collapse; it is not close to collapsing.

**Second half — high-share under-reading became slightly
worse (Measured).** The (0.6,1] bias deepens from −0.070 to
**−0.078** (549 pairing) and from −0.070 to **−0.073** (D
pairing); the (0.3,0.6] bias deepens likewise (−0.062 →
−0.069; −0.066 → −0.069), and the (0.6,1] RMSE rises slightly
(0.132 → 0.138; 0.134 → 0.135). The deepening is small —
0.003 to 0.008 of bias, at the edge of replication noise —
but it is consistent in direction across both configurations
and both top bands, so it is recorded as a real, small
worsening, not dismissed. The top two bands' share of site
SSE does not grow (59.3% → 57.5%; 63.3% → 60.6%), because
total site SSE is larger under exclusion; in bias terms,
which is how the criterion was set, the second half is
**not cleanly met**.

**Classification under Godson's predetermined interpretation
table.** The three readings offered were: correction
substantially survives = strong evidence of transferable
character learning; correction partially survives = both
transferable learning and target-specific adaptation
contributed; correction collapses = Phase 7's apparent
calibration gain depended substantially on Bird Island shape
exposure. The measured outcome is:

> **The correction partially survives — both transferable
> learning and target-specific adaptation contributed.**

The transferable term is dominant: roughly five-sixths to
nine-tenths of the zero-share correction (the phase's named
target) is present with Bird Island's shape entirely absent,
including on Bird Island itself. But a Bird Island specific
term is real and measured: the site holdout gives back
0.0073 of its 0.0202 gain in the 549 configuration and
0.0045 of its 0.0220 gain in the D configuration (site-score
retention 64% / 80%), the give-back replicates in the A and
B models, and the joint criterion's high-share half slipped
slightly. Per Godson's standing wording rule, even the
surviving part is **strong evidence, not absolute proof** of
the transferable-character mechanism.

**Internal control (Measured).** Fish Hoek, whose shape
remained in the transplant family, is essentially unchanged
from Phase 7 under exclusion (score 0.9283 / 0.9290 vs
0.9305 / 0.9301; zero bias +0.030 / +0.020 vs +0.029 /
+0.019). The give-back appears where the shape was removed
and not where it was retained, which supports reading the
Bird Island give-back as Bird Island specific rather than a
general weakening of character training.

## Gate judgement

**Gate NOT MET (Measured)** — for the exclusion models as
for every earlier family. Their best holdout is period at
0.9483 against the 0.96 development minimum; site reads
0.9159 (549) / 0.9205 (D). No candidate file has been
produced and no submission made; slots remain 1 of 200. The
Phase 7 character configuration remains the baseline
reference for any future phase, now with its blindness
caveat bounded by this test: the caveat accounts for a
minority of the site gain, not the bulk of it.

## Standing qualifications (carried from Godson, binding on
interpretation)

- The real-window unresolved fraction rose from 5.9% (Phase
  5 baseline) to 11.2% / 10.1% in the Phase 7 configurations;
  the exclusion models sit in the same elevated range (11.1%
  / 9.1%). The character models agree less often on real
  windows than the baseline family did.
- Perturbation stability does not establish accuracy
  (Phase 7's test showed the models are stable, not that
  they are right).
- The A and B replications agreeing with the main model does
  not bring any of them to the 0.96 gate; agreement is
  consistency, not a pass.

## Reconciliation of two Phase 7 reporting discrepancies

Required by Godson before this phase's results are read
against Phase 7's; both were checked against the underlying
evaluation records, and a short addendum is appended to
`RESULTS_phase7_calibration.md`.

**(a) The ±0.002 tolerance overstates the D configuration's
long holdout (corrected).** The Phase 7 gate judgement states
that random, period and long sit "within ±0.002 of the Phase
5 baseline on the main models". That holds for the 549
configuration (deltas −0.0006 / +0.0019 / −0.0005) but not
for the D configuration: its long holdout is **0.9396
against the baseline's 0.9422, a decline of 0.0026**, outside
the stated band (its random and period deltas, −0.0012 and
+0.0005, are inside). The D section's summary "−0.001 to
−0.002 on random and long" (against the 549 configuration)
is likewise marginally out on long: 0.9396 vs 0.9417 is
−0.0021. Corrected wording, recorded in the addendum: the
±0.002 band holds for every main-model holdout delta of both
configurations **except** the D configuration's long
holdout, which sits 0.0026 below the Phase 5 baseline (and
0.0021 below the 549 configuration). Neither discrepancy
changes any Phase 7 conclusion; the tolerance wording, not
the numbers, was wrong.

**(b) The D configuration's 7.1% zero-band SSE share is
traced and stands.** The figure appears in the Phase 7 gate
judgement but not in the detailed D section (which reports
no SSE share; the 549 section reports 8.5%). It traces
exactly to the underlying evaluation record:
`models/phase7_full_D.json`, `val_sets.val_site.bands_main`,
band `0`, `sse_share` = **0.0712** (main model). It was
independently recomputed in this phase's evaluation from the
saved booster and reproduces as 0.071. The figure is
correct; only its citation was missing. The addendum records
the citation.

## What this experiment does and does not license

It bounds the Phase 7 blindness caveat: Bird Island shape
exposure accounted for a minority of the site calibration
gain (roughly a fifth to a third of the site-score gain,
roughly a tenth of the zero-band correction), and the
remainder is consistent with transferable character
learning. It does not raise any model to the gate, it does
not license a candidate or a submission, and — per the
stop/go direction — **no further Phase 8 work is run on the
back of it**. Any graded-transplant or high-share workstream
is a separate decision for Godson, informed by the finding
that the high-share under-read is now the largest single
component of site error and moved slightly the wrong way
when Bird Island's shape was removed.

## Files

- Generation: `code/phase8_gen_excl.py`; corpora
  `data/sim_p8excl_{train,trainA,trainB}.npz`; record
  `data/sim_p8excl_meta.json`.
- Training/evaluation: `code/phase8_train_eval.py`; models
  `models/phase8_excl_{base,D}{,_A,_B}.txt` with iteration
  records; full metrics `models/phase8_excl.json`.
- Diagnostic test predictions (not submission files):
  `models/phase8_diag_*_test_predictions.csv`.
