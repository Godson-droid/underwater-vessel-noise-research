# Underwater Vessel Noise: a consolidated research report (Phases 0–8B and Step 3)

**Competition:** Underwater Vessel Noise Challenge by the French Embassy in
South Africa (Zindi). **Entrant:** Godson (Godson22), solo.
**Period covered:** 16–18 November 2026, Phases 0 through 8B and Step 3
(Stages 0 and 1).
**Status of this report:** the consolidated account of the completed
research programme, and its final write-up. The Phase 0–8B sections
reproduce the earlier Phase 0–8B report unchanged in substance, including
its settled numbers and wording; the Step 3 section (§11) and the closing
conclusions (§12) integrate the decomposition work that was authorised,
tested and formally closed on 18 November 2026. The research phase is
closed: no candidate was produced, no further submission was made, and
this report is also the record on which any future decision to resume
would have to build.

---

## Abstract

The task is to predict, for each of 1,500 fifteen-minute underwater acoustic
windows, the fraction of the window's total energy that came from a vessel
passage the organisers added to a real recorded background. The score is
1 − RMSE on that fraction. There is no labelled training set: the only
training material is 1,715,779 unlabelled one-minute band-energy records
from ten South African sites, so all labelled data must be manufactured by
simulating the organisers' construction — real backgrounds plus controlled
additions drawn from the VTUAD v1.3.2 passage corpus the starter notebook
names.

This report describes the research programme on that problem — Phases 0–8B,
and Step 3, in which the one genuinely different modelling direction left
open at the end of Phase 8B was taken up under a separate authorisation.
Its findings, in order of importance:

1. The dominant early failure was not the model but the **passage
   conversion**. Every model generation trained on passages converted with
   the starter notebook's deci-decade edge clipping over-read the real
   windows by roughly 0.16 of share. Extending the two edge bands to the
   spectrum edges (band 0 from 0 Hz, band 24 to Nyquist) removed most of
   that distortion in a single step (**Measured**).
2. With the conversion corrected, the largest remaining error was a
   **site-dependent zero-share hallucination**: on backgrounds from unseen
   sites, natural low-frequency character was read as added vessel energy
   (+0.065 bias at true share zero on the site holdout). Training with
   transplanted site characters reduced that bias to +0.025 and halved its
   error, at the cost of under-reading genuinely large additions on unseen
   sites (**Measured**).
3. A controlled exclusion experiment showed that most of that correction
   **transfers**: it survives, at 89–91% of its bias correction, when the
   problem site's own spectral shape is removed from training entirely
   (**Measured**; classified at the time as *the correction partially
   survives*).
4. A graded-strength experiment then mapped the trade-off as a smooth
   dose–response and identified **quarter strength** as the operating point
   that keeps most of the zero-share correction while recovering part of
   the high-share sensitivity full-strength training had sacrificed
   (**Supported**, with the recovery explicitly bounded).
5. **The validation gate was not met.** No configuration in any phase
   reached the pre-set development minimum of 0.96 on the fixed validation
   and both independent cross-checks; the best holdout score recorded in
   the Phase 0–8B programme is 0.9488 and the best site-holdout score is 0.9250. The
   submission standard of approximately 0.998 was never approached, exactly
   one submission was ever made (an early diagnostic that scored below the
   constant benchmark), and this report does not pretend otherwise.
6. A physically motivated alternative — decompose each window by
   estimating the background share of its observed-cell energy, and derive
   the share as one minus that ratio — was **independently constructed,
   tested, and stopped when it failed** (Step 3, §11). A separately
   implemented synthetic validation used passage clips disjoint by hull
   and recording date from the family-side clips, and included an
   unseen-site evaluation drawn from two sites excluded entirely from
   background-family fitting. Its construction passed its verification
   checks (oracle 0.9989; the frozen Phase 5 baseline scored 0.8854 as a
   reference point). Both minimal estimators then failed the
   predeclared null test on pure backgrounds from held-out sites by wide
   margins — mean estimated shares of 0.5747 and 0.0999 against a pass
   line of 0.03 — and the work stopped there, exactly where the protocol
   required, with no main-set run, no tuning and no reinterpretation of
   the thresholds. The decomposition hypothesis, in its two minimal
   forms, is **Rejected** for this programme (**Measured**).

The negative findings are given the same prominence as the positive ones,
because several of them are the load-bearing results: background matching
was refuted on real data and closed; background-relative features and
leave-one-site-out ensembling failed for identified reasons; model class,
loss function, training volume and variation breadth were each shown not to
be the constraint; a competing participant's data-expansion claim was
audited and excluded for a concrete reason; and a physically motivated
decomposition was independently constructed and rejected on a predeclared
null test, reproducing the same unseen-site background failure through an
unrelated mechanism.

**Classification discipline.** Every substantive claim in this report
carries one of four labels, used consistently through the project.
**Measured**: directly demonstrated by an experiment. **Supported**:
consistent with multiple independent results, short of direct
demonstration. **Unresolved**: plausible but not established. **Rejected**:
contradicted by the available evidence.

---

## 1. The problem, and the rules we worked under

Each test window is a 15 × 25 grid: fifteen one-minute observations across
25 frequency bands, given as band-energy summaries in decibels. There is no
raw audio and no labelled real training set. The target,
`share_added_vessel`, is the fraction of the window's total acoustic energy
attributable to the controlled added passage; a second column,
`score_added_vessel`, is a zero-weight diagnostic and was predicted
throughout as a fixed monotone companion of the share (√share). The public
leaderboard reflects about 30% of the test windows; the constant predictor
0.1 scores the benchmark 0.753867825. The public leaderboard's top score of
0.994250562 indicates that very high performance has been reported on the
public subset. It suggests that substantial recoverable signal may exist,
but does not establish equivalent performance on the private test subset.

Three constraints shaped everything that followed.

**Test inputs are inference only.** Nothing was fitted, calibrated,
threshold-selected or pseudo-labelled from the test windows in any phase.
Real-window prediction distributions were used as consistency diagnostics
and are reported as such; a model's mean prediction on real windows is a
diagnostic, never a validation score and never an estimate of the true
test distribution.

**A gate set in advance, and never moved.** A candidate could only be
proposed at a development minimum of 0.96 on the fixed validation *and*
both independent cross-checks (two further models trained on disjoint
splits), with all four holdouts — random, site, period and long — reported
separately. The eventual submission standard was approximately 0.998 under
genuinely independent validation. (An earlier 0.93 standard governed
Phases 1–4 and was superseded by Godson's tightened standards from Phase 5
onward; each phase's report states the standard in force at the time.)
Submission budget: 2 per day, 200 in total. Exactly one slot was ever spent.

**Reproducibility as a working habit.** Fixed seeds, paired comparisons on
identical windows, predeclared experiment designs and survival rules, and a
frozen archive of every report, script and model record, indexed by an
archive manifest that ties each reported score to the exact record file,
configuration, seed set and data construction behind it.

The modelling approach was fixed from Phase 0: build a simulator that
mirrors the organisers' construction. Backgrounds are screened windows
from the released South African acoustics (87,653 clean 15-minute windows;
Richards Bay's sentinel-contaminated records screened out). Passages are
VTUAD v1.3.2 recordings (CC BY 4.0, Zenodo DOI 10.5281/zenodo.23037895,
upstream Ocean Networks Canada) converted to the same band representation.
Mixing is done in the linear energy domain at an exact target share, with
the label defined as the realised share over the window's observed cells.
The base model uses LightGBM with 549 fixed per-window features, trained
under a frozen protocol. Specific experiments extended this representation
to 602 or 605 features, as documented in their respective phase records.
Two independent replications (A and B, trained on splits disjoint in both
backgrounds and passage clips) accompany every full run as the cross-check.

---

## 2. Why conventional features failed to transfer (Phases 0–4)

### 2.1 The simple statistic fails its null check

The natural first estimator for this problem is a low-band excess statistic:
vessel radiated noise concentrates below 1 kHz, so measure the excess energy
in the low bands over the window's own quiet level and call that the added
share. Built as a fixed transform (baseline B1), it produces plausible
numbers on test windows — median 0.526. Run on 2,100 ambient-only windows
whose true share is zero by construction, the same statistic reads a median
of 0.344 and a mean of 0.398 (**Measured**). Ambient low-band fluctuation,
including genuine vessel traffic that is not a controlled addition, produces
most of its output. The lesson generalised and was applied to every later
method: in this problem, any estimator that has not passed a null check on
pure backgrounds is not yet an estimator. The B1 statistic survived only as
one feature among 549.

### 2.2 Synthetic success, real failure

The first simulator (v1) and its model scored 0.905–0.917 on held-out
synthetic windows, far above the constant baseline's 0.75 on the same sets.
But the model's predictions on the real test windows averaged 0.330,
against a synthetic share mean of 0.17 — the model was reading far more
"added-like" energy in real windows than its training world contained. The
second simulator (v2) fixed the weaknesses v1's report had named — passage
events assembled to span most of the window instead of a median three
minutes, wider time-warp and colouring, level-stratified backgrounds — and
beat v1 on every v2 validation set, including a clear gain on long events
(0.8981 against 0.8685). Its real-window predictions did not move at all:
mean 0.330, correlating 0.958 with v1's. Event duration was a real mismatch
and fixing it was real progress *inside* the simulator; it was not the
transfer failure.

The v2 model was submitted once, as a deliberate smoke test, on 16 November
2026. It scored **0.687868143** on the public board, rank 600, below the
constant benchmark — the direct measurement that the simulator-to-real gap
was large and that synthetic scores were not evidence about the real task.
That was the first and last submission; every later phase worked under the
rule that a failed gate means diagnosis, not another slot.

Two public scores plus our own predictions also pinned down what the truth
could look like, by arithmetic alone (Phase 3): the plausible true mean
share lies in the band 0.13–0.20, our predictions correlate with the truth
at only about 0.26–0.51, and even the optimally shrunk version of the v2
predictions caps near 0.78–0.79. Shrinkage was a floor defence, never a
route (**Measured**).

### 2.3 A detour that had to be closed: background matching

The near-perfect top score suggested a different route: if the test
backgrounds are verbatim windows from the released acoustics, a test
window's background could be found in the release and the added component
recovered by subtraction. Phase 3 tested this properly. Two independent
matchers were built, and on synthetic windows built the organisers' way the
machinery worked exactly as hoped where it worked at all: when a matcher
found the true background, subtraction was oracle-quality (0.944–0.950
against the oracle's 0.9528), and an exact-minute fingerprint — untouched
minutes reproduce the test window cell-for-cell on the 0.1 dB grid —
certified its own answers perfectly, scoring 0.9493 on the 86% of synthetic
windows where it fired.

On real windows the fingerprint fired **zero times in 60 sampled windows**,
across a median of 5,829 candidates per window; no best candidate contained
a single exact minute, and even after best gain alignment the best release
candidates still violated the physical background constraint in 8% of
cells. The deployable matchers also failed among themselves: the two
independent matchers disagreed at RMSE 0.3057, an order worse than the
agreement the gate would need (**Measured**). The hypothesis that the test
backgrounds are in the release is **Rejected**, and background matching was
closed as a modelling direction by ruling, permanently — including its
gated and hybrid variants, which cap near 0.85 on synthetic data in any
case. This is recorded as a first-class result, not an embarrassment: it
eliminated the most plausible explanation of the top score and redirected
the work to where the failure actually lived.

### 2.4 Fixing the label did not fix the transfer

Simulator v3 (Phase 4) fixed the one thing provably broken in the
construction: the label. Under the original definition (share computed
over the full window before missingness), even an oracle that knows the
true background caps at 0.9510, which makes a 0.994-class score
arithmetically impossible. Defining the label as the realised share over
the **observed cells** raised the construction's oracle ceiling to 0.9983,
consistent with the real top score (**Measured**). V3 also trained against
harder backgrounds — windows containing genuine vessel traffic (never
counted toward the label), blended background mixtures, channel jitter and
ambient drift. Synthetic scores rose to 0.926–0.928. The real-window mean
did not move: 0.3372, the same over-reading as v2.

V3 also produced the project's most useful methodological warning. Scored
on identical windows from the *other* simulator generation, v3's apparent
gains evaporated: a model can look better simply because the world it is
tested on is the world it was trained on. From that phase on, no claim was
allowed to rest on a single generation's holdouts — paired cross-generation
comparisons, and ultimately the real-window diagnostic, arbitrate. The
diagnosis at the end of Phase 4 was precise: four models differing in
training data all read real windows at 0.335–0.339 while reading every
synthetic background mode at the prior's 0.18, so the gap was not in the
backgrounds we could generate, the label, or the model. The remaining
candidate location, ranked first by the evidence, was the passage side —
specifically the conversion from audio to band energies, which the starter
notebook itself calls an unverified reconstruction of the organisers'
export.

---

## 3. The conversion was the fault (Phase 5)

Phase 5 treated the passage conversion as the suspect and screened ten
variants under an identical protocol, changing exactly one axis at a time:
aggregation (mean power, magnitude, dB-mean, median), window function,
detrending, segment length (1 s, 5 s, 10 s, whole minute), overlap, and the
band boundaries themselves. The variant pipeline was first verified to
reproduce the original passage harvest bit-exactly (maximum deviation
0.00 dB), so every difference measured is a variant's mathematics, not
pipeline drift.

Nine of the ten variants left the real-window mean between 0.319 and 0.336,
against v3's 0.337 (**Measured**, variant by variant). The two robust
aggregators moved it slightly (dB-mean 0.319, median 0.324) — **Supported**
as a small effect in the robust direction, **Rejected** as a sufficient
account, since at most a tenth of the gap moves. The STFT axes and minute
alignment did not move it at all.

One variant transformed the picture. **Bandext** changes nothing except the
two edge bands: band 0 integrates from 0 Hz instead of its deci-decade
lower edge (about 56 Hz), and band 24 integrates up to Nyquist (16 kHz)
instead of about 8.9 kHz. Its real-window mean fell to **0.2042**, a
movement of 0.133 against a gap of about 0.16, and it simultaneously
produced the best synthetic holdouts of the phase. The mechanism is
consistent with every measurement (**Supported**): the released background
windows are the organisers' own conversion, so they already contain
full-range edge bands — real band 0 carries the large sub-56 Hz energy of
ocean recordings. Our clipped conversion made synthetic additions carry far
less edge-band energy, relative to the interior, than real additions do, and
models calibrated on that distortion systematically over-read real windows.

Two companion audits closed their questions in the same phase. The target
arithmetic audit scored every plausible label definition by its oracle
ceiling: the observed-cells sum is the only definition consistent with the
real top score and the label moments; mean-based, band-restricted and
amplitude-domain definitions are **Rejected (Measured)**, with ceilings of
0.78–0.94, and the rounding-order question is immaterial (RMSE 0.0008). The
missingness audit showed the real mask is an independent, edge-minute
process carrying no label information (**Rejected (Measured)** as an
explanation), and that the one real anomaly — a louder minute-1-missing
sub-population — was mostly another symptom of the same conversion error:
its excess over complete windows collapsed from +0.065 under v3 to +0.0075
under bandext. A domain-shift hypothesis was also tested directly: training
for invariance to release-sized per-band channel responses (RMS about
11 dB, the size the release exhibits between its own sites) changed
real-window readings not at all (0.3350 against 0.337) — **Rejected
(Measured)** for that family.

On the corrected base, the small corrections were tested factorially.
Maskfix — training on the mask form the real windows actually have — added
about 0.01 of real-mean improvement at no synthetic cost and mainly repaired
the windows models had disagreed about most (the unresolved group's mean
fell from 0.457 to 0.391). D-mean aggregation added nothing on the corrected
conversion (**Measured**); its earlier small effect had been a partial
symptom of the same distortion. The combination cell was therefore not run,
under the standing rule that only independently justified variants are
combined.

The resulting construction, **bandext_maskfix**, became and remains the
baseline: holdouts 0.9482 random / 0.9030 site / 0.9469 period / 0.9422
long, cross-checks at 0.938–0.944, construction ceiling 0.9981, real-window
mean 0.2061 with the clean majority of real windows (51.3%) reading 0.124 —
inside the plausible band for the first time. The transfer failure, as a
phenomenon, was solved. What remained was ordinary, stubborn model error on
a now-correct simulator, plus one specific weakness the phase exposed
rather than created: the site holdout, at 0.9030, was the weakest number in
the table, and Phase 5's own analysis pointed at the first band's site
character as the reason.

---

## 4. The model was not the bottleneck (Phase 6)

Phase 6 worked on the frozen bandext_maskfix construction — conversion,
labels and missingness were not reopened — and asked where the remaining
error lives. Its share-band analysis split the failure into two components
that structured everything afterwards (**Measured**):

- On seen-site holdouts, textbook shrinkage: slight over-reading at low
  shares, under-reading above 0.3, with the 0.1–0.6 bands carrying about
  58% of squared error. A persistent estimation difficulty, present in every model
  tried.
- On the site holdout, a different failure entirely: at true share zero the
  bias is **+0.065** (RMSE 0.090 against 0.026 on random), the zero band
  alone carrying 21.4% of the site holdout's squared error. Split by site,
  Bird Island (0.9011, bias +0.026) is the problem; Fish Hoek (0.9284) is
  essentially fine.

Band 0 sits at the centre of both stories. It holds 32% of the model's
gain and is the dominant vessel channel — permuting its features costs
0.194 of score — and it is also site-confounded: two band-0 level features
alone identify a background's site at 0.42 against 0.16 chance. But it is
not an exclusive fingerprint: removing band 0 entirely leaves site
identifiability at 0.977, because site character saturates the whole
spectrum. The failure is therefore **calibration, not identity leakage**
(**Measured**): at unseen sites the natural band-0 level sits higher than
the training range, and the learned mapping from band-0 level to added
share over-fires.

The phase then eliminated, one at a time, the obvious remedies. Deleting
the band-0 features is strictly harmful on every holdout (**Rejected**).
Leave-site-out early stopping and site-balanced weights are both null.
Three feature families were screened: per-band excess (**Rejected** —
the base set already encodes it), temporal concentration and spectral-shape
residuals (small screen gains, carried forward). Four model alternatives
on identical windows — XGBoost, histogram gradient boosting, an MLP, and a
Huber-loss variant — all tie LightGBM or trade the site holdout for the
random one: **model class and loss are not the constraint (Measured)**.
Doubling training volume bought +0.003–0.004 on seen holdouts and nothing
on site; broadening the variation axes at equal volume was a synthetic wash
and worsened the real diagnostic. Neither volume nor breadth of the
existing axes is the answer (**Measured**).

The phase's discipline was tested at its end, and held. The justified
combination of the two surviving feature families, run at full protocol,
scored 0.9487 / 0.9012 / 0.9482 / 0.9439 against the baseline's 0.9482 /
0.9030 / 0.9469 / 0.9422 — a random-for-site trade of exactly the kind ruled
out in advance. It was reported as a regression, not progress, and the
Phase 5 baseline remained the best construction. One signal was left
honestly open (**Unresolved**): the combined models read real windows far
lower (main mean 0.172, clean majority 0.094), which might be a genuine
over-read correction that synthetic holdouts cannot see, or might be
under-reading. Phase 7's predeclared perturbation test later showed those
lower readings are *stable* under physically defensible perturbations —
the models are not perched on a knife edge — while stating in the same
breath that stability is not correctness.

---

## 5. Background character causes the hallucination, and character training corrects it (Phase 7)

Phase 7's central objective was narrow: reduce the zero-share hallucination
on unseen sites without sacrificing the corrected passage signature or the
other three holdouts. Three independent levers were tested, and the
contrast between them is the phase's real content.

**Background-relative features (family D)** reference each window against
its own quiet minutes. They improved the site holdout slightly at screen
scale (0.9042 against baseline A's 0.9002) but did not move the zero-band
bias at all (+0.064 against +0.064): the whole window, quiet minutes
included, carries the site character, so the self-estimated background
absorbs the very shift it was meant to expose (**Measured** interpretation;
the family itself **Supported**, small).

**Leave-one-site-out ensembling** — six models, each blind to one training
site, averaged — made things worse: site 0.8979, zero-band bias deepening
to +0.078 (**Rejected, Measured**). The members' unseen-site errors are
correlated, because every site-blind model leans on the same pooled
level-to-share mapping; averaging cannot cancel a shared bias. Site
diversity in the training set does not create calibration diversity at an
unseen site.

**Site-character synthesis** attacked the mechanism directly. Each site's
character was summarised as its median per-band energy-share vector, and
drawn backgrounds were transplanted onto other sites' characters with
total energy preserved. The measured characters explain the pathology:
Bird Island's band-0 share is **1.70 times** the all-site median — its
natural backgrounds are band-0-heavy in exactly the way the baseline reads
as vessel energy. At full protocol, character training delivered the
phase's objective (**Measured**): site holdout 0.9030 → **0.9232** (549
features) and **0.9250** (with family D); site zero-band bias +0.065 →
**+0.028 / +0.025**; zero-band RMSE 0.090 → **0.045 / 0.040**; the zero
band's share of site squared error 21.4% → **8.5% / 7.1%**; and the seen
holdouts stayed within noise of the baseline (period best, 0.9488). The
A and B replications agreed throughout.

The correction has a price, and the report states it as part of the result
rather than a footnote: having learned that band-0-heavy backgrounds are
usually natural, the model discounts genuine large additions on unfamiliar
backgrounds too. On the site holdout the top two share bands come to carry
about 59–63% of squared error, with the (0.6, 1] bias deepening to −0.070.
The error did not vanish; it migrated upward.

Two qualifications were recorded at the time and bind any reading of
Phase 7. First, a **blindness caveat**: the transplant family included the
holdout sites' aggregate median shapes — by explicit direction, with no
holdout window trained on — so the site score could carry some upward
flavour from shape exposure. One internal control argued against pure
memorisation (Fish Hoek's shape is equally in the family, yet Fish Hoek
did not improve), but the caveat was substantial enough to require its own
experiment. Second, the character models' real-window behaviour is
double-edged: they read lower where they agree (clean means down to 0.095),
but they agree less often — the unresolved fraction of real windows rose
from 5.9% under the baseline to 10.1–11.2% (**Measured**). Lower readings
are not thereby truer readings.

---

## 6. How much of the correction transfers (Phase 8A)

Phase 8A ran exactly one experiment, designed to separate transferable
learning from target-specific adaptation: the Phase 7 generation rerun
with Bird Island's aggregate shape removed from the transplant targets, so
a draw selecting Bird Island leaves the background natural. Everything else
— seeds, pools, features, protocol, evaluation windows — was frozen, and
the control was verified before any result was read: the exclusion corpora
are bit-identical to the Phase 7 corpora except in precisely the **12,009
windows** that had drawn Bird Island, and the evaluation harness reproduced
every Phase 5 and Phase 7 number exactly from the saved models.

The zero-share correction **survives** without Bird Island's shape
(**Measured**): site zero-band bias +0.032 / +0.028 (549 / 549+D
configurations) against the baseline's +0.065 and full-strength Phase 7's
+0.028 / +0.025 — retention of 89% / 91% of the bias correction and 83% /
88% of the RMSE correction, holding on Bird Island itself (+0.033 / +0.029)
as well as pooled. The site score gives back a minority of its gain
(retention 64% / 80%), and Fish Hoek — whose shape stayed in the family —
is essentially unchanged, which supports reading the give-back as
Bird-Island-specific rather than a general weakening.

The experiment's joint criterion, fixed in advance, had two halves: the
correction survives, *and* high-share under-reading does not worsen. The
first half was met. The second was **not cleanly met (Measured)**: the
(0.6, 1] bias deepened slightly in both configurations (−0.070 → −0.078
and −0.073), a small change at the edge of replication noise but consistent
in direction, even though the top bands' share of site error did not grow.
Under the interpretation table set before the run, the classification is:
**the correction partially survives** — transferable learning dominant
(five-sixths to nine-tenths of the zero-band correction), with a real,
bounded Bird-Island-specific term. That is strong evidence for the
mechanism, and the record says plainly that it is not absolute proof, and
that Phase 7's site numbers carried some optimism from shape exposure —
now measured, and a minority share.

Phase 8A also closed two reporting discrepancies in the Phase 7 record: a
tolerance phrase that overstated one configuration's long-holdout parity
(the decline is 0.0026, outside the stated ±0.002 band; corrected wording
recorded), and a zero-band SSE figure (7.1%) whose citation was missing —
traced to the underlying evaluation record and independently recomputed;
the figure stands. Both are preserved here because they are part of what
the project means by verification: a number that cannot be traced is
treated as not yet a result.

---

## 7. The trade-off, mapped: graded character strength (Phase 8B)

Phase 8B asked the question Phase 7's result implied: if character training
trades low-share accuracy against high-share sensitivity, is there a
strength at which the trade is favourable? Character strength was varied
on a predeclared graded scale — 0, 0.25, 0.5, 0.75, 1.0, where 0 is no
transplant and 1.0 is the full Phase 7 transplant — with total background
energy preserved, and, as the experiment's central requirement, **every
strength level crossed with the same drawn-share distribution**, so
stronger transplants could not systematically receive larger or smaller
additions. The generation was paired: identical draw sequences at every
level, verified before training, and the full-strength corpus reproduced
Phase 7's corpus bit-for-bit (40,000 of 40,000 rows) — the pipeline
validation on which the whole comparison rests (**Measured**).

One qualification belongs in the body of any account of this experiment,
not in a footnote. The drawn shares are identical across levels by
construction, but the realised labels are computed against observed-minute
background energy, which the transplant preserves only to second order.
The labels are therefore exactly identical for **90.62%** of windows; the
mean absolute drift between the endpoints is 9.1×10⁻⁴, but a small number
of mix-first windows differ individually by up to **0.605**. The correct
statement, and the wording rule adopted for all summaries of this work, is:
**independent drawn-share assignment with small measured realised-label
variation** — never identical labels across strength levels.

The screen produced a smooth dose–response (**Measured**). Zero-band bias
falls monotonically with strength (+0.073 → +0.038 → +0.031 → +0.029 →
+0.028) and saturates by half strength; the (0.6, 1] penalty grows with
strength (−0.067 → −0.069 → −0.081 → −0.089 → −0.088), with most of the
penalty arriving between quarter and three-quarter strength while most of
the zero correction is already present at quarter strength. The seen
holdouts are flat across the whole scale (within ±0.002): the strength
dial trades low-share against high-share calibration on unseen sites only,
which is exactly the axis the experiment targeted. Under a survival rule
declared before the screen ran, only **quarter strength** advanced to the
full protocol — half strength missed the high-share condition by 0.003.

At full protocol, judged against five questions fixed in advance:

1. **The zero-share correction survives** on the site holdout and both
   cross-checks (**Supported**): main-model zero bias +0.030 / +0.028, at
   the full-strength level, with most of the correction retained on A and B.
2. **The high-share improvement replicates, partially** (**Supported,
   bounded**): the (0.6, 1] bias improves against full strength on the main
   model and both cross-checks in both configurations (−0.056 / −0.064 on
   the mains against −0.070), but remains worse than the untreated
   baseline's −0.051. Quarter strength recovers part of the
   character-induced penalty; it does not erase it.
3. **The result holds per site** (**Supported**), driven by Bird Island;
   one noise-level reversal at Fish Hoek in a single cell is recorded as
   **Unresolved** rather than smoothed over.
4. **The seen holdouts are preserved** (**Measured**), all within ±0.002 of
   the reference families; quarter strength's long-holdout score, 0.9426 in
   the 549 configuration, is the best long score of any family.
5. **A/B agreement remains in the established range** (**Measured**),
   0.030–0.035, tighter than the baseline family on the site holdout.

The intermediate share bands interpolate monotonically between the baseline
and full-strength values in both configurations, and the (0.3, 0.6] band
actually improves on full strength: the extremes were not repaired at the
middle's expense (**Measured**). Site scores at quarter strength are
0.9163 / 0.9218 — below full strength's 0.9232 / 0.9250, above both Phase 8A
exclusion configurations in the D pairing — so the joint criterion (zero
correction intact, high-share bias toward zero, no recreated hallucination)
is met in its Supported form, at the price of headline site score.

The real-window diagnostics are reported under the same discipline as
everywhere else in this project: quarter strength's clean-window means
(0.103 and 0.095) sit at full-strength level, and its overall real-window
means are 0.2032 and 0.1869. Those overall means are **model diagnostics,
not estimates of the true test prevalence**, and the unresolved fractions
remain above the Phase 5 baseline's. Nothing in this section converts them
into a claim about the real distribution, because nothing in the Phase 0–8B programme
has validated them against real labels, which do not exist outside the
organisers' process.

---

## 8. An external claim, audited and excluded

During Phase 7, a competing participant published a declaration describing
an expanded training-side source bank: 36 additional strict vessel
recordings from the same public VTUAD release, disjoint from his own
18-clip pilot, verified against the released checksum ledger. Because a
larger passage pool is an obvious candidate improvement, the claim was
audited against five requirements before any thought of incorporation:
inventory, provenance, passage diversity, split integrity and incremental
value.

The audit's findings (**Measured**, from the declaration text itself):
provenance is met as far as a public declaration can meet it — version,
DOI, licence, archive parts and the checksum ledger all named, which also
confirms the recordings are unchanged. The inventory is partial (a count,
with exact filenames retained privately) and split integrity is asserted
rather than demonstrated. On the decisive points: the 36 recordings are
new only relative to the author's own pilot; they come from the same
public release that defines everyone's pool, from which this project's
harvest already converts 542 clips — relative to our pool the expansion
adds nothing. And no performance benefit is offered or tested: the
declaration is a declaration, not a results post. One further detail
mattered. The declared conversion follows the starter approach and does
not document the corrected edge-band treatment that Phase 5 proved was
the principal source of transfer failure, so conversion compatibility is
**not established**: a larger pool processed through an uncorrected
conversion is not assumed to improve a model whose central lesson is that
the conversion was the fault.

The ruling, recorded verbatim in substance: source expansion genuine
within the author's experiment; new material for this project, none;
conversion compatibility not established; performance benefit untested;
**excluded**, and the workstream permanently closed for Phase 7. We record
it here as a useful negative result — a competing data-expansion claim
investigated, its provenance and source overlap verified, and incorporation
rejected for a concrete reason rather than by instinct.

---

## 9. Why the gate was not met, stated plainly

The development gate — 0.96 on the fixed validation and both cross-checks,
across all four holdouts — was not met by any configuration in any phase.
The submission standard of approximately 0.998 was not approached. The
scorecard, against the requirements as they stood at the end:

| Requirement | Verdict |
|---|---|
| Generation identity and paired design (Phase 8B) | Measured |
| Graded dose–response in character strength | Measured |
| Zero-share correction retained at quarter strength | Supported |
| High-share sensitivity partially recovered | Supported, bounded |
| No recreation of the zero-share hallucination | Supported |
| Bird Island / Fish Hoek behaviour | Supported, with one Unresolved cell |
| Intermediate-band behaviour | Measured |
| A/B agreement within the established range | Measured |
| 0.96 development gate | **Not met** |
| Submission eligibility | **Not established** |

The arithmetic of the shortfall is not mysterious, and the phases measured
its components separately. The seen-holdout ceiling has sat at about 0.948
since Phase 5 and did not move under changes of model class, loss, feature
families, training volume or variation breadth. The mid- and high-share
bands retain a persistent estimation difficulty under every model, feature
and variation strategy tested. Whether this reflects a fundamental
identifiability limit or a remaining mismatch in the simulated observation
process has not been established. The site holdout improved
more than any other number in the project (0.9030 → 0.9250 at best) and
still sits 0.035 below the gate, with its remaining error concentrated in
high-share under-reading that character calibration itself induces. The
cross-check models sit generally below the mains, as honest
replications do. The construction ceiling of 0.9981 demonstrates that the
target is arithmetically recoverable under our mirror of the organisers'
process. Within that construction, the remaining model-to-oracle gap is
estimation error. Whether residual simulator-to-real mismatch also
contributes to real-test error remains unresolved, because the real test
windows have no available ground-truth labels.

It is equally plain what the work did establish. The transfer failure that
made the first submission score below a constant was found, localised to a
two-band boundary choice in a conversion the organisers themselves describe
as unverified, and fixed. The site-calibration failure that replaced it was
explained mechanistically — natural background character mimicking the
spectral signature of added energy — and substantially corrected, with the
correction's transferability bounded by a controlled exclusion experiment
and its strength trade-off mapped by a graded one. Neither quarter-strength
configuration is a replacement submission candidate, and none is declared
one. The reference structure the project ends with is: Phase 5, the
pre-calibration reference; Phase 7, the full-strength character-calibration
reference; Phase 8A, the bound on that calibration's transferability;
Phase 8B, the graded-strength compromise between the two competing errors.

Model development stops here, by decision rather than by exhaustion of
ideas, on the judgement that another round of feature engineering or
calibration on the same pipeline would most likely produce more of the
same. The experimental record is frozen as a traceable research record — every
phase report, script, configuration and model record, with a manifest
tying each reported score to its exact provenance. It is not a fully
self-contained executable reproduction bundle: the LightGBM booster files,
the generated training and evaluation corpora, and the per-model
prediction and feature-importance diagnostics were deliberately excluded
and are listed as exclusions in the manifest, so exact reproduction would
require regenerating them from the archived code, seeds and source data.
What the frozen record does guarantee is that the work can be resumed
from evidence rather than from memory, and that this write-up's numbers
can be checked against their sources.

---

## 10. Outlook (the hypothesis recorded at the close of Phase 8B)

One genuinely different direction remains on the table, recorded here only
as an outlook. Every model in this project predicts the share directly
from window features. An alternative formulation would decompose the
window physically: estimate the background contribution and the
added-vessel contribution as separate quantities, and derive the share
from those two estimates. That is a different modelling hypothesis, not a
variation of the existing pipeline. It has not been designed, prototyped
or validated, no experiment on it is authorised, and nothing in this
report should be read as evidence for or against it. If the project
resumes, it would begin with a carefully designed proposal and independent
validation of that decomposition idea — or with a decision that the
frozen record and this write-up are the final deliverables.

That resumption is what happened. On 18 November 2026 the decomposition
idea was authorised as Step 3, under a written proposal with its
validation path and its kill and promise criteria fixed in advance, and
with the Phase 0–8B record left frozen. The next section reports what was
built, what was measured, and why the work stopped where it did.

---

## 11. The decomposition hypothesis, tested and rejected (Step 3)

Step 3 is not a continuation of the Phase 0–8B pipeline, and it should not
be read as "another model was tried". It is the one physically motivated
alternative the programme had left: an estimator built on a different
mechanism from every Phase 0–8B model, independently constructed against
a separately built validation, tested against criteria fixed before any
result was seen, and stopped at the first of those criteria when it
failed. Its negative result is a finding of the programme, given the same
prominence here as the Phase 5 conversion result, because it closes the
question the Phase 0–8B record deliberately left open (§10).

### 11.1 The hypothesis and the authorisation

Every model in Phases 0–8B predicts the share directly from window
features. The decomposition formulation instead estimates the background
contribution to a window's observed-cell energy and derives the share as
one minus that ratio, so the added passage is what remains after the
background is accounted for, rather than a pattern recognised in the
window as a whole. The project's first lesson (§2.1) was that in this
problem an estimator that has not passed a null check on pure backgrounds
is not yet an estimator; Step 3 was designed around that lesson, with the
null check as its first and decisive gate.

The work was authorised on 18 November 2026 under Option A of a written
proposal: Stage 0 and Stage 1 only. Stage 0 had to establish a genuinely
useful independent synthetic validation, including oracle and baseline
checks. Stage 1 had to test the minimal estimators against predeclared
null and performance criteria. Any kill criterion ends the work; an
intermediate result means stop and report, not adjust thresholds; Stage 2
requires a separate decision. No candidate file, no Zindi submission and
no submission-slot expenditure were authorised, and no inference on the
real test set was run at any point. The purpose, in the terms of the
approval, was to find out cheaply and rigorously whether estimating the
physical background contribution offers something the previous regression
approach could not.

One qualification binds every number in this section, and is stated here
in the form in which it was imposed: the independent construction is
independent in implementation and held-out material, but it still uses
the same underlying public data sources and synthetic energy-additive
assumptions as the frozen programme. It is an independent synthetic
validation. It must never be represented as independent real-world
validation.

### 11.2 Stage 0: an independent construction, verified

The validation (called I1 in the working records) was built with fresh
code only; no archived simulator, converter or label code was imported or
reused. Its converter implements the settled bandext definition (band 0
integrated from 0 Hz, band 24 to Nyquist) and was verified against the
archived bandext conversion of an unvaried clip: maximum absolute
difference 0.0006 dB, mean difference 0.0000 (**Measured**).

Passage clips were separated by hull and date between family fitting
and I1 evaluation. For backgrounds, the two unseen sites were excluded
entirely from family fitting, while the seen and period evaluation
windows were held out by window partition within the six family sites.
The separation was re-verified from the saved split files. Passage clips were assigned to sides by hull and by date
(rectangle assignment refined by seeded annealing), and a clip was kept
only where the two assignments agree: family side 212 clips (83 hulls,
15 dates); I1 side 211 clips (94 hulls, 18 dates); 102 clips excluded by
assignment disagreement; 17 class-archive clips excluded for lack of hull
or date metadata. Backgrounds: Bird Island and Fish Hoek, the frozen
programme's unseen sites, are held out entirely — their windows carry
only evaluation (2,909) and null (550, disjoint) roles. Within the six
family sites, the latest 15% of windows by start time form a period
holdout (12,629); the remainder splits into disjoint fit (35,782) and
seen-evaluation (35,783) halves. The share prior was deliberately
different from the frozen construction (25% exact zero, otherwise uniform
on [0, 0.9]). Corpora produced: a calibration set of 1,500 windows from
family-side material only; a main set of 2,000 windows (1,192 seen, 293
period, 515 unseen-site; label mean 0.3346, zero fraction 0.264); and a
null set of the 550 held-out-site background windows with nothing added.
Estimator-facing files carry no truth; truth was held in separate files
read only by the scorer, the oracle check and the hand-check.

Verification was independent of the construction code. Disjointness was
re-verified from the saved split files: **0 shared hulls, 0 shared
dates**; the unseen sites carry only their evaluation and null roles;
the null and main unseen-site sets are disjoint partitions (**Measured**).
Labels for a random sample of main windows were recomputed from the
stored component energies by a separate explicit-loop implementation:
worst label difference 2.7e-09, worst summary difference 6.1e-06 dB — the
harness arithmetic is exact (**Measured**). Two process facts are recorded
for honesty: the first corpus generation run, held in memory, was
destroyed by a machine replacement at about 1,250 of 1,500 calibration
windows, and generation was rewritten to save resumable 250-window
chunks, after which a second process death cost nothing; and the
construction's time-warp step, FFT resampling, discards source energy
above the resampled Nyquist, as standard resampling does — a property of
the construction, noted rather than hidden.

The **oracle check** lands where the proposal said it should: an
estimator given the true background energies scores **0.9989** on the
main set, at the construction's stated ceiling of about 0.998
(**Measured**, with the standing caveat that a ceiling computed under a
construction's own label definition is partly an identity). Under this
construction the labels are recoverable when the true background is
known; the estimand is well posed here, and the remaining difficulty is
estimation — exactly the question Stage 1 asks. The oracle score does
not demonstrate that an estimator can recover the background, or that the
same recovery is possible on real test windows.

The **baseline check** supplies the reference point: the frozen Phase 5
baseline, run read-only and unmodified on I1 material, scores **0.8854**
overall (seen 0.8909, period 0.8771, unseen-site 0.8783; unseen-site
true-zero-share bias +0.0579). The classification, stated as the proposal
requires, is **Supported**: the baseline is materially weaker on I1 than
in family, which is evidence that I1 is a genuinely different and harder
test rather than a restatement of the frozen construction; the baseline
score does not by itself validate I1, and on this material the frozen
regressor would itself fall below the 0.93 floor of the Stage 1 lines.

Stage 0's own kill criterion — that hull-disjoint and date-disjoint
material of usable size cannot be assembled — did not fire. **Stage 0
stands**: a verified independent synthetic validation now exists, and it
remains available if further work is ever authorised.

### 11.3 The lines, fixed before the results

The null check runs first. On pure backgrounds from held-out sites under
the independent construction (true share zero), an estimator passes only
if its mean estimated share is at most 0.03 **and** its median at most
0.02.

Kill, if any of: the null check fails; or the I1 overall score is below
0.93; or unseen-site true-zero-share bias is above +0.040; or the (0.6, 1] share-band
bias is more negative than −0.070. Promise — the only outcome that may
recommend Stage 2: I1 at least 0.96 overall, the unseen-site subset at
least 0.93, unseen-site true-zero-share bias at most +0.030, (0.6, 1] bias no more negative
than −0.055, and the null check passed. These lines were written into the
proposal before Stage 0 was built and were not changed afterwards.

### 11.4 Stage 1: two minimal estimators, one null test

The reference families were fitted on family-side material only: a
background family from 6,000 fit-partition windows (pooled median
spectrum shape, per-site median shapes for the six family sites, and the
top four principal components of window log-shape variation), and passage
prototypes from 764 vessel minutes and 61 ambient minutes of family-side
clips, converted fresh without variation.

**S1, the temporal envelope**, takes per band a low quantile of the
window's minute energies, scaled by the number of observed minutes and
corrected by a per-band factor fitted on the calibration corpus. It was
already weak in family before any held-out test (**Measured**): across
the quantile grid {0.05, …, 0.35} the calibration RMSE fell only to
0.3286, at the best level q = 0.35, which sits at the grid edge, a
calibration score of 0.6714 on its own training side.

**S2, the spectral family decomposition**, fits each window jointly
against the background family and the passage prototypes and reads the
share from the passage component's energy.

The null check, run first as predeclared, on the 550 pure-background
windows from the held-out sites (true share exactly zero; pass requires
mean at most 0.03 and median at most 0.02):

- **S1: mean estimated share 0.5747, median 0.6285. FAIL.**
- **S2: mean estimated share 0.0999, median 0.0920. FAIL.**

Context, **Measured**: the frozen Phase 5 regressor also fails this null
bar on the same 550 windows (mean 0.0471, median 0.0340, against its
in-family true-zero readings of 0.014–0.018), but by a small multiple of
the line, where S2 fails by about a factor of three on the mean and S1 by
about a factor of nineteen. The null bar is strict by design; the
decomposition estimators do not come close to it.

### 11.5 The implementation incident, reported before the verdict

The first null run produced an apparent S1 pass, mean 0.0132. That number
was an artefact: the prediction code summed a cell-level mask where a
minute count was required, inflating the background estimate about
25-fold and driving predictions to near zero on pure backgrounds. The
defect was caught on review of the code path before any reliance was
placed on the number; the code was fixed, the stale prediction file was
deleted, and a smoke test on the fixed code showed the opposite
behaviour. Every Stage 1 number in this report comes from the fixed code.
The incident is recorded because the first figure existed and was wrong;
it plays no part in any verdict, and documenting it makes the evidence
more credible, not less.

### 11.6 Why there are no further Stage 1 scores

Under the binding order, Stage 1 ended at the null check. Neither
estimator was run on the main set, and no I1 overall score, unseen-site
score or share-band bias was produced for either estimator, because the
predeclared sequence does not reach those numbers once the null fails.
No thresholds were adjusted, no variant was tried, and none will be under
this authorisation. Any account of Step 3 that quotes an overall score
for S1 or S2 would be manufacturing a comparison from results the
protocol deliberately did not produce; there are none.

### 11.7 Reading of the failures

**S1's mechanism (Supported).** The per-band correction is fitted on
calibration windows in which events are present and typically occupy most
of a window's minutes, so the fitted envelope is event-contaminated and
the correction factor learns to discount it. On a pure background there
is no contamination to discount, the background is under-restored, and
the missing background energy is read as vessel share — at more than half
the window on average. The failure is structural to fitting an envelope
correction under contamination, not a matter of the quantile level.

**S2's mechanism (Supported).** The background family is fitted on family
sites. At the held-out sites it cannot express the local background
character closely enough, and the joint fit attributes the unexplained
background energy to a passage component: about a tenth of pure
background energy at Bird Island and Fish Hoek is read as an added
passage that is not there. This is the same low-share hallucination the
frozen programme measured in the gradient-boosted regressor, reproduced
by an unrelated mechanism on independently built material.

**Joint reading (Supported).** Three unrelated estimation mechanisms —
the frozen regressor, a temporal envelope, and a spectral family
decomposition — all over-read share on pure backgrounds from unseen
sites, under the independent synthetic validation as under the frozen
one. The persistent estimation difficulty, under every model, feature and
variation strategy this programme tested, sits in unseen-site background
characterisation. That is a statement about the strategies tested.
Whether the 15 × 25 summaries make the decomposition fundamentally
unidentifiable remains **Unresolved**, as it was in the frozen record,
and nothing in Step 3 proves it either way.

### 11.8 Closure

The promise line was not met, so no Stage 2 recommendation arose from
this work, and none is made here. On 18 November 2026 the closure
decision was recorded: **Step 3 is closed under this authorisation; the
physical energy decomposition hypothesis, in its two minimal forms, is
Rejected for this programme; Stage 2 is not authorised.** A more
complicated hybrid would be a new hypothesis, requiring a separate
written proposal and a separate approval — not a continuation of this
one. The yield of the two stages is twofold: a verified independent
synthetic validation now exists and can be reused if further work is ever
authorised, and a clean negative result answers the question the
authorisation was granted to ask. Estimating the physical background
contribution by either minimal route does not clear the first honesty
screen on held-out sites; decomposition, in these forms, does not offer
what the regression approach could not.

---

## 12. Conclusions: one closed programme

The programme's governing decisions, as they now stand:

- **Phase 0–8B:** the frozen evidence and calibration record — the
  conversion fault found and fixed, the site-character correction
  measured and its transferability bounded, the gate not met.
- **Step 3:** formally closed. S1 and S2 were rejected on the predeclared
  null test, and the work stopped at that test by design.
- **Stage 2:** not authorised.
- **Any future hybrid:** a new hypothesis, requiring a written proposal
  and separate approval.
- **Submission:** no candidate, no submission, no slot spent. The one
  slot ever used remains Submission 1 of 16 November 2026.

Across every approach the programme tested — direct regression in many
configurations, background matching, background-relative features,
site-character calibration, a temporal envelope and a spectral family
decomposition — one difficulty persisted: **unseen-site background
characterisation**. Natural background character at a site the estimator
was not fitted on is read as added vessel energy, and no strategy tested
removed that over-reading at its root; the best of them (Phase 7's
character training, within the regression family) corrected it partially
and at a measured price elsewhere. Step 3's contribution to that picture
is that the failure is not an artefact of the regression formulation: a
physically motivated alternative, independently constructed, met it
again through a different mechanism.

What remains unresolved is stated as plainly as what was established.
Whether a more expressive estimator could separate background and passage
energy from the 15 × 25 summaries is **Unresolved**; Step 3 tested two
minimal forms, not the space of all forms. Whether the unseen-site
difficulty transfers to the real test windows is **Unresolved**; the real
windows have no ground-truth labels outside the organisers' process, and
every validation in this programme is synthetic — the frozen one by
construction, the Step 3 one independently implemented but built, by the
qualification recorded in §11.1, on the same public sources and the same
energy-additive assumptions. And the submission gate was never met by
anything: the development minimum of 0.96 was not reached in any phase,
and the approximately 0.998 submission standard was never approached.

The research phase is closed. What the programme leaves behind is a
consolidated, traceable record — the frozen Phase 0–8B archive, the
Step 3 working record and its closure — from which any future work would
have to start from evidence rather than from memory, and this write-up,
which is the deliverable the record supports.

---

## Appendix — provenance and compliance

- **Records.** Every number in this report traces to a phase report:
  `RESULTS_sim_v1.md`, `RESULTS_sim_v2.md`, `RESULTS_phase3_diag.md`,
  `RESULTS_sim_v3.md`, `RESULTS_conversion.md`, `RESULTS_phase6_model.md`,
  `RESULTS_phase7_calibration.md` (with its Phase 8A addendum),
  `RESULTS_phase8_exclusion.md`, `RESULTS_phase8b_graded.md` (with Godson's
  review-qualifications addendum), and `KOLESHJR_AUDIT.md`. The Step 3
  section (§11) traces to `PROPOSAL_step3_decomposition.md` (with the
  decision addendum recording the Option A approval and its boundaries),
  `RESULTS_step3_stage01.md` and `STEP3_CLOSURE.md`; its working artefacts
  are inventoried at the end of the results record and live under `step3/`,
  outside the frozen archive. The frozen
  archive and `ARCHIVE_MANIFEST.md` record, for each reported score, the
  model record file, configuration, seed set and data construction behind
  it, with SHA-256 checksums for all 242 archived files (verified at
  freeze: 242 of 242 byte-identical to their sources).
- **Data.** Backgrounds: the competition's released South African
  acoustics (1,715,779 records, ten sites). Passages: VTUAD v1.3.2,
  CC BY 4.0, Zenodo DOI 10.5281/zenodo.23037895, upstream Ocean Networks
  Canada (Oceans 3.0) — the source the organisers name in the starter
  notebook. No other external data was used; one external expansion claim
  was audited and excluded (§8).
- **Compliance.** Open-source tools only; no AutoML; manual
  hyperparameters throughout. Test inputs used for inference and fixed
  per-window transformations only; the single perturbation study (Phase 7)
  ran under predeclared restrictions and modified no model. Natural vessel
  traffic in a background was never counted toward any label. Submissions:
  one of 200 slots used (Submission 1, simulator v2, 2026-11-16, public
  score 0.687868143); no candidate file was produced by any later phase,
  because no result met the gate that would have justified one. Step 3
  likewise produced no candidate and spent no slot, and ran no inference
  on the real test set; all of its material is validation material built
  for the purpose.
- **Editorial corrections (2026-11-17, on Godson's review).** After the
  freeze, a review pass tightened the claims in this write-up and in
  `ARCHIVE_MANIFEST.md` so that each says exactly what the evidence
  establishes: the construction-ceiling passage no longer claims the
  entire real-world gap is estimation error; the identifiability wording
  in the Phase 6 discussion and in the conclusion is qualified as a
  persistent estimation difficulty under the strategies tested; the
  programme is named the Phase 0–8B research programme consistently; the
  feature-set description distinguishes the 549-feature base model from
  the 602- and 605-feature experiments; the public leaderboard score is
  qualified as public-subset performance; the cross-check comparison
  reads "generally" rather than an absolute, with the Phase 5 exception
  (cross-check B, site 0.9051 against the main model's 0.9030) recorded
  in the manifest; and the archive is described as a frozen, traceable
  research record rather than a self-contained reproduction bundle. The
  manifest also now documents the byte-identical Phase 7 model records
  (`phase7_full.json` and `phase7_full_D.json`). No number, result or
  conclusion changed, and the frozen archive itself was not modified.
- **Consolidation provenance (2026-11-18).** This consolidated write-up
  rests on: the Phase 0–8B write-up (`WRITEUP_phase0_8b.md`), whose body
  is carried over unchanged in substance, numbers and settled wording;
  the phase results files listed under Records above; `ARCHIVE_MANIFEST.md`
  and the frozen archive it indexes (242 files, checksums verified at
  freeze); and, for Step 3, `PROPOSAL_step3_decomposition.md`,
  `RESULTS_step3_stage01.md` and `STEP3_CLOSURE.md`. The consolidation
  was prepared after the research phase was closed and the write-up made
  the priority; it integrates the Step 3 record into the programme's
  account without altering any Phase 0–8B result, and the source files
  remain on disk alongside this one.
