# Step 3 Proposal: Physical Energy Decomposition

Competition: Underwater Vessel Noise Challenge by the French Embassy in
South Africa (Zindi). Entrant: Godson (Godson22), solo.
Date of this proposal: 2026-11-18.
Status: a design and validation proposal only. It authorises nothing by
itself. No experiment described here has been run, no code for it has been
written, and the frozen Phase 0–8B record is untouched by it.
Decision requested: see Section 1.

Classification discipline, as throughout this project: every substantive
claim carries one of four labels. Measured: directly demonstrated by an
experiment in the frozen record. Supported: consistent with multiple
independent results, short of direct demonstration. Unresolved: plausible
but not established. Rejected: contradicted by the available evidence.

---

## 1. Purpose and decision requested

The Phase 0–8B research programme ended on 2026-11-17 by decision, not by
exhaustion of ideas, with the gate unmet and the record frozen. The
write-up (WRITEUP_phase0_8b.md, Section 10) recorded exactly one
genuinely different direction as an outlook: physical energy
decomposition. Godson named that direction himself on 2026-11-17, and was
explicit that naming it was not permission to start it: it would need a
designed proposal and independent validation before any implementation.
This document is that proposal. On 2026-11-18, after a live check of the
leaderboard (top score 0.994250562, held by melihk; Godson22 at rank 624
of 640 on the strength of the single early submission), Godson authorised
the writing of this proposal. Implementation remains unauthorised.

The hypothesis in one paragraph. Every model in Phases 0–8B predicts the
share directly: a gradient-boosted regressor maps 549 summary features of
a window to share_added_vessel, trained on windows manufactured by our
simulator. The decomposition hypothesis is that the share should instead
be derived: estimate, for each test window separately, how much of the
window's observed energy is background and how much is the added passage,
using the physical and statistical structure of those two components, and
compute the share from the two estimates by the same arithmetic the
organisers' construction uses. The share is not learned; it is computed
from a split that is estimated.

What Godson is being asked to decide is narrow:

- Option A: approve Stage 0 and Stage 1 only (Section 7). Stage 0 builds
  the independent validation construction; Stage 1 builds the minimal
  decomposition estimator and screens it against predeclared kill
  criteria. The pair ends with a written report against those criteria,
  whatever the outcome. Nothing beyond Stage 1 is approved by Option A,
  and Stage 2 would come back for a separate decision informed by the
  Stage 1 report.
- Option B: decline. The frozen record and the write-up stand as the
  project's final deliverables, and this proposal is filed with them as a
  considered and rejected option.

The gate discipline that governs any work under this proposal is the
standing one, quoted here because it binds every later section:

- Development minimum: 0.96 on the frozen validation protocol AND both
  independent cross-checks (A and B), across all four holdouts (random,
  site, period, long), reported separately.
- Submission standard: approximately 0.998 under genuinely independent
  validation.
- Never lower the gate. Never change validation criteria after seeing
  results. Never spend a submission slot on a failure. No Zindi
  submission without Godson's explicit approval. (Slots used: 1 of 200;
  199 remain.)
- Real-window prediction distributions are diagnostics only: never proof
  of accuracy, never a selection criterion, on their mean or any other
  summary.

The competition closes 2026-12-13. That date is not a reason to hurry the
science; it is a boundary condition recorded in Section 7, because a Step
3 that cannot complete its validation well before the close is not worth
starting.

---

## 2. What the top score and the oracle ceiling establish, and what they do not

Three numbers frame the case for trying a different formulation. Each is
stated with the qualification the record attaches to it.

The top public score, 0.994250562 (melihk, live-checked 2026-11-18), is a
score on the public subset, which reflects about 30% of the test windows.
It establishes that a score in the 0.994 class has been reported on that
subset. It does not establish equivalent performance on the private
subset, and it does not establish anything about how the score was
achieved. Its evidential use inside this project has always been narrower
and stronger than leaderboard watching: the Phase 5 target arithmetic
audit showed that the observed-cells sum is the only audited label
definition under which a 0.994-class score is arithmetically possible at
all (the mean-based, band-restricted and amplitude-domain alternatives
cap at 0.78 to 0.94 and are Rejected, Measured). The top score is
therefore evidence about the label and about recoverability, not evidence
about any particular method, including this proposal's.

The oracle ceiling under our simulator mirror is 0.9981 for the frozen
bandext_maskfix construction (and 0.9983 for the Phase 4 construction
against its stored labels). The corrected wording from Godson's
2026-11-17 review governs here, and this proposal adopts it verbatim in
substance: the ceiling demonstrates recoverability under our mirror of
the organisers' process; within that construction, the remaining
model-to-oracle gap is estimation error; whether residual
simulator-to-real mismatch also contributes to real-test error remains
Unresolved, because the real test windows have no available ground-truth
labels. A further oracle fact matters specifically for decomposition:
the Phase 5 audit's oracle estimator is a true-background residual
estimator, meaning it computes the share by subtracting the known
background's energy from the window's total energy, cell by cell, over
the observed cells. With the background known exactly, that arithmetic
scores at the construction ceiling (Measured). Decomposition is, in
effect, the proposal to replace the oracle's known background with an
estimated one and keep the arithmetic.

What neither number establishes: that the background of a real test
window can be estimated well enough, from the 15 by 25 summary alone, for
the derived share to beat 0.9488, let alone reach the gate. That is the
open question this proposal exists to test cheaply, and Section 5 states
the reasons it might have a negative answer with the same prominence as
Section 4's reasons for optimism.

One more framing fact, because it disciplines expectations. Our only
submission (simulator v2, 2026-11-16) scored 0.687868143 and sits at rank
624 of 640. It predates the label fix, the conversion fix and every
calibration result. The frozen programme's best models were never
submitted, because none met the gate. The leaderboard position is
therefore not a measure of the frozen record's quality, and improving the
leaderboard position is not this proposal's purpose. The purpose is to
test whether a different formulation can meet the development gate that
the direct formulation could not.

---

## 3. The decomposition approach in concrete terms

### 3.1 The estimand

For one test window, let the observed cells be the (minute, band) pairs
that are not missing. Missingness in this task is whole-minute only
(Measured, Phase 0 audit): a window has between 12 and 15 observed
minutes, each with all 25 bands. Write E(m, b) for the linear energy in
cell (m, b), recovered from the summary's decibel values. In the
organisers' construction, the mixture is additive in the linear energy
domain: each observed cell's energy is the background's contribution plus
the added passage's contribution, up to the 0.1 dB quantisation of the
export. Define, over observed cells only:

- E_tot, the window's total observed energy: the sum of E(m, b).
- E_bg, the background's contribution to those same cells.
- E_add, the added passage's contribution: E_tot minus E_bg.

The settled label (Phase 5 audit, Section 3.5 below) is the realised
share over observed cells: share = E_add / E_tot = 1 − E_bg / E_tot.

The estimand of the decomposition is therefore a single quantity per
window: E_bg, the background's share of the observed energy. Everything
else is arithmetic. This reduction is the proposal's core structural
claim (Supported, by the audit identity above): the regression problem
the GBM solves, mapping 549 features to a share, is replaced by an
estimation problem with a known exact link between the estimated quantity
and the target. Error in the share is a determined function of error in
E_bg: overestimating the background by a fraction f of E_tot
underestimates the share by exactly f. There is no learned mapping left
to miscalibrate; there is only the quality of one estimate.

### 3.2 What is estimated from the window itself, and what comes from reference data

Estimated from the window itself (per window, independently of every
other test window):

- The background's spectral shape and level for that window, allowed to
  drift slowly across the window's minutes.
- The added passage's per-minute amplitude trajectory and its total
  energy, as the residual structure the background estimate does not
  explain.
- The split itself, cell by cell or in aggregate, from which E_bg and
  the share follow.

Taken from reference data (all training-side, all frozen before any test
window is touched):

- The background family: the statistical structure of real background
  spectra, built from the 87,653 screened 15-minute windows of the
  released South African acoustics (Richards Bay sentinel contamination
  screened, as in every phase since Phase 1). Concretely at design
  level: per-site and pooled median spectra, a small set of spectral
  shape components spanning the variation the release actually exhibits
  (inter-site spectral differences in the release have an RMS median of
  14.2 dB, Measured in Phase 5, so the family must be wide), and the
  typical minute-to-minute drift behaviour of a real background.
- The passage family: the spectral and temporal structure of added
  passages, built from the 542 VTUAD v1.3.2 clips our harvest converts,
  in the bandext representation (Section 3.5). Concretely: passage
  spectral shapes (which are low-frequency dominated, with band 0 and
  band 1 carrying most energy), the range of passage spectral variation
  across clips, and passage temporal envelopes (events that, in the
  organisers' construction as mirrored by our simulator, span most of a
  window: realised mean duration about 10 minutes, 47.9% of events at
  least 12 minutes, Measured in the Phase 2 construction).
- The null behaviour: the distribution of the estimator's output on
  windows known to contain no addition, which is a property of the
  estimator and the background family, measured on training-side and
  validation-side pure backgrounds, never on test windows.

Nothing is estimated from the test set as a whole. No statistic is pooled
across test windows. Each test window is processed by a fixed procedure
whose every parameter was fixed from training-side data in advance. That
is what keeps the method inside the rules (Section 8) and inside the
project's standing discipline.

### 3.3 Candidate estimation strategies (design level, no code)

Three strategies are specified, in increasing order of ambition. Stage 1
(Section 7) implements only the first two in minimal form; the third is
the Stage 2 candidate if the screen justifies it.

Strategy S1, temporal envelope estimation. For each band separately,
model the window's minute series as a slowly varying background level
plus a non-negative passage contribution. Estimate the background level
per band from the lower part of the minute distribution (a robust low
quantile or a fitted lower envelope), with a contamination correction:
because passages occupy most minutes in most windows, the low quantile
of the mixture is not the background, and the correction factor must be
learned from training-side mixtures where the truth is known. Aggregate
the per-band background estimates over observed minutes to E_bg.
Strength: simple, fully per-window, and its failure modes are measurable
on pure backgrounds before anything else is built. Weakness: it uses
almost no spectral structure, and its naive ancestor already failed in
this project (Section 5).

Strategy S2, spectral family decomposition. Represent each observed
minute's 25-band energy spectrum as a non-negative combination of a
background component and a passage component. The background component
for a minute is drawn from the background family: a spectrum constrained
to lie near the family's shape space, with its level and shape
coefficients varying slowly across the window's minutes. The passage
component is drawn from the passage family, with a per-minute amplitude
that is non-negative and follows a single-event envelope (rising,
sustaining, falling, or a sub-window thereof, since the organisers vary
time trajectories, including reversal). Fit the whole window jointly by
constrained least squares in the linear energy domain, with the
smoothness and envelope constraints as the regularisation. E_bg is the
sum of the fitted background components over observed cells. Strength:
it uses the strongest physical asymmetry in the problem, that background
spectra and passage spectra have different, separately measurable shape
families, and it estimates the background for minutes that contain
passage energy rather than only from minutes that do not. Weakness: the
families overlap exactly where it matters most, in the low bands
(Section 5).

Strategy S3, anchored hybrid. Use S1's temporal analysis to identify the
window's least-contaminated minutes; anchor the background spectrum on
those minutes; extrapolate the background across all minutes under the
slow-drift model; then run one S2-style refinement pass in which the
passage family's shapes weight the residual attribution. S3 is the
closest analogue of the oracle estimator's form (a residual against an
estimated background), with the anchor minutes playing the role the true
background plays for the oracle. It is sketched here and designed in
detail only if Stage 1's screen earns it.

In all three strategies, the final share is computed by the settled
arithmetic of Section 3.5, clipped to [0, 1], with the companion column
score_added_vessel predicted as the square root of the share, the fixed
monotone companion used throughout the frozen programme.

### 3.4 Missing minutes

Missing minutes are handled by exclusion, not imputation, at the level
of the estimand: the share sums run over observed cells only, so a
missing minute contributes to neither E_tot nor E_bg nor E_add. This is
not a choice but a consequence of the settled label definition
(Section 3.5). Internally, an estimator may need a background level for a
missing minute's neighbours (for drift modelling across a gap at minute
1, which is missing in 41.6% of test windows, Measured); such internal
interpolation is permitted because it never enters the sums. The maskfix
finding carries over as a design constraint on validation: any synthetic
evaluation of the estimator must apply the measured real mask form
(complete 0.517, minute 1 only 0.413, minute 15 only 0.065, both edges
0.005), because estimator behaviour at window edges, where the passage
envelope is least observed, is part of what is being tested.

### 3.5 Constraints inherited from Phase 5 (settled, not reopened)

Two Phase 5 results constrain this design as facts, and the proposal
does not reopen either.

The label arithmetic. The label is the realised share over observed
cells, computed as a sum of energies (Phase 5 audit: the only audited
definition consistent with the real top score and the label moments;
rounding order immaterial at RMSE 0.0008). The decomposition's derived
share must use exactly this arithmetic, or it would be estimating a
different quantity from the one scored. This is also why the estimand is
defined over observed cells in Section 3.1.

The conversion. The test summaries live in the organisers'
representation, in which band 0 integrates from 0 Hz and band 24 to
Nyquist; the starter notebook's deci-decade edge clipping was the
principal source of the original transfer failure (Measured, Phase 5:
nine of ten conversion variants left the real-window mean at 0.319 to
0.336; bandext moved it to 0.2042). Every reference family in
Section 3.2, background and passage alike, must therefore be built in
the bandext representation. A passage family built with the starter
conversion would re-import the single largest measured fault in the
project's history into the new method's foundations. The released
background windows are already in the organisers' representation and
need no conversion; the passage side must be converted with bandext, as
the frozen harvest already is.

### 3.6 What decomposition is not

It is not background matching. Phase 3 tested whether a test window's
background could be found verbatim in the release and subtracted; the
hypothesis was refuted on real data (zero exact-minute fingerprints in
60 sampled windows, across a median of 5,829 candidates per window) and
the direction was closed permanently by Godson's ruling, including its
gated and hybrid variants. Decomposition does not look up any test
window's background anywhere. Its background family is distributional
structure learned from training-side data, of the same kind the frozen
programme already uses; the per-window background is estimated from the
window itself. The distinction is the same one Phase 6 drew when it
proposed per-window background modelling in feature space: the reference
is the window itself and the training-side families, never a release
window matched to the test window.

It is also not another feature set for the existing regressor. Family D
in Phase 7 already fed within-window background deviations to the GBM as
features; Section 5 records what that did and did not do. Decomposition
differs in kind: the background estimate is the answer's substance, not
an input to a learned mapping, and no regressor stands between the split
and the share.

---

## 4. Why it might beat 0.9488

Each claimed advantage below is tied to a measured failure in the frozen
record. The plateau to beat is precise: best holdout 0.9488 (Phase 7
character configuration, period holdout, main model), best site holdout
0.9250 (Phase 7 character configuration with family D), against a
development gate of 0.96 on the main model and both cross-checks across
all four holdouts. No claim in this section asserts that decomposition
will clear that gate. The claims assert that specific measured failure
mechanisms of the direct approach are absent, or attackable, in the
decomposition formulation. Whether that is enough is what Stage 1 tests.

Claim 1. The site-holdout failure is a learned-calibration failure, and
decomposition has no learned cross-site calibration to fail. Phase 6
localised the site failure exactly: at unseen sites the natural band-0
level sits higher than the training range, and the model's learned
mapping from band-0 level to added share over-fires, producing the +0.065
bias at true share zero (RMSE 0.090, 21.4% of site squared error;
Measured). The failure lives in a mapping fitted on other sites.
Decomposition estimates the background from the window's own minutes, so
an unseen site's character enters the estimate itself rather than being
judged against training sites' levels. Bird Island's band-0-heavy
character (band-0 share 1.70 times the all-site median, Measured) would
be absorbed into that window's background estimate by construction,
instead of being read as vessel energy by a mapping that had to be
taught, through training-distribution surgery, not to. The mechanism of
the failure is Measured; that per-window estimation removes it is
Supported, not Measured, because the estimator has never been run. This
is the proposal's strongest claim, and it is aimed at the largest single
measured error source in the record: the zero and (0, 0.05] bands
together carried 38.3% of the site holdout's squared error in Phase 6.

Claim 2. The low-share correction and the high-share penalty are coupled
in the direct approach and decoupled in decomposition. Character
calibration (Phases 7 and 8B) bought the zero-share correction at a
measured price: the (0.6, 1] bias deepened from the untreated baseline's
minus 0.051 to about minus 0.070 at full strength, and the Phase 8B
graded experiment mapped the exchange as a smooth dose-response in
character strength, with quarter strength recovering only part of the
penalty (minus 0.056 and minus 0.064 on the two main configurations).
The coupling is structural in a regressor: one mapping must serve both
ends of the share range, and teaching it that band-0-heavy windows are
usually natural teaches it to discount genuine large additions on such
windows too. A decomposition estimator has no such coupling. Its share
is a ratio of two estimated energies; improving the background estimate
at low shares does not mechanically shrink the estimate at high shares,
because nothing in the arithmetic prefers small shares. The related
seen-holdout symptom, textbook shrinkage (over-reading below 0.05,
under-reading above 0.3, with the mid and high bands carrying about 58%
of seen-holdout squared error), is a property of mean-squared-error
regression toward a prior (Measured across every model class in Phase
6). The decomposition estimator is not trained by minimising share
error and has no prior mean to shrink toward. Supported: the coupling
and the shrinkage are Measured properties of the direct approach; their
absence in decomposition follows from the formulation, but the size of
the benefit is Unresolved until Stage 1 measures it.

Claim 3. The plateau was invariant to everything except formulation.
Phase 6 tested, on identical windows, three further gradient-boosting
implementations and an MLP (all ties within noise), a Huber loss (a
random-for-site trade, Rejected), doubled training volume (plus 0.003 to
0.004 on seen holdouts, nothing on site), and broadened variation axes
(synthetic wash, worse real diagnostic). Phase 7's levers were
site-directed and moved the site holdout only. The seen-holdout ceiling
has sat at about 0.948 since Phase 5. When model class, loss, volume,
breadth and calibration strategy have each been varied and the ceiling
does not move, the remaining untested axis of comparable size is the
formulation itself: what quantity the method estimates. Decomposition is
that axis. The invariance is Measured; the inference that a formulation
change is the most promising remaining direction is Supported; that this
particular formulation succeeds is Unresolved.

Claim 4. The arithmetic link is exact, so estimation effort is not
diluted by a learned link. With the true background, the residual
estimator scores at the construction ceiling (0.9983 against the Phase 4
labels, Measured in the Phase 5 audit). The direct approach's gap, by
contrast, is distributed across a 549-feature mapping whose failure modes
had to be discovered one phase at a time. In decomposition, every
improvement in background estimation translates into share accuracy
through the identity of Section 3.1, and every error can be attributed:
a wrong share is a wrong split, measurable per band and per minute
against known truth in validation. This attributability is also a
practical advantage for a short programme: Stage 1 can fail informatively,
locating the error in the background family, the passage family, or the
split optimisation, rather than in an opaque mapping. Measured for the
identity; Supported for the practical benefit.

Claim 5. The total-energy budget is a hard constraint the direct approach
uses only softly. A feature regressor sees energies as inputs and can
return shares inconsistent with any possible split of the window it was
given; nothing in its training enforces that background plus addition
equals the observed total, cell by cell. Decomposition enforces that
identity by construction, and additionally enforces non-negativity of
both components and the passage envelope's temporal coherence. Whether
these constraints carry enough information to matter at the gate's scale
is Unresolved; that they are real information the GBM does not use is
Supported by the Phase 6 finding that the remaining error sits in the
information the features carry, not in the optimiser.

---

## 5. Why it might fail

The risks are stated at the same strength as the hopes. Several of them
are not hypothetical: they are measured failures of this idea's near
relatives inside the frozen record.

Risk 1. The split may not be identifiable from the summary alone. A
window offers at most 375 observed numbers. The decomposition asks them
to determine a background spectrum that varies by site (inter-site
differences of 14.2 dB RMS, Measured) and over time, plus a passage
contribution whose spectrum overlaps the background's most variable
region. Band 0 is both the dominant vessel channel (32% of the frozen
model's gain; grouped permutation cost 0.194 on the random holdout,
Measured) and site-confounded (two band-0 level features alone identify
a background's site at 0.42 against 0.16 chance, Measured). Many
different splits reproduce the same observed window within the 0.1 dB
quantisation, and the choice among them is made by the reference
families, which is to say by priors. A wrong prior does not produce a
random error; it reproduces a systematic bias in a new form, and the
record contains a measured example of exactly that pattern in the
character-calibration trade of Phases 7 and 8B. The standing
qualification from Godson's review applies with full force: what the
frozen programme established is a persistent estimation difficulty under
every model, feature and variation strategy tested. Decomposition is one
more strategy, not an exemption from that finding. Classification:
Unresolved, and it is the proposal's principal scientific risk.

Risk 2. There may be no clean minutes to anchor on. The naive version of
within-window estimation is the B1 low-band excess statistic of Phase 0,
and it failed its null check decisively: on 2,100 ambient-only windows
whose true share is zero by construction, it read a median of 0.344 and
a mean of 0.398 (Measured). Ambient low-band fluctuation, including
genuine vessel traffic that is not a controlled addition, produces most
of its output. The structural reason persists in any temporal strategy:
the organisers' additions span most of a window (realised mean duration
about 10 of 15 minutes; 47.9% of events at least 12 minutes), so in the
majority of windows a majority of minutes contain passage energy, and
the lower envelope of the mixture is a biased estimate of the
background. S1's contamination correction and S2's family constraints
exist precisely because of this risk, but their adequacy is Unresolved.
Any decomposition variant that has not passed a null check on pure
backgrounds from held-out sites is, by this project's own Phase 0 lesson,
not yet an estimator. Stage 1's kill criteria therefore put the null
check first (Section 7).

Risk 3. Self-referencing has a measured failure mode in this project.
Phase 7's family D referenced each window against its own quiet minutes
and did not move the zero-band bias at all (plus 0.064 against plus 0.064
for the baseline on identical windows, Measured): on Bird Island
backgrounds the whole window, quiet minutes included, carries the site
character, so the self-estimate absorbs the character it was meant to
expose. Decomposition is not family D (Section 3.6), but any variant
that estimates the background primarily from the window's own quiet
minutes inherits this risk in arithmetic form: the minutes least
contaminated by the passage are also the minutes most purely expressive
of the site's character, and if the passage family's shapes cannot
separate character from addition in the remaining minutes, the split
will misattribute in the same direction the GBM did. Supported as a
risk, from a Measured precedent.

Risk 4. The added component is not clean vessel energy. The organisers'
own caveat, recorded in PLAN.md from the starter notebook: passage
references may themselves include ambient sound from their original
recording, so the label counts the passage recording's whole energy,
ambient floor included. A decomposition whose passage family models a
clean vessel spectrum will attribute the passage's ambient floor to the
background, biasing shares downward, plausibly worst at low shares where
the floor is a large fraction of the addition. The passage family must
therefore model the recorded passage as a whole (vessel plus its own
ambient), which the harvest-side clips do represent, since they are the
recordings themselves. Even so, the organisers varied the passages'
relative channel and energy before combination, and the family's
fidelity to the varied additions is Unresolved.

Risk 5. Simulator-to-real mismatch does not retire just because the
formulation changes. The reference families are built from the same
harvest side as every previous phase: the released backgrounds and the
VTUAD clips under the bandext conversion. Phase 5 proved the conversion
axis alone was worth 0.133 of real-window mean movement, and the
corrected wording stands: whether residual mismatch also contributes to
real-test error is Unresolved. If the organisers' passage processing
differs from bandext in some further unmeasured way, the passage family
is wrong in a direction no amount of validation on our own constructions
can see. This is the deepest reason Section 6 refuses to let our own
simulator arbitrate the idea, and it is also why even a fully successful
Step 3 would end with the submission standard judged under genuinely
independent validation, not under our mirror.

Risk 6. The effort may reproduce the plateau by a more expensive route.
If the information in the 15 by 25 summaries supports only about 0.95
under any estimator, decomposition will find the same ceiling with
better explanations. The frozen record cannot rule this out: the
identifiability question was left open by Godson's own correction pass,
which replaced a stronger claim with the qualified one. Stage 1 is
designed so that this outcome, if it is the outcome, is reached in about
two days of work against predeclared numbers, not after another
programme. Classification: Unresolved.

Risk 7 (context, not evidence). The direction is visible to competitors.
A participant write-up entry on physics-informed linear acoustic energy
separation and total-energy normalisation already exists in the
competition's discussions (noted during the write-up prize check of
2026-11-18). This is recorded as context only. It is not evidence about
our design, which was specified independently from the frozen record; it
does not make the direction more or less likely to work; and matching
another entrant's framing is not a goal. It has one practical
implication Godson may wish to weigh: if the write-up prizes matter, an
energy-separation write-up from this project would no longer be the only
one of its kind, though ours would rest on the Phase 0–8B record, which
no other entrant has.

---

## 6. Independent validation design

This section is the core of the proposal. Its premise is Phase 4's
methodological warning, learned expensively: a model can look better
simply because the world it is tested on is the world it was trained on,
and the frozen simulator is the world that produced the 0.9488 plateau.
If decomposition is validated only on windows manufactured by that
simulator, a good score would show that the estimator agrees with our
mirror, which is the claim already in doubt. The validation must
therefore include a path whose ground truth does not come from our
mixture simulator.

### 6.1 Path I1: a separately implemented construction (primary)

A second construction, built for this validation and for nothing else,
with the following independence properties, each addressing a specific
way the frozen simulator could be wrong in decomposition's favour:

- Separate implementation. The mixing code is written fresh, by a
  separate implementation pass, and does not import the archived
  simulator, converter or label code. The label is computed by the
  settled observed-cells arithmetic, implemented independently from the
  definition in Section 3.5, so that a shared coding error in the label
  cannot manufacture agreement.
- Waveform-side passage handling. Passages are taken from the VTUAD
  recordings and converted by the fresh implementation in the bandext
  representation. Where the frozen simulator mixes pre-converted passage
  summaries in the energy domain, the independent construction applies
  the organisers' stated variations (time trajectory variation including
  reversal and warping, relative channel variation, energy rescaling
  with absolute gain discarded) at the recording level before
  conversion, so the passage summaries it produces are not the frozen
  harvest's summaries replayed.
- Disjoint material. The passage clips used are disjoint from the
  harvest clips the reference families were built on, split by hull and
  by recording date, so no clip, and no vessel, appears on both sides.
  The backgrounds are real windows from the release that the reference
  families' fitting excluded: whole sites and periods held out from
  family fitting, including at least one site whose character the
  families have never summarised, so the site question is asked honestly
  of the estimator and not only of a regressor.
- A different share prior. The drawn shares follow a distribution
  deliberately different from the simulator's prior (for example, a
  uniform draw over [0, 0.9] with a separate exact-zero mass), with
  independent drawn-share assignment throughout, so a method that has
  merely absorbed our prior's shape is exposed rather than rewarded.
- The measured mask form (Section 3.4), applied by the fresh code.

The estimator under test sees only the mixed summaries and its frozen
reference families. The construction's component energies, and hence the
true labels, are withheld from it by the harness.

### 6.2 Path I2: blinded organisers-way set (secondary)

A smaller set built the Phase 3 way: synthetic windows assembled to
mirror the organisers' construction with component identities withheld
from the estimator and from whoever scores the run until scoring is
complete. I2 adds blinding discipline rather than construction
independence; it is a guard against unconscious tuning to I1's quirks,
and it reuses the project's existing habit (Phase 3's withheld-identity
matchers) rather than inventing a new one. I2 is supporting evidence
only; no kill or promise decision rests on it alone.

### 6.3 The frozen protocol remains the gate

Neither I1 nor I2 replaces the standing gate. If, and only if, the
estimator passes Stage 1 on the independent path, Stage 2 evaluates it
under the frozen protocol unchanged: the fixed validation and both
independent cross-checks (in decomposition terms, two estimator variants
whose reference families are fitted on disjoint training-side splits,
mirroring the A and B discipline), all four holdouts reported
separately. The development minimum of 0.96 applies there exactly as
written in Section 1, and the submission standard of approximately 0.998
under genuinely independent validation is judged on the independent path
at full scale, not on the frozen simulator. Nothing in this proposal
changes a validation criterion that already exists; the independent path
is an additional, earlier screen whose failure ends the project before
the frozen protocol is spent on it.

### 6.4 Predeclared promise and failure numbers for the independent path

Declared here, before any Stage 0 or Stage 1 work, and not revisable
after results are seen:

Null check (pure backgrounds from held-out sites, true share zero,
independent construction): the estimator passes if its mean estimated
share is at most 0.03 and its median at most 0.02. Reference points: B1
failed this check at mean 0.398 and median 0.344 (Measured); the frozen
GBM family reads true-zero windows from its own construction at mean
0.014 to 0.018 (Measured). Failure of the null check kills the project
at Stage 1 regardless of any other number.

Promise line (Path I1, main set): score at least 0.96 overall, with the
unseen-site subset at least 0.93, site zero-band bias at most plus 0.030
(the corrected level the character configurations achieved, Measured),
and (0.6, 1] bias no more negative than minus 0.055 (the better of the
two Phase 8B quarter-strength main configurations; the untreated
baseline's minus 0.051 is the level decomposition ultimately has to
match or beat to claim the decoupling of Claim 2). Meeting the promise
line in full is the only route to a Stage 2 recommendation.

Failure line (Path I1, main set): score below 0.93 overall, or zero-band
bias above plus 0.040 (the survival-rule line Phase 8B predeclared for
its own screen), or (0.6, 1] bias more negative than minus 0.070 (the
full-strength character level, which would mean the estimator pays the
very trade it exists to avoid). Any one of these kills the project at
Stage 1.

Between the lines: if the result lands between failure and promise, work
stops and the Stage 1 report goes to Godson with the numbers as measured.
No continuation is automatic, no threshold is renegotiated, and no
substitute experiment is run under this proposal.

---

## 7. Staged plan with kill criteria

The plan is front-loaded with the cheapest decisive test, in keeping
with the way Phases 8A and 8B were each a single bounded experiment with
a predeclared stop. Effort estimates are calibrated on the frozen
programme's pace: Phase 5 (ten conversion variants, two audits, full
protocol) completed in about a day; Phase 6 (five workstreams and a full
protocol) in about a day; Phase 7 (four workstreams, a perturbation test
and two full protocols) in about a day; Phase 8A in a fraction of a day;
Phase 8B in an evening. The estimates below are in working days at that
pace, and they are estimates, not commitments.

Stage 0, validation first (about 1 day). Build Path I1 (Section 6.1):
the fresh mixing implementation, the disjoint hull and date passage
split, the held-out site and period background split, the independent
label implementation, the withheld-truth harness. Verify the harness
before any estimator exists: the harness must reproduce, from component
energies it withholds, labels that agree with a hand-checked sample; and
the frozen bandext_maskfix baseline model, scored through I1 as a
sanity reference, must land in a plausible relationship to its frozen
holdout scores (it is not required to match them; I1 is a different
world, and a wild divergence in either direction is a finding about I1
to be resolved before Stage 1, not a number to be used). Stage 0
deliverable: the I1 construction, its splits, and its verification
record. Kill criterion for Stage 0 itself: if the disjoint material
cannot be assembled (for example, if the passage pool cannot support a
hull-disjoint and date-disjoint split of usable size), the independent
path does not exist, the proposal's core requirement cannot be met, and
the project stops here with that reported.

Stage 1, the minimal estimator and the screen (about 1 day). Implement
S1 and S2 in minimal form (Section 3.3), with reference families fitted
only on the training-side material I1's splits leave available. Run, in
this order: the null check on held-out-site pure backgrounds; then the
I1 main set, reported overall, on the unseen-site subset, and by share
band, against the predeclared lines of Section 6.4. Kill criteria,
exact and complete: the project ends at Stage 1 if the null check fails
(mean above 0.03 or median above 0.02 at true share zero), or if the I1
score is below 0.93 overall, or if the site zero-band bias exceeds plus
0.040, or if the (0.6, 1] bias is more negative than minus 0.070. The
project proceeds to a Stage 2 recommendation only if the full promise
line of Section 6.4 is met. Anything between is a stop-and-report.
Stage 1 deliverable: a results record in the project's format, with
every number classified, including if the verdict is that the idea is
dead.

Stage 2, the full estimator (sketch only, about 2 to 3 days, separately
approved). S3 and refinements of whichever of S1 or S2 survived;
reference families at full scale; the frozen protocol with its A and B
cross-check structure; per-band and per-minute error attribution against
known truth on I1 at full size. The gate that governs Stage 2's outcome
is unchanged: 0.96 on the fixed validation and both cross-checks across
all four holdouts. A Stage 2 result below that gate is reported as not
met, exactly as every phase before it was.

Stage 3, the submission question (sketch only, about 1 day plus
Godson's decision time). Only a Stage 2 result that meets the
development gate raises the submission question at all, and it is raised
to Godson with the full record, including the independent-path numbers
against the approximately 0.998 submission standard. No candidate file
is produced and no slot is spent without his explicit approval; 199 of
200 slots remain, and this proposal contemplates spending at most one,
ever, and only at the standard.

Deadline boundary. The competition closes 2026-12-13, 25 days after
this proposal. Stages 0 and 1 fit comfortably. A full programme through
Stage 3 fits only if Stage 1 passes promptly; if Stage 1 has not
reported by about 2026-12-01, the honest recommendation built into this
plan is to stop, because a rushed Stage 2 against a fixed close would
pressure exactly the gate discipline this project has kept for eight
phases.

---

## 8. Rules compliance check

Checked against the competition rules as verified first-hand in PLAN.md
(Section 1 and Section 3, from the live Info and Data tabs, 2026-11-16).

Test inputs are inference only. The rules prohibit fitting, fine-tuning,
calibration, pseudo-labelling and building training examples from test
inputs; fixed per-window transformations are expressly allowed.
Decomposition as designed is a fixed per-window transformation: every
parameter of the estimator (the background family, the passage family,
the envelope and smoothness constraints, the null-derived constants) is
fitted on training-side data and frozen before test inference, and each
test window is then processed independently, using only its own cells
and the frozen parameters. Estimating a window's background from that
window's own minutes is inference on that window, of the same kind as
the fixed transforms every phase has applied. Two design elements are
expressly excluded to keep this clean, and the exclusion is part of the
design, not an afterthought: first, no statistic may be pooled or fitted
across test windows (a test-set-wide background level, a clustering of
test windows to refine the families, or any threshold chosen by looking
at test outputs would each be calibration on test inputs, and none is
permitted); second, method selection among S1, S2 and S3, and all tuning
of their constants, is done on training-side and independent-validation
data only, with real-window prediction distributions used, as
throughout this project, as diagnostics that are never a selection
criterion. One related diagnostic from Phase 6's recommendations,
adding a synthetic passage of known share to copies of real test windows
and checking prediction increments, is not part of this proposal's
validation design: Phase 6 itself recorded that it touches test inputs
beyond the inference-only use made of them so far and needs Godson's
separate ruling. If Godson wishes that diagnostic run, it is a separate
decision; nothing here depends on it.

One honest flag. If, during implementation, any design element were
found that cannot be expressed as a per-window procedure with frozen
parameters (for example, a background level that only exists as a
property of the whole test set), that element would fall outside the
inference-only allowance as this project has conservatively read it. The
correct course, stated in advance, is to drop the element or seek a
ruling from the organisers before using it, never to use it and argue
afterwards. No element of the present design has that character, but the
flag is recorded because the boundary is the one place a decomposition
method could drift without noticing.

Open source only, no AutoML. The design uses the project's existing
open-source stack (Python, NumPy, SciPy; the constrained fitting is
standard least squares with non-negativity and smoothness constraints).
No AutoML is involved at any stage; every constant is set by hand and
recorded.

External public data, declared. The method uses exactly the two sources
the frozen programme used and declared: the competition's released South
African acoustics (1,715,779 records, ten sites) and VTUAD v1.3.2
(CC BY 4.0, Zenodo DOI 10.5281/zenodo.23037895, upstream Ocean Networks
Canada), the passage source the organisers themselves name in the
starter notebook. Both are permitted under the rules, which expressly
allow participants to create their own labelled synthetic data from
permitted data; Path I1's mixtures are such synthetic data, built for
validation. No new external source is introduced by this proposal. If a
future stage proposed one, it would be declared with source, version,
licence and preprocessing before use, and the Koleshjr precedent
(audited and excluded in Phase 7) is the standing example of how
external claims are treated here.

Reproducibility and code review. The competition requires the top 10 on
the private leaderboard to submit runnable code reproducing their
predictions within 48 hours of the close. A decomposition pipeline is,
if anything, easier to keep review-ready than the frozen programme: it
has no trained booster artefacts, its parameters are the two reference
families and a short list of constants, and its per-window procedure is
deterministic given fixed seeds for family fitting. Any Stage 2 work
would keep the project's existing habits (versioned generation scripts,
recorded seeds, declared splits) so that a code review could be met from
the record.

---

## 9. What stays frozen regardless

Approval of any part of this proposal changes none of the following.

- The archive. The frozen Phase 0–8B record (242 files, checksums
  verified 242 of 242 at freeze) is not modified, regenerated or
  re-opened. Step 3 work, if approved, lives in new files outside the
  archive, exactly as this proposal does.
- The gate. The 0.96 development minimum on the frozen validation and
  both independent cross-checks across all four holdouts, and the
  approximately 0.998 submission standard under genuinely independent
  validation, stand as written in Section 1. They are not lowered for
  this idea, and no result from the independent path substitutes for
  them at Stage 2.
- The validation criteria. The frozen holdout protocol, the A and B
  cross-check discipline, the four-holdout separate reporting, and the
  rule that real-window distributions are diagnostics only, all stand.
  Section 6 adds a screen; it amends nothing.
- The reference structure. Phase 5 bandext_maskfix remains the
  pre-calibration baseline; Phase 7 full-strength remains the
  character-calibration reference; Phase 8A remains the transferability
  bound; Phase 8B quarter strength remains the calibration reference.
  Phase 6's combination remains disqualified. A decomposition result, if
  any, is reported against these references, never instead of them.
- The settled findings. The label arithmetic, the bandext conversion,
  the maskfix mask form, and the closure of background matching
  (permanent, including gated and hybrid variants) are not reopened.
  Decomposition is designed inside those findings, not around them.
- The write-up. WRITEUP_phase0_8b.md stands as the settled record of
  Phases 0–8B. If Step 3 runs, its outcome, success or failure, is a new
  document appended to the project's record, not an edit to the
  write-up's account of the completed programme. (Its Section 10 outlook
  already describes this direction accurately as a hypothesis,
  unauthorised and untested; only the authorisation status would change,
  and that change would be recorded in the new document.)
- The submission posture. One slot used of 200. No candidate file
  exists. No submission without Godson's explicit approval, at the
  standard, whatever Step 3 produces.

---

## Appendix: numbers used in this proposal and their sources

Every load-bearing number above traces to the frozen record:

- Benchmark 0.753867825; public subset about 30% of test; top public
  score 0.994250562 (melihk) and Godson22 rank 624 of 640: competition
  leaderboard, live-checked 2026-11-18; also PLAN.md Section 1.
- Submission 1: simulator v2, 2026-11-16, 0.687868143, Zindi ID
  tsB9FZT6: SUBMISSIONS.md; ARCHIVE_MANIFEST.md Section 5, Phase 2.
- Label audit (observed-cells sum the only surviving definition;
  alternatives capped at 0.7756 to 0.9351; rounding immaterial at
  0.0008): RESULTS_conversion.md, workstream 2.
- Construction ceilings 0.9981 (bandext_maskfix) and 0.9983 (Phase 4
  construction): RESULTS_conversion.md; RESULTS_sim_v3.md.
- Phase 5 baseline holdouts 0.9482, 0.9030, 0.9469, 0.9422 (random,
  site, period, long); real-window mean 0.2061; conversion screen
  (nine variants 0.319 to 0.336; bandext 0.2042): RESULTS_conversion.md.
- Site failure anatomy (zero-band bias plus 0.065, RMSE 0.090, 21.4% of
  site squared error; zero and (0, 0.05] bands 38.3% combined; Bird
  Island 0.9011 with bias plus 0.026; Fish Hoek 0.9284; mid and high
  bands about 58% of seen-holdout squared error): RESULTS_phase6_model.md,
  workstream 2.
- Band 0 (32% of model gain; grouped permutation cost 0.194; two level
  features identify site at 0.42 against 0.16 chance; site
  identifiability 0.977 without band 0; failure is calibration, not
  identity leakage): RESULTS_phase6_model.md, workstream 1.
- Model class, loss, volume and breadth eliminations (XGBoost, HistGB,
  MLP ties; huber trade Rejected; 160k volume plus 0.003 to 0.004 seen,
  nothing on site; diverse wash with real mean 0.243):
  RESULTS_phase6_model.md, workstreams 4 and 5, and final protocol.
- Character calibration (site 0.9030 to 0.9232 and 0.9250; zero bias
  plus 0.065 to plus 0.028 and plus 0.025; zero-band RMSE 0.090 to 0.045
  and 0.040; (0.6, 1] bias to minus 0.070; Bird Island band-0 share 1.70
  times the all-site median): RESULTS_phase7_calibration.md.
- Family D null result on zero-band bias (plus 0.064 against plus
  0.064): RESULTS_phase7_calibration.md, workstream 1.
- Phase 8A retention (89% and 91% of bias correction; 83% and 88% of
  RMSE correction; site-gain retention 64% and 80%; correction
  partially survives): RESULTS_phase8_exclusion.md.
- Phase 8B graded result (dose-response; quarter strength site 0.9163
  and 0.9218; (0.6, 1] bias minus 0.056 and minus 0.064; wording rule:
  independent drawn-share assignment with small measured realised-label variation):
  RESULTS_phase8b_graded.md, including Godson's review-qualifications
  addendum.
- Background matching refutation (0 of 60 real windows with an exact
  minute; oracle with true background 0.9528 on the Phase 3
  construction; gated matcher cap about 0.85 to 0.95 on synthetic data;
  direction closed permanently): RESULTS_phase3_diag.md.
- B1 null check (median 0.344, mean 0.398 on 2,100 ambient-only
  windows): PLAN.md Section 4.2; ARCHIVE_MANIFEST.md Section 5, Phase 0.
- Plausible true mean share band 0.13 to 0.20 (arithmetic from the two
  public scores): RESULTS_phase3_diag.md; WRITEUP_phase0_8b.md
  Section 2.2.
- Inter-site spectral-difference RMS median 14.2 dB; applied channel RMS
  11.2 dB; channel family Rejected: RESULTS_conversion.md, workstream 4.
- Background pool 87,653 screened windows; release 1,715,779 records
  across ten sites; test-background periods withheld from the release;
  whole-minute missingness only, minute 1 missing in 41.6% of test rows;
  maskfix form fractions: PLAN.md Section 3; RESULTS_conversion.md,
  workstream 3.
- Passage events (realised mean duration about 10 minutes; 47.9% at
  least 12 minutes): ARCHIVE_MANIFEST.md Section 5, Phase 2.
- True-zero in-family readings 0.014 to 0.018 (bandext full run):
  RESULTS_conversion.md.
- A and B agreement range 0.031 to 0.040 across the frozen families:
  RESULTS_conversion.md; RESULTS_phase8b_graded.md, Q5.

End of proposal. Decision requested of Godson: Option A (approve Stages
0 and 1 only) or Option B (decline, and the frozen record stands).


Decision addendum, recorded 2026-11-18

Godson's decision: Option A. Stages 0 and 1 only are approved, subject to the boundaries below. Stage 2 requires a separate decision. No candidate file, no Zindi submission and no submission slot expenditure is authorised.

Boundaries as he stated them:

- Stage 0 must establish a genuinely useful independent synthetic validation, including its oracle and baseline checks.
- Stage 1 must test the minimal estimators against the predeclared null and performance criteria.
- Any kill criterion ends the work.
- An intermediate result means stop and report, not adjust thresholds.
- The purpose is not to promise another breakthrough. It is to find out, cheaply and rigorously, whether estimating the physical background contribution offers something the previous regression approach could not.

His qualification, binding on all reporting from this work: the independent construction is independent in implementation and held out material, but it still uses the same underlying public data sources and synthetic energy additive assumptions. It must not be represented as independent real world validation.
