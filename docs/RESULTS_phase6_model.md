# Phase 6 — Model estimation and site generalisation: results (2026-11-17)

Written incrementally as each workstream completes, so the findings
survive interruption. Status markers per section.

**Standards (Godson, binding, unchanged from Phase 5).** Development
minimum: **0.96** on the fixed validation AND both independent
cross-checks (A and B), all four holdouts (random, site, period,
long) reported separately. Submission standard: **≈0.998 under
genuinely independent validation**. No candidate submission file is
produced and no Zindi slot is spent under any outcome of this phase
without Godson's explicit approval. The real-window prediction
distribution is a consistency diagnostic only, never a substitute
for validation scores.

**Frozen construction (not reopened).** bandext_maskfix:
bandext passage conversion (band 0 from 0 Hz, band 24 to Nyquist),
observed-cells sum label, maskfix mask form, natural background
traffic never counted toward the label, no uniform-noise
augmentation, background matching closed. All Phase 6 experiments
train and evaluate on the identical frozen generation files
`data/sim_cf_bandext_maskfix_*.npz` (train 80k / trainA 40k /
trainB 40k / four 6k holdouts / oracle 2k), so every comparison
against the baseline is paired on identical windows. The only
exception is workstream 5, where the training data itself is the
variable under test; evaluation still uses the identical frozen
holdout files.

**Baseline to beat** (Phase 5 full run, `models/conv_bandext_maskfix_full.json`):
main holdouts random **0.9482** / site **0.9030** / period **0.9469**
/ long **0.9422**; A 0.9434 / 0.9002 / 0.9439 / 0.9381; B 0.9436 /
0.9051 / 0.9439 / 0.9383; A–B agreement RMSE 0.031–0.040;
construction ceiling 0.9981; real-window mean 0.2061 (clean 51.3%
at 0.124, ambiguous 42.7% at 0.279, unresolved 5.9% at 0.391).
Screening protocol for this phase: candidate variants are trained
on the frozen trainA file (40k) and paired against the baseline A
model (trained on the same file) on the identical frozen holdouts;
anything that survives screening is re-run under the full
main/A/B protocol before any gate claim. Model-to-model noise at
screen scale is ≈±0.005; effects inside that band are reported as
unresolved, not as wins.

**Godson's caution, binding:** a variant that improves the random
holdout but worsens the site holdout is NOT progress and is
reported as such. The site holdout is a first-class criterion.

**Finding classifications:** **Measured** / **Supported** /
**Unresolved** / **Rejected**, as before.

---

## Workstream 2 — Share-band error analysis — COMPLETE

Code: `code/phase6_ws2_bands.py`; record: `models/phase6_ws2.json`.
Method: frozen baseline main model on the identical frozen
holdouts; bias = mean(prediction − truth) per share band.

| holdout | band | n | bias | RMSE | SSE share |
|---|---|---|---|---|---|
| random (0.9482) | 0 | 1473 | +0.0133 | 0.0262 | 6.3% |
| | (0,0.05] | 1313 | +0.0147 | 0.0328 | 8.8% |
| | (0.05,0.1] | 489 | +0.0090 | 0.0408 | 5.1% |
| | (0.1,0.3] | 1369 | −0.0015 | 0.0569 | 27.6% |
| | (0.3,0.6] | 757 | −0.0160 | 0.0806 | 30.6% |
| | (0.6,1] | 599 | −0.0236 | 0.0765 | 21.8% |
| site (0.9030) | 0 | 1481 | **+0.0649** | 0.0904 | **21.4%** |
| | (0,0.05] | 1364 | **+0.0525** | 0.0837 | 16.9% |
| | (0.05,0.1] | 525 | +0.0317 | 0.0735 | 5.0% |
| | (0.1,0.3] | 1342 | +0.0082 | 0.0908 | 19.6% |
| | (0.3,0.6] | 758 | −0.0335 | 0.1312 | 23.1% |
| | (0.6,1] | 530 | −0.0510 | 0.1221 | 14.0% |
| period (0.9469) | 0 | 1489 | +0.0126 | 0.0283 | 7.1% |
| | (0,0.05] | 1273 | +0.0119 | 0.0354 | 9.4% |
| | (0.05,0.1] | 521 | +0.0080 | 0.0499 | 7.7% |
| | (0.1,0.3] | 1317 | −0.0070 | 0.0618 | 29.7% |
| | (0.3,0.6] | 785 | −0.0210 | 0.0799 | 29.6% |
| | (0.6,1] | 615 | −0.0178 | 0.0673 | 16.5% |
| long (0.9422) | (0,0.05] | 1721 | +0.0126 | 0.0351 | 10.6% |
| | (0.05,0.1] | 695 | +0.0057 | 0.0429 | 6.4% |
| | (0.1,0.3] | 1724 | −0.0091 | 0.0594 | 30.3% |
| | (0.3,0.6] | 1031 | −0.0192 | 0.0790 | 32.2% |
| | (0.6,1] | 829 | −0.0215 | 0.0704 | 20.5% |

(val_long has no share-0 windows by construction: forced long
events always have s > 0.)

**Findings.**

1. **Seen-site holdouts show textbook shrinkage (Measured).** On
   random / period / long the model overestimates low shares
   (+0.012…+0.015 at 0 and (0,0.05]) and underestimates high
   shares (−0.016…−0.024 above 0.3) — the classic
   regression-to-the-mean of an MSE-trained regressor. Squared
   error concentrates in (0.1,0.3] (≈28–30%) and (0.3,0.6]
   (≈30–32%), confirming Phase 5's mid-band diagnosis with finer
   resolution.
2. **The site holdout fails differently: hallucinated additions
   on unseen-site backgrounds (Measured).** At true share 0 the
   bias is **+0.065** (RMSE 0.090 against 0.026 on random) and
   the zero band alone carries **21.4%** of the site holdout's
   total squared error; (0,0.05] adds +0.053 bias and 16.9%.
   The model reads natural background character at unseen sites
   as added-vessel energy. Split by site: Bird Island (n=5,540)
   scores 0.9011 with overall bias +0.026; Fish Hoek (n=460)
   scores 0.9284 with bias +0.003 — **Bird Island is the problem
   site**, and its failure is concentrated at low true shares.
3. **The model's reliance on band 0 is extreme (Measured).**
   Grouped permutation of the 19 band-0-specific features (of
   549) costs **0.194** of score on val_random and **0.125** on
   val_site; the size-matched band-12 control group costs
   0.0015 / 0.0017. Band 0 is where the vessel signal lives
   (physically expected: vessel radiated noise is low-frequency)
   — and, per finding 2, it is also the channel through which
   unseen-site background character is misread as signal.
   Workstream 1 now has to separate those two roles.

## Workstream 1 — Site-robust learning — COMPLETE

### The band-0 question, answered with evidence

Code: `code/phase6_ws1_diag.py`; record: `models/phase6_ws1_diag.json`.
Site classifier: LightGBM multiclass on 18,512 pure background
windows sampled from the release pool (≤3,000 per site, 8 sites;
majority-class chance 0.162), 80/20 split.

| feature set | site accuracy |
|---|---|
| all 549 features | 0.9814 |
| band-0-specific features only (19) | 0.5231 |
| band-0 level only (band_share_0, band_mean_db_rel_0) | 0.4161 |
| all except band-0-specific (530) | 0.9768 |

Baseline main model's gain importance: the band-0 group holds
**32.0%** of total gain; `band_share_0` alone is the top feature
at 25.1%, followed by `band_share_1` (15.3%), `band_share_24`
(11.7%) and `band_range_db_24` (11.4%).

**Answer (Measured).** Both roles are real, and they are not the
same mechanism. (i) Band 0 *is* a site channel: two band-0 level
features alone identify a background's site at 0.42 against 0.16
chance, and the 19 band-0 features reach 0.52. (ii) But band 0 is
*not* the model's site fingerprint in any exclusive sense:
removing it leaves site identifiability at 0.977 (from 0.981) —
site character saturates the whole spectrum, so "stop the model
learning site identity" is not an achievable or well-posed aim.
(iii) What band 0 uniquely is: the dominant *prediction* channel
(32% of gain; grouped permutation cost 0.194 on val_random,
workstream 2). The site-holdout failure mode (workstream 2,
finding 2) is a calibration failure, not an identity leak: at
unseen sites the natural band-0 share level sits higher than the
training sites' range, and the model's learned mapping from
band-0 share to added share over-fires, producing +0.065 bias at
true share 0. The correct target for site-robust work is
therefore the *form* in which band-0 (and low-band) level enters
the model — contrasts and within-window references rather than
absolute share levels — plus training strategies that penalise
site-specific calibration. The strategy screens below test
exactly those levers.

### Strategy screens

Each variant is trained on the frozen trainA file under the
frozen protocol with one change, paired against baseline A
(0.9434 / 0.9002 / 0.9439 / 0.9381, real mean 0.2099). Code:
`code/phase6_ws1_screens.py`; record: `models/phase6_ws1_screens.json`.

**dropb0 — remove the 19 band-0-specific features: Rejected
(Measured).** Scores 0.9326 / 0.8925 / 0.9340 / 0.9272 — worse
than baseline A on *every* holdout (random −0.011, site −0.008,
period −0.010, long −0.011) — and the real-window mean rises to
0.2583. It also trains to the 6000-round cap without early
stopping firing, i.e. the remaining features keep finding slow
signal but never reach the band-0 model's quality. Band 0 is
load-bearing for the vessel signal itself; deleting it is not a
site-robustness strategy. This closes the "memorisation channel"
remedy in its naive form: the problem is the *calibration* of
band-0 level across sites, not the band's presence.

**loso — leave-site-out early stopping (St Croix Island as the
internal validation site): no measurable effect (Measured).**
Scores 0.9420 / 0.8984 / 0.9444 / 0.9371 against baseline A's
0.9434 / 0.9002 / 0.9439 / 0.9381 — every difference within
±0.002, inside screen noise; real mean 0.2018 vs 0.2099, also
inside the ±0.01 model-to-model variation seen in Phase 5. The
site-held-out validation loss kept improving to round 5993 of
the 6000 cap, i.e. at this capacity the model is not overfitting
to seen sites in a way that site-held-out stopping can detect.
**Rejected as an improvement lever.**

**bal — site-balanced sample weights: no measurable effect
(Measured).** Scores 0.9427 / 0.9020 / 0.9440 / 0.9362 against
baseline A — site +0.002, everything else within ±0.002; real
mean 0.2046 vs 0.2099. Countering the pool's 65% Richards Bay
skew by weighting does not change what the model learns about
unseen sites. **Rejected as an improvement lever.**

### Workstream 1 synthesis

The band-0 question is answered: band 0 carries both the
dominant vessel signal and a site-dependent level calibration,
and the site-holdout failure is the calibration, concentrated at
Bird Island backgrounds and at low true shares. None of the
three training-side strategies moves the site holdout beyond
noise: deleting band 0 is strictly harmful (**Rejected**),
leave-site-out stopping is null (**Rejected** as a lever),
site-balanced weights are null (**Rejected** as a lever). What
remains untested on the site axis is the *form* of the features
themselves — within-window contrasts instead of absolute share
levels — which is workstream 3's family C, and the diversity of
the training backgrounds, which is workstream 5. Workstream 1 is
complete; its surviving hypothesis passes to those two
workstreams.

## Workstream 3 — Feature improvements — COMPLETE

Protocol: one family at a time added to the frozen 549, trained
on trainA under the frozen protocol, paired against baseline A
(0.9434 / 0.9002 / 0.9439 / 0.9381, real mean 0.2099). Code:
`code/phase6_features.py`, `code/phase6_ws3_screens.py`; record:
`models/phase6_ws3_screens.json`.

**Family A — per-band excess (50 features): Rejected (Measured).**
Justification was that an added passage is excess energy over
the window's own per-band background level, generalised from the
≤1 kHz B1 statistic to all bands. Scores 0.9430 / 0.8962 /
0.9449 / 0.9362: random −0.000, site **−0.004**, period +0.001,
long −0.002; real mean 0.2094. No holdout gains beyond noise and
the site holdout moves the wrong way. The base feature set
(per-band range/std, B1) already encodes this information; the
explicit per-band form adds nothing.

**Family B — per-band temporal concentration (25 features):
small positive, the phase's first site-holdout gain (Supported).**
Justification: a passing vessel concentrates a band's energy
into a few minutes while ambient is diffuse; the base set has
this only in aggregate (max/top-3 minute share over all bands).
Scores 0.9437 / **0.9037** / 0.9449 / 0.9377 against baseline A:
site **+0.0035**, random +0.000, period +0.001, long −0.000;
real mean 0.2064 vs 0.2099. The gain is at the edge of screen
noise (±0.005) and is claimed only as Supported, but it is the
only variant so far that moves the site holdout in the right
direction without costing any other holdout, so it is carried
into the final combination for confirmation at full scale.

**Family C — spectral shape residuals + band-0 contrasts (28
features): the strongest variant of the phase so far
(Supported).** Justification: site identity lives in the smooth
spectral tilt; the vessel is a localised bump on top of it, so
the residual of the mean spectrum after a 5-band smooth, plus
band-0 contrasts against its neighbours, should carry the vessel
signal in a site-invariant form. Scores 0.9455 / 0.9019 /
0.9452 / 0.9396 against baseline A — small uniform gains on all
four holdouts (random +0.002, site +0.002, period +0.001, long
+0.002) — and the real-window mean falls from 0.2099 to
**0.1574**, inside the 0.13–0.20 plausible prior band where the
whole Phase 5 family sat at or above its top edge (0.196–0.217).
The synthetic gains alone would be inside noise; the combination
of uniform synthetic gains with a large real-diagnostic move in
the predicted direction is what makes this Supported rather than
Unresolved. The diagnostic caveat stands: a lower real mean is
consistent with better calibration but cannot prove it without
labels.

### Workstream 3 synthesis

Family A is **Rejected**; families B and C are each
independently non-negative on every holdout and both act on the
site/identity axis workstream 1 identified. The justified
combination carried to the final full protocol is therefore
**549 + B + C (602 features)**, with A excluded. Workstream 3 is
complete.

## Workstream 4 — Model comparisons — COMPLETE

Protocol: identical windows, frozen 549 features, trained on
trainA, paired against baseline A (LightGBM, 0.9434 / 0.9002 /
0.9439 / 0.9381, real mean 0.2099). Code:
`code/phase6_ws4_models.py`; record: `models/phase6_ws4.json`.
No AutoML anywhere; all hyperparameters are the manual frozen
set or the documented analogue in the other library.

**XGBoost 3.4.1 (hist, depth 6, lr 0.05, cap 6000, ES 200):
statistical tie with LightGBM (Measured).** Scores 0.9430 /
0.8970 / 0.9452 / 0.9373 — every holdout within ±0.003 of
LightGBM, site −0.003, period +0.001; real mean 0.2174; best
iteration 5953 (966 s). The GBM implementation is not the
bottleneck: two independent boosting libraries land on the same
scores, which is itself evidence the remaining error is in the
information the features carry, not the optimiser.

**sklearn HistGradientBoostingRegressor: also a tie
(Measured).** Scores 0.9419 / 0.8991 / 0.9440 / 0.9368, every
holdout within ±0.002 of LightGBM, real mean 0.2102 against
LightGBM's 0.2099; 4,925 iterations (978 s on rerun; the first
run's identical training was lost to a service restart before
recording). Three independent GBM implementations now agree to
within noise on every holdout and on the real-window diagnostic:
the model class within gradient boosting is conclusively not
where the remaining error lives.

**MLP (sklearn, 128-64, standardised features): same
neighbourhood (Measured).** Scores 0.9409 / 0.9010 / 0.9405 /
0.9308 — a smooth function class lands within 0.003 of the
GBMs on three holdouts (long −0.007 is its worst), real mean
0.2088. Not better overall and not a candidate, but it extends
the convergence evidence beyond tree models entirely.

**Huber-loss LightGBM: first run vacuous, rerun with a
meaningful alpha.** The first run used alpha=0.9; with labels in
[0,1] the residuals never reach 0.9, so huber degenerates to
exactly L2 — verified empirically (a 50-round paired test gives
max prediction difference 0.0) and visible in the record (that
run's entry was bit-identical to baseline A on every holdout,
which also serves as a bit-for-bit determinism check of this
phase's harness against Phase 5). The rerun (alpha=0.05, on the
scale of the actual residuals; 4,743 rounds) scores 0.9457 /
0.8987 / 0.9463 / 0.9395, real mean 0.2149: small gains on
random (+0.002), period (+0.002) and long (+0.001) bought with
a small site loss (−0.0015). By Godson's rule — random gains at
the site holdout's expense are not progress — huber is
**Rejected** as the final configuration, though the differences
are near noise in both directions.

### Workstream 4 synthesis

Four alternatives to the frozen LightGBM/L2 configuration were
tested on identical windows: XGBoost (tie), HistGradientBoosting
(tie), an MLP (tie within 0.003 on three holdouts, worse on
long), and huber loss (small random-for-site trade). **Measured:
the model class and the loss are not the binding constraint.**
Independent implementations with different inductive biases
converge to the same scores and the same real-window readings,
which localises the remaining error in the information content
of the features and the training distribution — workstreams 3
and 5's territory. Workstream 4 is complete; LightGBM with the
frozen protocol remains the model.

## Workstream 5 — Training coverage — COMPLETE

Protocol: evaluation always on the identical frozen holdouts,
paired against baseline main (0.9482 / 0.9030 / 0.9469 / 0.9422,
real mean 0.2061). Generation: `code/phase6_gen_coverage.py`
(frozen train pool and maskfix mask; fresh recorded seeds —
moresame 20373116, diverse 20374116). Training:
`code/phase6_ws5_train.py`; record: `models/phase6_ws5.json`.

**More of the same distribution (160k = frozen 80k + fresh
80k): small on-distribution gains, no site gain (Measured).**
Scores 0.9515 / 0.9037 / 0.9513 / 0.9461: random +0.003,
period +0.004, long +0.004, but site only **+0.001**; real mean
0.2152 (within variation of 0.2061). Doubling the volume of the
same distribution buys a little accuracy where the distribution
already matches and nothing where it does not — direct evidence
that the site gap is a distribution/calibration problem, not a
sample-size problem.

**Diverse (80k broadened distribution, equal volume): neutral
on synthetic holdouts, worse on the real diagnostic
(Measured).** Scores 0.9477 / 0.9046 / 0.9480 / 0.9403: random
−0.001, site +0.002, period +0.001, long −0.002 — a wash. But
the real-window mean rises to **0.2428**, above the plausible
band, against the baseline's 0.2061: the flattened share prior
(more mid/high-share mass by design) raises the model's prior
expectation on real windows in the wrong direction. Broadening
the variation axes (colour, warp, gain, background modes) did
not buy site robustness either — the site holdout moved only
+0.002, inside noise.

### Workstream 5 synthesis — volume and diversity are different
levers, and neither is the site answer

Reported separately, as required: **more windows of the same
distribution** give a real but small on-distribution gain
(+0.003…+0.004 on random/period/long) and no site gain;
**more diverse windows** at equal volume give no synthetic gain
and a worse real-window diagnostic. The binding constraint is
not how much training data the model sees, nor how widely the
existing variation axes are stretched. Workstream 5 is
complete.

## Final full protocol — 549 + families B + C — COMPLETE

Configuration: frozen bandext_maskfix splits, frozen LightGBM
protocol, 602 features (549 + B temporal concentration + C
spectral shape residuals — the two families independently
justified at screen scale). Code: `code/phase6_final.py`;
record: `models/phase6_final.json`; models
`models/phase6_final{,_A,_B}.txt` (main and A were checkpointed
to disk and resumed after a service restart killed the first
run during B's training; B trained fresh in the resumed run).

| holdout | main | A | B | baseline main / A / B | A–B agreement |
|---|---|---|---|---|---|
| random | 0.9487 | 0.9457 | 0.9455 | 0.9482 / 0.9434 / 0.9436 | 0.0280 |
| site | **0.9012** | **0.8944** | **0.8985** | 0.9030 / 0.9002 / 0.9051 | 0.0385 |
| period | 0.9482 | 0.9454 | 0.9441 | 0.9469 / 0.9439 / 0.9439 | 0.0304 |
| long | 0.9439 | 0.9400 | 0.9411 | 0.9422 / 0.9381 / 0.9383 | 0.0306 |

**Verdict (Measured): the combination does not replicate at
full scale, and by Godson's rule it is not progress.** Random,
period and long improve marginally (+0.001…+0.002 on main), but
the site holdout is worse for all three models (main −0.002,
A −0.006, B −0.007), and the site zero-band bias grows slightly
(+0.071 vs the baseline's +0.065). The screen-level site gains
of families B and C were split-specific noise; the full protocol
is the arbiter and it votes against. The baseline
bandext_maskfix construction therefore remains the best
construction. A random-for-site trade is exactly the failure
mode Godson named in advance, and it is reported as one.

**The one genuinely open signal (Unresolved).** The B+C models
read the real windows far lower than the baseline family: main
mean **0.1718** (baseline 0.2061), A 0.1480, B 0.1480, with the
clean majority growing to 55.4% reading 0.094 and the
unresolved group shrinking to 5.1% at 0.407. Two readings of
this remain open: the shape features may correct a real
over-read that synthetic holdouts cannot see (they share the
simulator's background distribution; only the site holdout
probes distribution shift, and it votes weakly against), or
they may make the model under-read real additions. The phase's
own standards make the site holdout the arbiter, so this signal
is recorded as Unresolved, not as evidence of improvement — but
it is the most interesting open question the phase leaves.

## Gate judgement — NOT MET (no candidate)

Against the binding standards, for the phase's best full
protocol (and, in parentheses, the unchanged baseline, which
remains the best construction):

- Development minimum 0.96 on the fixed validation AND both
  cross-checks: **not met.** Best holdout 0.9487 (random, main);
  cross-checks peak at 0.9457 / 0.9455; the site holdout is
  0.9012 (baseline 0.9030).
- Submission standard ≈0.998: **not met.** Distance to aim on
  the best holdout ≈0.049; the construction ceiling (0.9981)
  still shows the gap is estimation error, not construction.

**No candidate submission file was produced** (`submissions/`
still holds only the pre-Phase-5 files), **no Zindi slot was
spent** (still 1 of 200), and the Zindi site was not opened.
The Phase 6 test-prediction CSVs in `models/` are diagnostics,
labelled as such.

**What Phase 6 established, in one paragraph.** The remaining
error is not in the model class (four model families converge),
not in the loss (huber trades site for random), not in training
volume (+0.003 on-distribution only), not in the breadth of the
existing variation axes (neutral synthetic, worse real
diagnostic), and not removable by deleting, reweighting or
re-stopping around band 0. It has two measured components:
(i) an **identifiability limit in the mid share bands** —
(0.1,0.3] and (0.3,0.6] carry ≈58% of squared error on seen
holdouts with textbook shrinkage, identical across every model
class tried; and (ii) a **site-generalisation failure at low
shares** — on unseen Bird Island backgrounds the model
hallucinates +0.065 of added share at true zero, and no lever
tested in this phase moved it. Component (ii) is the largest
single remaining error source: the zero and (0,0.05] bands
together carry 38–42% of the site holdout's squared error.

**Phase 7 recommendation** (for Godson's decision; nothing
started). Attack identifiability and site calibration
directly, not capacity: (1) a **leave-one-site-out ensemble** —
train one model per held-out training site (six folds) and
average; each member's calibration is site-blind in a different
direction, and the average directly targets the mechanism the
single LOSO stopping rule could not reach. (2) **Per-window
background modelling in feature space**: estimate each window's
own background spectrum from its quiet minutes and featurise
the *deviation* of the whole window from that self-estimate —
a generalisation of the B1 idea to all bands and both edge
bands, distinct from the closed background-matching direction
(no release windows are matched or subtracted; the reference
is the window itself). (3) A **decisive test of the family-C
real-diagnostic divergence**, which needs Godson's ruling
because it touches test inputs: add a synthetic passage of
known share to copies of real windows and check whether each
model family's predictions increment by the known amount — a
consistency diagnostic, not a fit, but beyond the
inference-only use made of test data so far. (4) If a simulator
axis is reopened at all, the measured candidate is Bird
Island's spectral character: synthesise training backgrounds
with the held-out sites' measured low-frequency character
(without training on those sites' windows) and test whether
the zero-band hallucination shrinks on the untouched site
holdout.

**Seeds and configurations.** All Phase 6 trainings use the
frozen protocol (LightGBM, seed 20261116, 6000-round cap,
patience 200, identical internal split) unless a workstream
states its own change: LOSO split site '5 St Croix Island';
site weights w = N/(n_sites·n_site); XGBoost 3.4.1 (depth 6,
lr 0.05, ES 200); HistGB (max_leaf 63, ES via validation
fraction 0.1, patience 200); MLP (128-64, alpha 1e-3, max 200
iters, ES 20); huber alpha 0.05 (the alpha 0.9 run is vacuous,
documented in workstream 4). Coverage generation seeds:
moresame 20373116, diverse 20374116 (recorded in
`data/sim_p6*_meta.json`); diverse axes are listed in
`code/phase6_gen_coverage.py`. Site classifier seed 20261117.
Per-run best iterations are in the `models/phase6_*.json`
records. Process note: the phase survived three service
restarts by checkpointing every result to disk as it completed;
the only losses were one histgb training's scores (rerun,
identical iteration count) and the final run's B training
(resumed from saved main/A models).
