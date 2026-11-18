# Phase 7 — Site calibration: results (2026-11-17)

Written incrementally as each workstream completes, so the findings
survive interruption. Status markers per section.

**Standards (Godson, binding, unchanged from Phases 5–6).**
Development minimum: **0.96** on the fixed validation AND both
independent cross-checks (A and B), all four holdouts (random,
site, period, long) reported separately. Submission standard:
**≈0.998 under genuinely independent validation**. No candidate
submission file is produced and no Zindi slot is spent under any
outcome of this phase without Godson's explicit approval (still
1 of 200 used). The Zindi site is not visited. The real-window
prediction distribution is a consistency diagnostic only, never a
substitute for validation scores.

**Frozen construction (not reopened).** bandext_maskfix:
bandext passage conversion (band 0 from 0 Hz, band 24 to
Nyquist), observed-cells sum label, maskfix mask form, natural
background traffic never counted toward the label, no
uniform-noise augmentation, background matching closed.
Conversion, target arithmetic and missingness are NOT reopened
in this phase. All Phase 7 screens train and evaluate against
the identical frozen generation files
`data/sim_cf_bandext_maskfix_*.npz` (train 80k / trainA 40k /
trainB 40k / four 6k holdouts / oracle 2k), so every comparison
against the baseline is paired on identical windows. Where a
workstream varies the training data itself (workstreams 2 and
3), evaluation still uses the identical frozen holdout files.

**Baseline reference** (Phase 5 bandext_maskfix full run,
`models/conv_bandext_maskfix_full.json`): main holdouts random
**0.9482** / site **0.9030** / period **0.9469** / long
**0.9422**; A 0.9434 / 0.9002 / 0.9439 / 0.9381; B 0.9436 /
0.9051 / 0.9439 / 0.9383; A–B agreement RMSE 0.031–0.040;
construction ceiling 0.9981; real-window mean 0.2061 (clean 770
at 0.124, ambiguous 641 at 0.279, unresolved 89 at 0.391).
Phase 6's final combination (0.9487 / 0.9012 / 0.9482 / 0.9439)
is NOT the reference: its site regression disqualifies it under
Godson's rule. Screening protocol (as Phase 6): variants are
trained on the frozen trainA file (40k) and paired against the
baseline A model (0.9434 / 0.9002 / 0.9439 / 0.9381, real mean
0.2099) on the identical frozen holdouts; anything that survives
screening is re-run under the full main/A/B protocol before any
gate claim. Model-to-model noise at screen scale is ≈±0.005;
effects inside that band are reported as unresolved, not wins.

**Central objective (Godson's words govern):** reduce
zero-share hallucination on unseen sites without sacrificing
the corrected passage signature or performance on the other
three holdouts. Known target, measured in Phase 6 workstream
2: on the site holdout, true-share-0 windows are over-read by
**+0.065** (RMSE 0.090 vs 0.026 on random) and the zero band
carries **21.4%** of site SSE; the 0 and (0,0.05] bands
together carry 38.3%. Bird Island (n=5,540) scores 0.9011 with
bias +0.026; Fish Hoek (n=460) scores 0.9284 with bias +0.003.

**Godson's cautions, binding:** each workstream is tested
INDEPENDENTLY against the baseline first; no combining up
front; only after independent results, ONE small interaction
experiment on the strongest candidates. A variant that improves
the random holdout but worsens the site holdout is NOT progress
(workstream 3 carries the stronger form: improvement required
on ALL FOUR holdouts). Never lower the gate; never change
validation criteria after seeing results.

**Finding classifications:** **Measured** / **Supported** /
**Unresolved** / **Rejected**, as before.

---

## Perturbation test — PREDECLARED SET (written before the test runs)

Godson has ruled this test **permitted with restrictions**, and
the restrictions are binding exactly as written:

- Apply predeclared, physically defensible perturbations to
  test inputs.
- Do not fit, calibrate, select thresholds, train models or
  generate pseudo-labels from test inputs.
- Do not use test outcomes to modify the model.
- Record prediction stability and disagreement as diagnostics
  only, not as a route to improve test predictions.
- Keep the test inputs unchanged for the eventual submission
  pipeline.

Purpose: determine whether the Phase 6 B+C models' lower
real-window predictions (main mean 0.1718, clean majority 55.4%
at 0.094) are robust, or whether those models are unusually
sensitive to small spectral changes. The test cannot establish
which prediction is correct; it will be reported that way.

**Predeclared perturbation set** (applied to copies of the raw
Test.csv windows; the stored test inputs are never modified;
features recomputed with the identical fixed transforms;
models: the Phase 5 baseline main model and the Phase 6 final
main model, i.e. the two model families whose real-window
readings diverge):

| id | perturbation | physical basis |
|---|---|---|
| G+ | global gain +2 dB (all cells) | organisers discarded absolute gain; level is not signal |
| G− | global gain −2 dB (all cells) | same |
| T+ | spectral tilt: −0.5 dB at band 0 rising linearly to +0.5 dB at band 24 | a slightly different export/recorder channel |
| T− | the reverse tilt (+0.5 dB at band 0 to −0.5 dB at band 24) | same |
| B+ | band 0 only, +1 dB | the dominant channel's calibration uncertainty |
| B− | band 0 only, −1 dB | same |
| M8 | minute 8 set to missing (NaN) where observed | whole-minute missingness is the test set's own measured pattern |
| RV | minute order reversed | the organisers vary time trajectories, including reversal (notebook cell 9) |

**Predeclared metrics, per perturbation and per model:** mean
signed Δ prediction, mean |Δ|, p95 |Δ|, fraction of windows
with |Δ| > 0.05, the perturbed real-window mean, and the RMSE
disagreement between the two model families before and after
perturbation. No other use is made of the outputs.

Status: PREDECLARED — test runs after workstreams 1–4.

---

## Workstream 1 — Background-relative features — COMPLETE

Family D, "self-referenced background deviation" (56 features;
code `code/phase7_features.py`; screen `code/phase7_ws1_screen.py`;
record `models/phase7_ws1.json`). Design: each window's own
background spectrum is estimated from its 5 quietest minutes
(per-band median energy); the features are (i) the per-band
deviation, in dB, of the window's mean level from that
self-estimate (25), (ii) the normalised spectral shape of the
excess above the self-estimate (25) plus the total excess
fraction, (iii) low-band deviation contrasts (2), and (iv) the
temporal concentration of the low-band excess (3). All are
ratios to the window's own background — the within-window
reference form Phase 6 workstream 1 named as untested, and
distinct from family A (excess over a per-band percentile as an
absolute energy share; Rejected in Phase 6).

Screen (trainA, 605 features, best iteration 5,503), paired
against baseline A on identical windows:

| holdout | family D | baseline A | Δ |
|---|---|---|---|
| random | 0.9445 | 0.9434 | +0.001 |
| site | 0.9042 | 0.9002 | **+0.004** |
| period | 0.9452 | 0.9439 | +0.001 |
| long | 0.9364 | 0.9381 | −0.002 |

Site holdout by share band (bias / RMSE; baseline A in
parentheses): 0: **+0.064 / 0.085** (+0.064 / 0.088);
(0,0.05]: +0.052 / 0.078 (+0.055 / 0.084); (0.05,0.1]: +0.034 /
0.074 (+0.035 / 0.076); (0.1,0.3]: +0.010 / 0.089 (+0.010 /
0.092); (0.3,0.6]: −0.035 / 0.133 (−0.040 / 0.141); (0.6,1]:
−0.055 / 0.132 (−0.055 / 0.133). Zero-band SSE share 19.6%
(baseline A 19.2%). Real-window mean 0.2039 (baseline A 0.2099).

**Findings.**

1. **The zero-band bias does not move (Measured).** The phase's
   named target — the +0.065 over-read at true share 0 on
   unseen sites — is +0.064 with family D against +0.064 for
   baseline A on identical windows. Referencing each window
   against its own quiet minutes does not correct the level
   shift: on Bird Island backgrounds the *whole window*,
   quiet minutes included, carries the site character, so the
   self-estimate absorbs the character instead of exposing it,
   and an added vessel's deviation profile on such a background
   still resembles what the model has learned to over-read.
2. **A small, broad RMSE improvement on the site holdout
   (Supported).** Site score +0.004 with RMSE lower in every
   share band (largest gains at (0,0.05] and (0.3,0.6]) and no
   cost on random or period; long −0.002 is inside screen
   noise. The gain is at the edge of the ±0.005 screen-noise
   band, so it is claimed only as Supported, but its direction
   is consistent across bands and it is the first feature
   family whose site gain comes with a lower real-window mean
   (0.204 vs 0.210) rather than a higher one.
3. Family D is therefore a **candidate for the interaction
   experiment** (non-negative where it matters, on the site
   axis), but it is not by itself a site-calibration fix: it
   reduces variance around the hallucination without reducing
   the hallucination.

## Workstream 2 — Held-out site-character synthesis — COMPLETE

Code: `code/phase7_gen_char.py` (generation), `code/phase7_ws2_screen.py`
(screen); records `data/sim_p7char_meta.json`,
`models/phase7_ws2.json`. Method, as directed: each released
site's character is summarised by the median per-band
energy-share vector of its screened pool windows. A background
drawn by the frozen pool is then, with probability 0.6, given
another site's character — per-band energies multiplied by
target_shape / source_shape, total energy preserved — with the
target drawn uniformly from all eight pool sites (the six
training sites plus Bird Island and Fish Hoek). Everything
else is the frozen construction: bandext passages, maskfix
mask, observed-cells label, passage variation unchanged and
independently controlled. 21,004 of the 40,000 trainA
backgrounds were transplanted (the remainder drew no
transplant or their own site's shape). Fresh generation seed
20358118. Screen model: frozen 549 features, frozen protocol,
best iteration 5,288; evaluation on the identical frozen
holdouts, paired against baseline A.

**The measured characters explain the pathology (Measured).**
Bird Island's median shape carries **1.70×** the all-site
median share in band 0 and 1.39× in band 1: its natural
backgrounds are band-0-heavy, which is precisely what the
baseline model reads as added vessel energy. Fish Hoek is the
mirror image (band 0 at 0.41×, band 5 at 2.88×) — and Fish
Hoek was never the problem.

| holdout | char-synth | baseline A | Δ |
|---|---|---|---|
| random | 0.9437 | 0.9434 | +0.000 |
| site | 0.9141 | 0.9002 | **+0.014** |
| period | 0.9448 | 0.9439 | +0.001 |
| long | 0.9372 | 0.9381 | −0.001 |

Site holdout by share band (bias / RMSE / SSE share; baseline
A in parentheses): 0: **+0.028 / 0.044 / 6.5%** (+0.064 /
0.088 / 19.2%); (0,0.05]: **+0.019 / 0.043 / 5.7%** (+0.055 /
0.084 / 16.2%); (0.05,0.1]: −0.000 / 0.052 / 3.2% (+0.035 /
0.076 / 5.1%); (0.1,0.3]: −0.026 / 0.082 / 20.1% (+0.010 /
0.092 / 18.9%); (0.3,0.6]: **−0.085 / 0.147 / 37.1%** (−0.040
/ 0.141 / 25.0%); (0.6,1]: **−0.088 / 0.151 / 27.3%** (−0.055
/ 0.133 / 15.6%). Per site: Bird Island 0.9132 (baseline A
0.8982), zero bias +0.028 (+0.067); Fish Hoek 0.9251 (0.9284),
zero bias +0.034 (+0.030). Real-window mean **0.1783** vs
baseline A's 0.2099 — inside the 0.13–0.20 plausible band,
with a synthetic gain attached for the first time.

**Findings.**

1. **The zero-share hallucination is largely a character
   effect, and character training removes most of it
   (Measured).** Zero-band bias falls from +0.064 to +0.028
   and its RMSE halves (0.088 → 0.044); the 0 and (0,0.05]
   bands' combined SSE share collapses from 35.4% to 12.2%.
   The site holdout gain (+0.014) is three times the screen
   noise band, at no cost on random or period (−0.001 on long
   is inside noise). This is the first lever in two phases
   that moves the phase's central target.
2. **The error migrates upward on unseen sites (Measured).**
   The same model under-reads genuinely large additions on
   the site holdout: (0.3,0.6] bias −0.085 and (0.6,1] bias
   −0.088, against −0.040 / −0.055 for baseline A, and those
   two bands now carry 64% of site SSE. Having learned that
   band-0-heavy backgrounds are usually natural, the model
   discounts real additions on such backgrounds too. On seen
   holdouts the high bands are unchanged, so the corrected
   passage signature itself is intact; the trade is specific
   to unseen-site calibration and is reported as part of the
   result, not hidden by the headline gain.
3. **Blindness caveat (recorded, binding on interpretation).**
   The transplant family includes the two holdout sites'
   aggregate median shapes, by Godson's explicit direction
   and the Phase 6 recommendation. No holdout window,
   temporal structure or traffic pattern is trained on, but
   the site holdout's spectral-shape dimension is not fully
   blind for this variant, so its site score is best read as
   the benefit of character coverage, with some upward
   flavour. One internal control argues the gain is not a
   trivial seen-shape artefact: Fish Hoek's shape is equally
   in the family, yet Fish Hoek does not improve (−0.003,
   inside noise) — the gain appears where the pathology was,
   not wherever a shape was seen.
4. Workstream 2 is the phase's strongest candidate and goes
   forward: to the interaction experiment (with family D,
   whose deviation features may specifically defend the
   high-share side the transplant discounts) and, in the
   winning configuration, to the full main/A/B protocol.

## Workstream 3 — Leave-one-site-out ensembling — COMPLETE

Code: `code/phase7_ws3_loso.py`; record: `models/phase7_ws3.json`;
member models `models/phase7_loso_*.txt`. Six folds over the
six training sites present in the frozen trainA backgrounds
(Richards Bay 25,913; St Croix Island 6,473; Saldanha Bay
4,190; St_Francis 2,004; Durban 1,309; Sea Point 111 — Cape
Recife and Riy Banks yielded no screened pool windows). Each
member is the frozen protocol trained on trainA minus one
site's windows (best iterations 3,544–6,000); the ensemble is
the mean of the six members' clipped predictions. Paired
against baseline A on identical windows.

| holdout | LOSO ensemble | baseline A | Δ |
|---|---|---|---|
| random | 0.9444 | 0.9434 | +0.001 |
| site | 0.8979 | 0.9002 | **−0.002** |
| period | 0.9455 | 0.9439 | +0.002 |
| long | 0.9388 | 0.9381 | +0.001 |

Site zero band: ensemble bias **+0.078**, RMSE 0.100, SSE
share 23.6% — against baseline A's +0.064 / 0.088 / 19.2%.
The (0,0.05] band likewise worsens (+0.068 vs +0.055). Real
mean 0.2189 vs baseline A's 0.2099 (wrong direction). Member
site scores individually: 0.864 (−St_Francis) to 0.900
(−St Croix); every member is at or below baseline A on the
site holdout, and the −Richards Bay member drops to 0.9313 on
random, as expected when 65% of the training backgrounds are
removed.

**Findings.**

1. **LOSO ensembling fails Godson's requirement (Measured).**
   Improvement was required on ALL FOUR holdouts; the site
   holdout regresses, and the zero-share hallucination — the
   quantity this workstream existed to reduce — grows by
   +0.014 of bias. **Rejected.**
2. **The mechanism is informative (Supported).** The members'
   errors on unseen sites are correlated, not complementary:
   each site-blind model leans on the same pooled
   level-to-share mapping, so on Bird Island backgrounds all
   six over-read in the same direction and averaging cannot
   cancel a shared bias. Site diversity in the *training set*
   does not create calibration diversity at an unseen site;
   the missing ingredient is information about the unseen
   site's own level, which no training-side ensemble member
   has. This is consistent with workstream 1's finding that
   the failure is a level calibration shift, and it closes
   the ensemble direction in this form.

## Workstream 4 — Koleshjr passage-pool expansion audit — COMPLETE (audit only; nothing incorporated)

Full transcription and five-point audit: `KOLESHJR_AUDIT.md`
(post text supplied by Godson from his signed-in session,
2026-11-17; discussions 35461 and 36170). Godson's verdict
governs: **audit only, not approved for incorporation** — and
on the evidence there is nothing to incorporate.

- **What the expansion is (Measured, from the post text).** 36
  additional strict vessel WAVs from the SAME public VTUAD
  (rebuilt) v1.3.2 strict-interval release (Zenodo record
  23037895, CC BY 4.0) that our own simulator already draws
  on, disjoint in hulls and dates from Koleshjr's own original
  18-clip pilot, selected from metadata and verified against
  the released per-WAV SHA-256 ledger; the first three minutes
  of each converted with 2-second Hann periodograms and
  linear-energy averaging (the starter-style approach).
- **Against the inventory baseline (Godson's figures):** the
  public release describes 525 recordings (509 vessel
  scenarios + 16 dedicated background recordings, ≈56 hours).
  Our own conversion already harvests 542 clips from this
  archive family. The 36 are previously-unused-by-him
  recordings from the same 509 scenarios — **no recordings
  beyond the public release exist in this expansion, so
  relative to our pool it adds nothing.**
- **Five-point audit outcome:** (1) recording inventory
  PARTIAL — count given (36), exact filenames not published
  (retained privately, he states); (2) source provenance MET
  as far as a public declaration can — version, archive parts,
  DOI, licence and the SHA-256 ledger check all named, which
  also confirms the WAVs are unchanged; (3) passage diversity
  — new only relative to his own pilot, not new source
  material; (4) split integrity ASSERTED, not demonstrated —
  hull/date disjointness and frozen evaluation windows are
  stated, no split lists published; (5) incremental value
  NONE OFFERED — the addendum is a declaration, not a results
  post; no baseline comparison or holdout numbers.
- **One open detail, noted:** his conversion follows the
  starter approach with no mention of the edge-band treatment
  Phase 5 identified as the transfer fault (band 0 from 0 Hz,
  band 24 to Nyquist). Whether his training-side additions
  carry the clipped edge bands is not stated.
- The Lawrence_Hlapa_JHB declaration (three recordings, 720
  training and 1,620 development examples, dependent examples
  acknowledged) is a DIFFERENT participant and is not
  attributed to Koleshjr anywhere in this audit.

**Classification: Measured** (from the post text itself).

**Godson's final ruling (2026-11-17): PERMANENTLY CLOSED for
Phase 7.** The 36 recordings are not to be reproduced and not
used as training additions. The workstream is recorded
exactly as: **source expansion genuine within his experiment;
new material for our project none; conversion compatibility
not established; performance benefit untested; decision
exclude.** (Conversion compatibility is not established
because the declaration uses the starter-style two-second Hann
conversion and does not document the corrected edge-band
treatment Phase 5 proved was a major source of transfer
failure; a larger pool through an uncorrected conversion is
not assumed to help — the point is also recorded in the final
section of `KOLESHJR_AUDIT.md`.) All Phase 7 effort goes to
the three substantive experiments (workstreams 1–3).

## Perturbation test — COMPLETE (set predeclared above)

Code: `code/phase7_perturb.py`; record:
`models/phase7_perturb.json`. Sanity anchors reproduce
exactly: unperturbed real means 0.2061 (Phase 5 baseline main)
and 0.1718 (Phase 6 final main); unperturbed cross-model
disagreement RMSE 0.0690. Mean |Δ| per perturbation (baseline
/ Phase 6 final), with the cross-model disagreement RMSE
after perturbation:

| perturbation | baseline | Phase 6 | disagreement |
|---|---|---|---|
| G+ (+2 dB) | 0.0013 | 0.0008 | 0.0692 |
| G− (−2 dB) | 0.0013 | 0.0007 | 0.0686 |
| T+ (tilt) | 0.0123 | 0.0093 | 0.0716 |
| T− (tilt) | 0.0126 | 0.0090 | 0.0675 |
| B+ (band 0 +1 dB) | 0.0156 | 0.0167 | 0.0688 |
| B− (band 0 −1 dB) | 0.0137 | 0.0132 | 0.0692 |
| M8 (minute 8 missing) | 0.0191 | 0.0154 | 0.0724 |
| RV (minute order reversed) | 0.0201 | 0.0160 | 0.0676 |

Signed responses are physically coherent in both models:
band-0 gain moves predictions up (+0.012 / +0.014) and band-0
loss moves them down (−0.010 / −0.010); tilts act in their
signed directions; global gain is a near-perfect invariant
(no window moves by more than 0.006 under ±2 dB), confirming
the share features are scale-free as constructed. Under no
perturbation does either model's perturbed mean approach the
other's unperturbed mean (Phase 6 range 0.162–0.186; baseline
range 0.196–0.219).

**Findings (diagnostics only, per the restrictions).**

1. **The Phase 6 model's lower real predictions are robust,
   not a sensitivity artefact (Measured, as a stability
   statement).** Its mean |Δ| is at or below the baseline's
   under six of the eight perturbations and within 0.001 on
   the remaining two; the cross-model disagreement is flat
   (0.068–0.072) under every perturbation. The two models
   differ structurally — a stable difference in what they
   learned — not because one of them sits on a knife edge of
   the input spectrum.
2. **What this cannot say (recorded per the ruling).** The
   test establishes stability, not correctness. Both models
   lean on band 0 to a similar degree (a ±1 dB band-0 change
   moves either mean by ≈0.01–0.014), and both could be
   stably wrong in different ways on real backgrounds. No
   model was modified, no threshold selected and no
   calibration derived from these outcomes; the test inputs
   remain unchanged for the eventual submission pipeline.

## Interaction experiment — COMPLETE

The one small combination, run only after the independent
screens: workstream 2's character-synthesis training ×
workstream 1's family D features (605 features, screen scale).
Code: `code/phase7_interact.py`; record:
`models/phase7_interact.json`. (Workstream 3 is not combined:
it was Rejected on its own terms, and combining a rejected
lever is outside the experiment's design.)

| holdout | WS2 × D | WS2 alone | WS1 alone | baseline A |
|---|---|---|---|---|
| random | 0.9435 | 0.9437 | 0.9445 | 0.9434 |
| site | 0.9154 | 0.9141 | 0.9042 | 0.9002 |
| period | 0.9445 | 0.9448 | 0.9452 | 0.9439 |
| long | 0.9353 | 0.9372 | 0.9364 | 0.9381 |

Site zero band: bias +0.027, RMSE 0.043 (WS2 alone +0.028 /
0.044). Site (0.3,0.6] bias −0.086 (WS2 alone −0.085).
Real-window mean 0.1665 (WS2 alone 0.1783, WS1 alone 0.2039).

**Findings.**

1. **The two levers neither compound nor interfere
   (Measured).** Every holdout is within ±0.002 of the
   character-synthesis parent; the site gain is the
   transplant's, carried unchanged. Family D's specific hope
   — defending the high-share side the transplant discounts —
   does not materialise: the (0.3,0.6] under-read is identical
   with and without D. The deviation features and the
   character training act on the same information (the
   window's spectral level relative to a reference), so once
   the training distribution covers the character variation,
   the features add nothing synthetic — though the real mean
   falls further (0.167), the same unresolved direction as
   Phase 6's family C signal.
2. Both configurations therefore go to the full main/A/B
   protocol (the phase's arbiter, per the Phase 6 lesson that
   screens mislead): character synthesis with the frozen 549,
   and character synthesis with 549 + D. The gate judgement
   below rests on those runs, not on the screens.

## Full protocol — character synthesis, frozen 549 features — COMPLETE

Main / A / B models on the character-synthesis train / trainA
/ trainB sets (best iterations 5,999 / 5,288 / 4,315), frozen
protocol, evaluated on the identical frozen holdouts against
the Phase 5 baseline. Record: `models/phase7_full_base.json`;
diagnostic predictions `models/phase7_full_base_test_predictions.csv`.

| holdout | main | A | B | baseline main / A / B |
|---|---|---|---|---|
| random | 0.9476 | 0.9437 | 0.9445 | 0.9482 / 0.9434 / 0.9436 |
| site | **0.9232** | **0.9141** | **0.9186** | 0.9030 / 0.9002 / 0.9051 |
| period | 0.9488 | 0.9448 | 0.9452 | 0.9469 / 0.9439 / 0.9462 |
| long | 0.9417 | 0.9372 | 0.9370 | 0.9422 / 0.9381 / 0.9383 |

A–B agreement RMSE 0.031–0.038, in the baseline family's range
(0.031–0.040). Site holdout by share band, main model (bias /
RMSE / SSE share; baseline main in parentheses): 0: **+0.028
/ 0.045 / 8.5%** (+0.065 / 0.090 / 21.4%); (0,0.05]: +0.021 /
0.044 / 7.3% (+0.053 / 0.084 / 16.9%); (0.05,0.1]: +0.001 /
0.049 / 3.5% (+0.032 / 0.074 / 5.0%); (0.1,0.3]: −0.019 /
0.075 / 21.3% (+0.008 / 0.091 / 19.6%); (0.3,0.6]: −0.062 /
0.125 / 33.2% (−0.034 / 0.131 / 23.1%); (0.6,1]: −0.070 /
0.132 / 26.1% (−0.051 / 0.122 / 14.0%). Real windows: main
mean 0.189 (baseline 0.206); categories clean 750 (50.0%) at
0.107, ambiguous 582 (38.8%) at 0.253, unresolved 168 (11.2%)
at 0.337 — the character models agree less often on real
windows than the baseline family did (unresolved 5.9% →
11.2%) but read lower where they do agree, and their
unresolved mean is lower too (0.337 vs 0.391).

**Findings.**

1. **The site gain replicates at full scale, on all three
   models (Measured).** Site: +0.020 (main), +0.014 (A),
   +0.014 (B). Random and long are within ±0.001 of the
   baseline; period improves slightly (+0.002). The phase's
   central objective is met in the terms it was set: the
   site-holdout zero-share over-read falls from +0.065 to
   +0.028 and its RMSE halves (0.090 → 0.045), with the zero
   band's SSE share down from 21.4% to 8.5%, and no holdout
   sacrificed.
2. **The residual site error is now concentrated at high
   shares (Measured).** The (0.3,0.6] and (0.6,1] bands carry
   59% of site SSE; the (0.6,1] band is the one place the new
   model is worse in RMSE (0.132 vs 0.122). Site calibration
   at the bottom is largely solved within this validation;
   calibration at the top on unseen sites is the remaining
   synthetic deficit — the mirror image of the problem Phase
   7 was set.
3. **The gate is still not met (Measured).** Best holdout is
   period at 0.9488 against the 0.96 development minimum;
   random 0.9476 and long 0.9417 also fall short, and the
   cross-check structure (A and B as independent replications
   of the main configuration) shows the same levels, so this
   is not a single-model artefact. The character lever closes
   about half the site gap to the gate (0.903 → 0.923 of the
   way toward 0.96) and none of the seen-holdout gap.

## Full protocol — character synthesis + family D — COMPLETE

Same protocol with the 605-feature set (best iterations 5,991
/ 5,466 / 2,662). Record: `models/phase7_full_D.json`.

| holdout | main | A | B | (549 config main) |
|---|---|---|---|---|
| random | 0.9470 | 0.9435 | 0.9439 | 0.9476 |
| site | **0.9250** | 0.9154 | 0.9214 | 0.9232 |
| period | 0.9474 | 0.9445 | 0.9446 | 0.9488 |
| long | 0.9396 | 0.9353 | 0.9355 | 0.9417 |

Site zero band, main: bias **+0.025**, RMSE **0.040** — the
best zero-band numbers recorded in any phase (baseline main
+0.065 / 0.090). The (0,0.05] band likewise improves slightly
(RMSE 0.039). The high-share cost is unchanged from the 549
configuration ((0.3,0.6] bias −0.066, (0.6,1] bias −0.070,
(0.6,1] RMSE 0.134 vs the baseline's 0.122). Real windows:
main mean 0.182; clean 733 (48.9%) at **0.095** — the lowest
clean-majority mean of any full-protocol model, level with
the Phase 6 B+C family's 0.094 — ambiguous 616 at 0.237,
unresolved 151 (10.1%) at 0.381.

**Findings.**

1. **At full scale the two configurations are within
   replication noise of each other (Measured).** D is +0.002
   on site, −0.001 to −0.002 on random and long. Family D's
   contribution remains what the screens showed: a small
   further shaving of the low-share site bands and a lower
   real-window read, no synthetic separation. Either
   configuration is a defensible phase output; the 549
   configuration is simpler, the D configuration has the
   best headline site and zero-band numbers.
2. **Neither configuration meets the gate (Measured).**
   Best holdout across both: period 0.9488 and site 0.9250
   against the 0.96 minimum. The remaining synthetic deficit
   is now dominated, on the site holdout, by the high-share
   bands (63% of site SSE in the D configuration) and, on the
   seen holdouts, by the same mid/high-share error that has
   bounded every phase since Phase 5.

## Gate judgement

**The phase's central objective is met; the gate is not.**

Against Godson's objective — reduce zero-share hallucination
on unseen sites without sacrificing the corrected passage
signature or the other three holdouts — the character
synthesis configuration delivers, at full protocol: site
zero-band bias +0.065 → **+0.025** and RMSE 0.090 → **0.040**
(D configuration, main model), the zero band's share of site
SSE down from 21.4% to 7.1%, site holdout 0.9030 → **0.9250**,
with random, period and long within ±0.002 of the Phase 5
baseline on the main models and the A/B replications in
agreement. Workstream 1's features add a small further
low-band improvement at no synthetic cost; workstream 3 is
Rejected with its mechanism recorded; workstream 4 is closed
by Godson's ruling with nothing to incorporate.

Against the unchanged gate — 0.96 development minimum on the
fixed validation and both independent cross-checks — every
holdout of every configuration falls short: best values are
period 0.9488, random 0.9476, site 0.9250, long 0.9417. The
submission standard of about 0.998 under genuinely
independent validation is further away still. **Gate NOT
MET (Measured).** No candidate file has been produced and no
submission made; slots remain 1 of 200 used. The baseline
reference for any future phase is now the Phase 7 character
configuration (its full record: `models/phase7_full_base.json`
and `models/phase7_full_D.json`), with the Phase 5 baseline
retained as the pre-calibration reference.

## Phase 8 recommendation (if Godson approves a further phase)

1. **Bound the blindness caveat first (cheap, decisive).**
   Retrain the screen configuration with Bird Island's shape
   excluded from the transplant targets and score Bird Island
   specifically. If the zero-band correction survives on a
   shape the training never targeted, the character mechanism
   is proven rather than partly assumed; if it collapses, the
   Phase 7 site numbers were substantially memorisation and
   the phase's conclusion must be revised before anything is
   built on it.
2. **Attack the new residual: high-share under-reading on
   unseen sites** (63% of site SSE now sits in the top two
   bands). Candidate levers, each tested independently under
   the same screen discipline: graded transplants (partial
   character strength, interpolating source and target
   shapes, so character is a continuum rather than a swap);
   interpolated shapes between measured sites (character
   diversity beyond the eight medians); and character
   augmentation applied with the addition level jointly
   varied, so the model cannot learn a blanket discount for
   band-0-heavy backgrounds.
3. **The seen-holdout ceiling (~0.948) has not moved in two
   phases.** Phase 6 excluded model class, loss and volume;
   Phase 7's levers were site-directed. The remaining
   untested axis on seen sites is passage-side diversity and
   band-specialised second-stage models on the residuals —
   either is a legitimate Phase 8 workstream, but the
   expected gain is smaller than the site gap that remains
   (0.925 vs 0.96), so site work should keep priority.
4. **Submission posture unchanged.** Even a Phase 8 pass at
   the 0.96 development gate would come to Godson first with
   the full record; nothing in this phase changes the rule
   that no candidate is produced and no slot spent without
   his explicit approval.

---

## Addendum (2026-11-17, appended during Phase 8A) — two reporting reconciliations

Recorded at Godson's direction after his read of this report;
the full reconciliation is in `RESULTS_phase8_exclusion.md`.
Neither item changes any number in this report or any Phase 7
conclusion; both concern wording and citation.

1. **Tolerance wording corrected.** The gate judgement above
   states that random, period and long sit "within ±0.002 of
   the Phase 5 baseline on the main models". That is exact
   for the 549 configuration but overstates the D
   configuration: its long holdout is 0.9396 against the
   baseline's 0.9422, a decline of **0.0026**, outside the
   stated band (its random −0.0012 and period +0.0005 are
   inside). The D section's "−0.001 to −0.002 on random and
   long" (against the 549 configuration) is likewise
   marginally out on long (−0.0021). Corrected wording: the
   ±0.002 band holds for every main-model holdout delta of
   both configurations **except** the D configuration's long
   holdout, which sits 0.0026 below the Phase 5 baseline and
   0.0021 below the 549 configuration.
2. **The 7.1% zero-band SSE share is traced and stands.** The
   gate judgement cites the D configuration's zero-band SSE
   share as 7.1%, a figure the detailed D section does not
   present (the 549 section reports 8.5%). It traces exactly
   to the underlying evaluation record:
   `models/phase7_full_D.json`, `val_sets.val_site.
   bands_main`, band `0`, `sse_share` = 0.0712 (main model),
   and was independently recomputed from the saved booster
   during Phase 8A (0.071). The figure is correct; only its
   citation was missing.
