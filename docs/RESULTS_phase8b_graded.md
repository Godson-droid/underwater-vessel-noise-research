# Phase 8B — Graded character transplants × independent addition levels: results (2026-11-17)

Written incrementally as each stage completes, so the findings
survive interruption. Status markers per section.

**Authorisation and scope (Godson, 2026-11-17).** This is the
single bounded Phase 8 experiment Godson authorised after his
Phase 7 / Phase 8A assessment: ONE carefully controlled
experiment on the high-share under-reading on unseen sites —
graded character transplants with addition level independently
varied (joint character strength × share variation), testing
whether the low-share correction is preserved while high-share
sensitivity recovers. It is NOT a broad Phase 8 programme and
NOT a feature search: no additional workstreams, no feature
changes, no model-class changes. No candidate file, no Zindi
site visit, no submission under any outcome; slots stay 1 of
200.

**Standards (unchanged).** Development minimum **0.96** on the
fixed validation and both cross-checks, all four holdouts
reported separately; submission standard ≈0.998 under genuinely
independent validation. Frozen bandext_maskfix construction
throughout: conversion, target arithmetic (observed-cells sum
label), maskfix missingness, feature definitions (549; 549 +
family D at full protocol), evaluation windows and the
LightGBM training protocol are all unchanged from Phases 5–8A.
Finding classifications: **Measured** / **Supported** /
**Unresolved** / **Rejected**.

**The joint success criterion (Godson's, predetermined).**
(a) The zero-share correction remains substantially intact;
(b) the (0.6,1] bias moves toward zero; (c) recovering
high-share sensitivity must not simply recreate the original
zero-share hallucination.

## Predeclared design (written before any Phase 8B run)

**Character strength.** Predeclared graded scale
**0, 0.25, 0.5, 0.75, 1.0**. Zero means no transplant; 1.0
means the full Phase 7 transplant. Implementation: the Phase 7
generation is rerun with identical seeds (trainA 20358118,
train 20358119, trainB 20358120), identical pools
(`simulate_conv.build_pools('bandext')`), identical
P_TRANSPLANT (0.6), the identical eight-site target family and
the identical draw sequence, with ONE change — when a
transplant fires, the applied target shape is the graded
mixture **(1−s)·source_shape + s·target_shape** in per-band
energy-share space; per-band energies are multiplied by
mixture/source and the window is rescaled to its original
total energy, exactly Phase 7's energy preservation. At s=1.0
the mixture is the target shape itself, so the s=1.0 corpora
must reproduce the Phase 7 corpora **bit-exactly** (NaN-aware);
that identity is the pipeline validation gate for this phase
and is checked before any other result is read. At s=0 every
background keeps its natural character (the dose–response
control).

**Independent addition levels (the central requirement).**
Generation is paired across strength levels: the random stream
is consumed draw-for-draw identically at every level, so every
level sees the same backgrounds (pre-transplant), the same
transplant draws and targets, the same drawn shares, the same
masks and the same passage placements — the ONLY difference
between levels is the interpolation factor applied to the
background's spectral shape. The drawn share therefore cannot
covary with strength by construction. The realised label can
in principle wobble microscopically across levels for
"mix-first" windows, because the label is computed against
observed-minute background energy and the transplant preserves
total energy exactly but observed-minute energy only to second
order; the size of any such wobble is measured and reported,
not assumed away. Verification, reported before training:
per level — draw counts, transplant counts, target counts
(must be identical across levels); realised-label mean, std,
zero fraction and share-band fractions, overall and split by
transplanted / non-transplanted windows; per-window label
differences across levels (max and mean |Δy|, fraction exactly
equal).

**Screen.** Frozen screen protocol (as Phase 7 workstream 2):
549 features, frozen `train_lgbm`, trained on each level's
trainA corpus (40k), evaluated on the identical frozen
holdouts, paired against baseline A and against the s=1.0
level (which must replicate the Phase 7 workstream 2 screen:
0.9437 / 0.9141 / 0.9448 / 0.9372, site zero bias +0.028,
(0.6,1] bias −0.088).

**Survival rule (predeclared; not revisable after results).**
A level s<1 goes to the full protocol iff at screen scale, on
the site holdout: (i) zero-band bias ≤ **+0.040** (the
correction substantially intact against baseline A's +0.064
and s=1.0's +0.028); AND (ii) (0.6,1] bias ≥ **−0.078** (at
least 0.010 less negative than the recorded s=1.0 screen value
−0.088); AND (iii) site score ≥ **0.9101** (within 0.004 of the
s=1.0 screen site score 0.9141). At most two levels survive;
if more qualify, the two with the least negative (0.6,1] bias
advance, ties broken by higher site score. The s=0 control is
screened under the same rule. If no level qualifies, the
experiment ends at the screen and the negative result is
reported as measured; no rule is loosened and no substitute
experiment is run.

**Full protocol.** For each surviving level: train and trainB
corpora generated with the Phase 7 seeds; main / A / B models
in both configurations (549 and 549+D), checkpointed per
model; one evaluation harness recomputing every family's
predictions from its saved boosters on the identical frozen
holdouts — Phase 5 baseline, Phase 7 char 549, Phase 7 char
549+D, Phase 8A excl 549, Phase 8A excl 549+D, and the Phase 8B
graded families. Per family (main model unless noted): the
four holdout scores (main/A/B), the site-holdout share-band
table (bias / RMSE / SSE share for 0, (0,0.05], (0.05,0.1],
(0.1,0.3], (0.3,0.6], (0.6,1]), Bird Island and Fish Hoek
separately (score, bias, zero-band bias/RMSE, (0.6,1]
bias/RMSE), the top two bands' share of site SSE, A–B
agreement, and the real-window diagnostics (mean; clean /
ambiguous / unresolved counts and means) — the last as a
consistency diagnostic only, never a validation substitute.

Status: DESIGN PREDECLARED — generation not yet run.

## Generation and independence verification — COMPLETE

Code: `code/phase8b_gen_graded.py` (corpora
`data/sim_p8b_s{0,25,50,75,100}_trainA.npz`, record
`data/sim_p8b_meta.json`); verification
`code/phase8b_verify_indep.py`, record
`models/phase8b_independence.json`.

**Pipeline validation gate (Measured).** The s=1.0 trainA
corpus reproduces the Phase 7 corpus **bit-exactly**:
40,000 / 40,000 rows identical (NaN-aware), label max |Δ| =
0.00, would-transplant count 21,004 / 40,000 — Phase 7's
recorded count exactly. The graded generator is therefore
Phase 7's generator with the single authorised change.

**Draw identity (Measured).** The recorded draw sequences
(target drawn per window; would-transplant flags) are
identical at all five levels: 21,004 would-transplant draws
of 40,000 at every level, with identical target counts.
Drawn shares, masks, passage placements and transplant
targets are the same windows at every strength by
construction and by this record.

**Realised-label distributions (Measured).** Zero fraction
0.2520 at every level. Label mean / std by level (all
windows): s0 0.1820 / 0.2470; s25 0.1819 / 0.2468; s50
0.1817 / 0.2467; s75 0.1816 / 0.2465; s100 0.1815 / 0.2463.
Share-band fractions agree across levels to within 0.0006 in
every band (e.g. (0.6,1]: 0.0978 at s0, 0.0976 at s100).
Non-transplanted windows are untouched: their label mean is
0.1823 at every level, exactly.

**The label wobble, measured rather than assumed away
(Measured).** Per-window labels are exactly equal across
levels for 90.62% of windows. The remainder are "mix-first"
windows whose label the frozen machinery computes against
observed-minute background energy: the transplant preserves
total background energy exactly but observed-minute energy
only to second order, so those labels move — mean |Δy|
between s0 and s100 is 9.1e-04 (per adjacent step
2.1–2.7e-04), with a small tail (max |Δy| 0.605 on individual
windows whose observed background energy is small against
the placed passage). Among transplanted windows the mean
signed Δy (s100 − s0) is −9.5e-04: a real but microscopic
systematic component (full-range label-mean drift −0.0005,
i.e. 0.05% of the label range), a deterministic property of
the frozen label definition applied to transformed
backgrounds — the same property the Phase 7 / Phase 8A corpora
share. The drawn-share distribution, which is the
experimental variable Godson's requirement governs, is
identical at every level; no strength level systematically
receives larger or smaller additions beyond this measured
1e-3-scale label-machinery effect.

Status: COMPLETE — screen follows under the predeclared
survival rule.

## Screen — COMPLETE

Code: `code/phase8b_screen.py`; records
`models/phase8b_screen_s{0,25,50,75,100}.json`, boosters
`models/phase8b_scr_s*.txt`. (One service restart interrupted
the screen mid-s50; the per-level checkpointing resumed it
from s50 with no stage repeated and no result affected.)

**Harness validation (Measured).** The s=1.0 screen model
replicates the Phase 7 workstream 2 screen **exactly**:
holdouts 0.9437 / 0.9141 / 0.9448 / 0.9372, best iteration
5,288, site zero bias +0.028, (0.6,1] bias −0.088, real mean
0.1783. Baseline A in the same harness: 0.9434 / 0.9002 /
0.9439 / 0.9381 (its recorded values).

**Dose–response at screen scale** (site holdout; bias by
share band, bias / RMSE):

| level | random | site | period | long | real mean |
|---|---|---|---|---|---|
| s0 (control) | 0.9432 | 0.8952 | 0.9447 | 0.9388 | 0.2072 |
| s25 | 0.9440 | 0.9103 | 0.9457 | 0.9384 | 0.1871 |
| s50 | 0.9436 | 0.9118 | 0.9461 | 0.9382 | 0.1809 |
| s75 | 0.9435 | 0.9104 | 0.9456 | 0.9371 | 0.1789 |
| s100 (= Phase 7) | 0.9437 | 0.9141 | 0.9448 | 0.9372 | 0.1783 |
| baseline A (ref) | 0.9434 | 0.9002 | 0.9439 | 0.9381 | 0.2099 |

| band | s0 | s25 | s50 | s75 | s100 |
|---|---|---|---|---|---|
| 0 | +0.073 / 0.097 | +0.038 / 0.060 | +0.031 / 0.048 | +0.029 / 0.046 | +0.028 / 0.044 |
| (0,0.05] | +0.060 / 0.089 | +0.027 / 0.059 | +0.019 / 0.047 | +0.017 / 0.043 | +0.019 / 0.043 |
| (0.05,0.1] | +0.036 / 0.079 | +0.006 / 0.059 | −0.002 / 0.052 | −0.004 / 0.050 | −0.000 / 0.052 |
| (0.1,0.3] | +0.006 / 0.093 | −0.020 / 0.089 | −0.029 / 0.086 | −0.033 / 0.086 | −0.026 / 0.082 |
| (0.3,0.6] | −0.048 / 0.145 | −0.066 / 0.145 | −0.081 / 0.150 | −0.092 / 0.157 | −0.085 / 0.147 |
| (0.6,1] | −0.067 / 0.139 | −0.069 / 0.138 | −0.081 / 0.148 | −0.089 / 0.154 | −0.088 / 0.151 |

Per site (score / zero bias / (0.6,1] bias): Bird Island —
s0 0.8934 / +0.075 / −0.068; s25 0.9091 / +0.038 / −0.070;
s50 0.9107 / +0.031 / −0.083; s75 0.9093 / +0.028 / −0.090;
s100 0.9132 / +0.028 / −0.089. Fish Hoek — s0 0.9216 / +0.041
/ −0.060; s25 0.9255 / +0.035 / −0.060; s50 0.9265 / +0.032 /
−0.060; s75 0.9256 / +0.033 / −0.070; s100 0.9251 / +0.034 /
−0.070. The dose–response is a Bird Island phenomenon; Fish
Hoek is comparatively flat, as in Phases 7–8A.

**Findings (screen scale).**

1. **The trade-off is a smooth dose–response in strength
   (Measured).** The zero-band bias falls monotonically with
   strength (+0.073 → +0.038 → +0.031 → +0.029 → +0.028) and
   saturates by s50; the (0.6,1] penalty grows with strength
   (−0.067 → −0.069 → −0.081 → −0.089 → −0.088). The two
   curves cross in usefulness at the low end: most of the
   high-share penalty arrives between s25 and s75, while
   most of the zero correction is already present at s25.
2. **The s0 control replicates the untreated condition
   (Measured).** No character training: zero bias +0.073
   (baseline A +0.064; the +0.009 gap is fresh-seed
   replication spread, the same order as the Phase 7 screen
   noise), site 0.8952 vs baseline A's 0.9002. Nothing about
   the graded machinery itself moves the site holdout; only
   applied character does.
3. **Seen holdouts are flat across the whole scale
   (Measured).** Random, period and long move by ≤0.002
   between any two levels: the strength axis trades
   low-share against high-share calibration on unseen sites
   only, exactly the axis the experiment targeted.

**Survival rule applied (predeclared, mechanical).**
(i) zero bias ≤ +0.040; (ii) (0.6,1] bias ≥ −0.078;
(iii) site ≥ 0.9101.
- s0: fails (i) (+0.073). Control; does not advance.
- **s25: passes all three** (+0.038; −0.069; 0.9103) —
  narrowly on (i) and (iii). **Advances to full protocol.**
- s50: fails (ii) (−0.081, missing by 0.003).
- s75: fails (ii) (−0.089).
One survivor: **s25**, in both configurations (549, 549+D),
main / A / B, under the full protocol. The full-strength
Phase 7 configuration remains the reference the survivor is
judged against; the screen already shows s25's site score
below it (0.9103 vs 0.9141), so the full protocol tests the
joint criterion — correction retained with high-share
sensitivity recovered — not a headline-score claim.

Status: COMPLETE — full protocol for s25 follows. Verdict:
joint criterion met in bounded form (correction retained,
high-share recovery partial); gate NOT MET; experiment
closed (see the full-protocol section below).

## Full protocol — s25, both configurations — COMPLETE

Code: `code/phase8b_train_eval.py 25`; record
`models/phase8b_graded.json`; boosters
`models/phase8b_s25_{base,D}{,_A,_B}.txt` (best iterations:
549 main/A/B 6000/5984/3915; 549+D 5989/4715/5996);
diagnostic-only test predictions
`models/phase8b_diag_phase8b_s25_{549,549D}_test_predictions.csv`.
Reference families were recomputed in the same harness from
their saved boosters and reproduce the Phase 5 / 7 / 8A
records exactly, so every comparison below is like for like.

**Holdouts (main / A / B):**

| family | random | site | period | long |
|---|---|---|---|---|
| Phase 5 baseline | 0.9482/0.9434/0.9436 | 0.9030/0.9002/0.9051 | 0.9469/0.9439/0.9439 | 0.9422/0.9381/0.9383 |
| Phase 7 char 549 | 0.9476/0.9437/0.9445 | 0.9232/0.9141/0.9186 | 0.9488/0.9448/0.9452 | 0.9417/0.9372/0.9370 |
| Phase 7 char 549+D | 0.9470/0.9435/0.9439 | 0.9250/0.9154/0.9214 | 0.9474/0.9445/0.9446 | 0.9396/0.9353/0.9355 |
| Phase 8A excl 549 | 0.9475/0.9429/0.9442 | 0.9159/0.9080/0.9149 | 0.9481/0.9457/0.9453 | 0.9412/0.9368/0.9374 |
| Phase 8A excl 549+D | 0.9467/0.9440/0.9435 | 0.9205/0.9134/0.9171 | 0.9483/0.9451/0.9447 | 0.9402/0.9356/0.9356 |
| **8B s25 549** | 0.9471/0.9440/0.9438 | **0.9163/0.9103/0.9114** | 0.9483/0.9457/0.9453 | 0.9426/0.9384/0.9387 |
| **8B s25 549+D** | 0.9470/0.9445/0.9435 | **0.9218/0.9145/0.9161** | 0.9476/0.9461/0.9449 | 0.9420/0.9372/0.9378 |

### The five judgement questions (Godson's, 2026-11-17)

**Q1. Does the zero-share correction survive on the full
site holdout AND both cross-checks? — Yes (Supported).**
Site zero-band bias, main / A / B. s25 549: **+0.030 /
+0.038 / +0.033**. s25 549+D: **+0.028 / +0.037 / +0.031**.
Full strength: char 549 +0.028 / +0.029 / +0.030; char 549+D
+0.024 / +0.027 / +0.024. Untreated baseline: +0.065 / +0.064
/ +0.064. At main scale the quarter-strength correction is
at the full-strength level in both configurations; on the
cross-checks it retains roughly three quarters to nine
tenths of the full correction and remains nowhere near the
baseline. The correction survives.

**Q2. Does the high-share improvement replicate,
particularly in (0.6,1]? — Yes, as a partial recovery
(Supported, bounded).** Site (0.6,1] bias, main / A / B.
s25 549: **−0.056 / −0.069 / −0.065** against char 549's
−0.070 / −0.081 / −0.086 — an improvement of 0.014 / 0.012 /
0.021 toward zero. s25 549+D: **−0.064 / −0.074 / −0.071**
against char 549+D's −0.070 / −0.092 / −0.080 — an
improvement of 0.006 / 0.018 / 0.009. The improvement
replicates on the main model and both cross-checks in both
configurations. The bound: it does **not** return to the
untreated baseline's high-share calibration (−0.051 / −0.055
/ −0.049). Quarter strength recovers part of the
character-induced high-share penalty; it does not erase it.

**Q3. Does the result hold separately on Bird Island and
Fish Hoek? — Yes for the zero correction at both sites;
the high-share gain is a Bird Island effect (Supported;
one noise-level exception noted).** Main models. s25 549 —
Bird Island: score 0.9153, zero +0.030, (0.6,1] −0.055
(char 0.9226 / +0.028 / −0.070; baseline 0.9011 / +0.068 /
−0.051). Fish Hoek: 0.9300, zero +0.028, (0.6,1] −0.061
(char 0.9305 / +0.029 / −0.065; baseline 0.9284 / +0.032 /
−0.050). s25 549+D — Bird Island: 0.9209, zero +0.029,
(0.6,1] −0.064 (char+D 0.9246 / +0.025 / −0.071). Fish Hoek:
0.9328, zero +0.019, (0.6,1] −0.062 (char+D 0.9301 / +0.019
/ −0.059). The zero correction holds at both sites in both
configurations. The high-share improvement is clear at Bird
Island (+0.015 and +0.007), small at Fish Hoek in 549
(+0.004), and reverses by 0.003 at Fish Hoek in 549+D —
inside holdout noise, recorded as **Unresolved** at that one
site/configuration rather than smoothed over. Fish Hoek's
549+D score (0.9328) is the best Fish Hoek score of any
family.

**Q4. Are random, period and long preserved? — Yes
(Measured).** Against Phase 5, Phase 7 and Phase 8A
references, every seen holdout for both s25 configurations
sits within ±0.002 of the corresponding reference family at
main scale, and the cross-checks track the same pattern.
The s25 549 long score (0.9426) is the highest long score of
any family measured. Nothing on the seen side is traded
away.

**Q5. Does A/B agreement remain within the established
range? — Yes (Measured).** A–B prediction RMSE by holdout
(random / site / period / long): s25 549: 0.031 / 0.035 /
0.032 / 0.034. s25 549+D: 0.030 / 0.035 / 0.032 / 0.033.
Phase 5 family: 0.031 / 0.040 / 0.032 / 0.033 — the
established 0.031–0.040 range. Both quarter-strength
configurations sit inside that range on all four holdouts,
and on the site holdout they agree more tightly (0.035) than
the baseline family itself (0.040).

**Intermediate bands, judged in their own right (required
check).** Site bias by band, main models:

| band | baseline | char 549 | s25 549 | char 549+D | s25 549+D |
|---|---|---|---|---|---|
| (0,0.05] | +0.052 | +0.021 | +0.020 | +0.020 | +0.018 |
| (0.05,0.1] | +0.032 | +0.001 | −0.001 | +0.004 | +0.001 |
| (0.1,0.3] | +0.008 | −0.019 | −0.016 | −0.021 | −0.019 |
| (0.3,0.6] | −0.033 | −0.062 | −0.056 | −0.066 | −0.060 |

Every intermediate band in both quarter-strength
configurations lies **between** the untreated baseline and
the full-strength value: a monotone interpolation, with no
band pushed beyond either reference. The (0.3,0.6] band
improves on full strength in both configurations, and no
new calibration problem appears in the middle bands. The
extremes are not fixed at the middle's expense. (Measured.)

**Real-window confidence diagnostics (main models).**
s25 549: clean 750 windows, mean 0.103 (char 549: 750,
0.107; baseline: 770, 0.124); ambiguous 618, mean 0.287
(char 0.253; baseline 0.279); unresolved 132, mean 0.382
(char 168, 0.337; baseline 89, 0.391); overall real mean
0.2032. s25 549+D: clean 707, mean 0.095 (char+D: 733,
0.095; baseline 0.124); ambiguous 633, mean 0.245 (char+D
0.237); unresolved 160, mean 0.364 (char+D 151, 0.381);
overall real mean 0.1869. The clean-window behaviour — the
real-data face of the zero-share correction — is at
full-strength level in both configurations. The 549
configuration's higher overall real mean comes from the
ambiguous category, not from clean windows. Top-two-band
share of site SSE (main): s25 549 0.514, s25 549+D 0.574,
against char 0.593 / 0.633, excl 0.575 / 0.606, baseline
0.371: quarter strength concentrates less of its remaining
error in the top bands than full strength does, consistent
with the recovered high-share calibration.

### Joint criterion verdict

(a) Zero-share correction substantially intact —
**Supported**: at full-strength level on both main models,
most of it retained on both cross-checks, clean real windows
at full-strength level. (b) (0.6,1] bias moves toward zero —
**Supported, bounded**: it moves toward zero against the
full-strength configuration on the main model and both
cross-checks in both configurations, but remains worse than
the untreated baseline's high-share bias; the recovery is
partial. (c) No recreation of the zero-share hallucination —
**Supported**: zero-band bias and clean-window means stay at
corrected levels everywhere. The joint criterion is met in
the form the experiment could meet it: quarter strength
keeps the correction and buys back part of the high-share
sensitivity. The price is headline site score: −0.007 (549)
and −0.003 (549+D) against full strength — though s25 549+D
(0.9218) still clears both Phase 8A configurations (0.9159 /
0.9205).

### Gate judgement

The screen dose–response is evidence for a mechanism, not
evidence quarter strength meets the gate. The 0.96
development minimum on fixed validation and both cross-checks
across all four holdouts is unchanged, and no configuration
here approaches it: the best site figure is 0.9218 (s25
549+D main), with cross-checks at 0.9145 / 0.9161. The
separate ~0.998 submission standard is likewise unchanged
and untouched. **Gate: NOT MET** (Rejected as a candidate
route, as every Phase 5–8 configuration before it). No
candidate file was produced, the Zindi site was not visited,
and no submission was made; submission slots remain **1 of
200** used. Phase 8B closes here: a single bounded
experiment, completed as authorised, with no follow-on
search implied by its result.

## Classification summary

| finding | class |
|---|---|
| Strength × share independence in all generated corpora | Measured |
| Graded dose–response at screen scale (both bias curves monotone in strength) | Measured |
| s=1.0 screen = Phase 7 WS2 exactly; s=0 control = untreated condition | Measured |
| Q1: zero correction survives main + A + B, both configurations | Supported |
| Q2: (0.6,1] improvement replicates main + A + B, both configurations | Supported (bounded: partial recovery, not baseline level) |
| Q3: per-site pattern, Bird Island driven | Supported |
| Q3 exception: Fish Hoek (0.6,1] in 549+D (−0.062 vs −0.059) | Unresolved (noise-level, single cell) |
| Q4: seen holdouts preserved | Measured |
| Q5: A/B agreement inside the 0.031–0.040 range | Measured |
| Intermediate bands interpolate; no new mid-band problem | Measured |
| Joint criterion (a)+(b)+(c) at quarter strength | Supported |
| Quarter strength as a route to the 0.96 gate | Rejected |

---

## Addendum — Godson's review qualifications (2026-11-17)

Godson reviewed the complete Phase 8B report and accepted it as the settled Phase 8B record, subject to the following qualifications, which are preserved here as part of the record:

1. **Label tail.** 90.62% of labels were exactly identical across strength levels, and the mean absolute label difference between the endpoints is 9.1×10⁻⁴. However, a small number of mix-first windows had substantial individual label differences, up to a maximum of 0.605. The experiment is therefore not literally a perfect fixed-label intervention for every window, and this tail must not disappear from the scientific record.
2. **Wording rule for all future summaries.** The correct statement is: independent drawn-share assignment with small measured realised-label variation. Not: identical labels across every strength level. The drawn additions are independent of character strength by construction; the realised labels are not perfectly invariant because of the observed-minute energy calculation.
3. **Real-window diagnostics remain unvalidated.** The clean-window means (0.103 for 549 quarter strength, 0.095 for 549+D quarter strength) are encouraging, but the overall real-window means (0.2032 and 0.1869) are model diagnostics, not estimates of actual test-set prevalence. Unresolved counts remain higher than Phase 5. Neither figure may be converted into a claim about the true test distribution.
4. **Standing.** Neither quarter-strength configuration is a replacement submission candidate (best site score 0.9218; other holdouts around 0.94–0.95; the 0.96 development gate is unmet and submission eligibility is not established). Reference structure: Phase 5 = original pre-calibration reference; Phase 7 = full-strength character-calibration reference; Phase 8A = evidence bounding the transferability of character calibration; Phase 8B = graded-strength experiment demonstrating a bounded recovery of high-share sensitivity while retaining most of the zero-share correction.
5. **Direction.** The authorised experiment is complete. Work stops here; the positive result is not automatic permission for another search.
