# Phase 5 — passage conversion investigation: results (2026-11-16)

Written incrementally as each workstream completes, so the findings
survive interruption. Status markers per section.

**Standards (Godson, binding).** Development minimum for this
iteration: **0.96**. Submission standard: **≈0.998 under genuinely
independent validation** — a candidate file is produced only if a
variant reaches the standard on the fixed validation AND on both
independent cross-checks (A and B, disjoint training splits). All four
holdouts (random, site, period, long) are reported separately, never
aggregated away. Paired comparisons against v3 use identical windows.
The real-window prediction distribution is a consistency diagnostic
only; its aggregate mean is not a substitute for ground truth.
Natural vessel traffic in a background is never counted toward the
label. No submission is made under any outcome of this phase.

**Protocol inheritance.** Everything not under test is frozen at v3:
split discipline (seed 20261116), share prior, background modes
(clean / busy / mixtures + jitter + drift), observed-cells label,
549-feature transform, LightGBM hyperparameters, 6000-round cap with
the same early stopping. Test.csv is inference only; nothing is
fitted, calibrated or pseudo-labelled from it.

**Finding classifications** (Godson's requirement): **Measured** —
directly demonstrated by an experiment; **Supported** — consistent
with multiple independent results; **Unresolved** — plausible but not
established; **Rejected** — contradicted by the available evidence.

---

## Workstream 2 — Target arithmetic audit — COMPLETE

Code: `code/audit_labels.py`; record: `models/audit_labels.json`.
Method: on the v3 oracle set (2,000 windows with the true composite
background stored), each candidate label definition L is scored by
ceiling(L) = 1 − RMSE(oracle estimate, L), where the oracle estimate
is the fixed true-background residual estimator. A definition whose
ceiling falls below the real top score 0.9943 cannot be the
organisers' definition. The observed-cells vs full-window question
was settled in Phase 4 and is not redone; L0 below is the adopted
observed-cells rule (sum of clipped added energy over observed cells
/ total energy over observed cells).

| definition | ceiling | label mean | RMSE vs v3 label | implied Var+(μ−0.1)² (real: 0.060581) | verdict |
|---|---|---|---|---|---|
| L0 control (sum, observed cells) | 1.0000 (estimator identity) | 0.1829 | 0.0017 | 0.0670, consistent | survives |
| L_round (target computed after 0.1 dB rounding) | 0.9982 | 0.1817 | 0.0008 | 0.0671, consistent | survives |
| L_cellmean (mean of per-cell shares) | 0.7756 | 0.2638 | 0.2247 | 0.0801, inconsistent | **RULED OUT** |
| L_minmean (mean of per-minute shares) | 0.8394 | 0.1760 | 0.1607 | 0.0491, inconsistent | **RULED OUT** |
| L_lowband (share within ≤1 kHz bands only) | 0.9351 | 0.1462 | 0.0646 | 0.0505, inconsistent | **RULED OUT** |
| L_amp20 (amplitude domain, dB/20) | 0.8916 | 0.1946 | 0.1086 | 0.0549, consistent on moments only | **RULED OUT** |

Notes. (i) The L0 ceiling of 1.0000 is an identity (the oracle
estimator *is* the L0 statistic computed from the true background);
the meaningful L0 number is the oracle against the stored v3 labels:
**0.9983**, the Phase 4 construction ceiling. (ii) L_round differs
from L0 by RMSE 0.0008 — whether the organisers computed the target
before or after rounding is **immaterial** at any achievable accuracy;
the two definitions are indistinguishable for modelling. (iii) Every
mean-based, band-restricted and amplitude-domain definition is
rejected twice over: its ceiling is below the real top score *and*
its implied label moments contradict the two public scores.

**Answer to the workstream question.** The target calculation
consistent with the organisers' process and the achievable ceiling is
a **sum-based share over the observed cells** — exactly the v3
construction. Rounding order is unresolved but immaterial. All other
audited definitions are **Rejected (Measured)**. Target arithmetic is
therefore not the location of the transfer gap: no label correction
remains that could move a model from 0.93 toward 0.998.

---

## Workstream 3 — Missingness dependence audit — COMPLETE

Code: `code/audit_missing.py`; record: `models/audit_missing.json`.
Design: 12,000 paired synthetic windows per arm (same seed per arm:
same backgrounds, shares and events; only the mask mechanism
differs), scored by the frozen v3 main model, plus the same
statistics on the 1,500 real windows (inference and structure only).
Arms: **indep** (v3's mask: per-minute Bernoulli drops from the
measured pattern, cap 3), **event** (same drop count, minutes chosen
with probability proportional to 0.25 + the placed event's per-minute
energy), **bg** (same drop count, minutes chosen by background
quietness).

**Pattern distribution (fraction of windows).**

| group | real | indep | event | bg |
|---|---|---|---|---|
| complete | 0.517 | 0.534 | 0.544 | 0.543 |
| minute 1 only | 0.413 | 0.387 | 0.016 | 0.025 |
| minute 15 only | 0.065 | 0.040 | 0.018 | 0.031 |
| other single minute | **0.000** | 0.005 | 0.389 | 0.369 |
| ≥2 missing | 0.005 | 0.034 | 0.034 | 0.032 |

The real mask's *form* is closest to the independent arm: single
drops are almost exclusively minute 1 or minute 15. The event and bg
mechanisms, which spread drops across all minutes, are flatly
inconsistent with the real pattern distribution. Two deviations from
the independent arm remain: real windows show **zero** "other single"
drops (the independent arm predicts ≈7–8 in 1,500) and far fewer ≥2
drops (0.5% vs 3.4%) — the real process is more concentrated on
minute 1 than v3's measured-pattern Bernoulli.

**The discriminating contrast.** Mean observed energy of minutes
2–15 in minute-1-missing windows minus complete windows:

| real | indep | event | bg |
|---|---|---|---|
| **+8.65 dB** | −0.35 dB | −1.00 dB | −1.80 dB |

No arm comes close. In the real data, minute-1-missing windows are
built from backgrounds roughly 7× more energetic than complete
windows (their total energy is 118.3 dB vs 110.0 dB despite holding
one fewer minute); minute-15-missing windows are not louder (109.1
dB). Every simulated mechanism leaves the surviving minutes'
energy unchanged or slightly lower. **Measured:** the real
minute-1-missing sub-population (41% of the test set) comes from a
louder background population, or through a different gain/export
path, than the complete windows — a localised domain shift that is
*correlated with* the mask but is not produced *by* any mask
mechanism we tested.

**Effect on the model.** The frozen v3 model reads real
minute-1-missing windows at 0.376 vs 0.311 for complete windows. If
the minute-1 group read at the complete-group level, the overall
real mean would fall from 0.337 to ≈0.310 — so this anomaly accounts
for roughly 0.03 of the ≈0.16 over-read. The complete windows alone
still read 0.311, far above the 0.13–0.20 band. **The bulk of the
transfer gap is not missingness-related.**

**Answer to the workstream question.** Does event-dependent
missingness explain additional target information? **Rejected
(Measured)** in the tested form: the event mechanism predicts the
wrong pattern distribution and the wrong sign of the energy
contrast, and the real mask's form is that of an independent,
minute-specific process. The audit did find a real anomaly the
simulator does not reproduce — the loud minute-1 sub-population —
which is **Supported** as a minor contributor to the over-read and
as further evidence of a background-population/export difference
(workstream 4's territory), and **Unresolved** as to its exact
origin (louder source recordings with genuine first-minute gaps vs
a different export gain for those windows).

**Follow-up after workstream 1 (Measured).** The bandext model's
real-window readings by pattern (`code/audit_missing_bandext.py`,
record `models/audit_missing_bandext.json`): complete 0.2124
(n=775), minute-1-only 0.2199 (n=620), minute-15-only 0.2321
(n=97), ≥2 0.2176 (n=8). The minute-1 group's excess over
complete windows collapses from +0.065 under v3 to **+0.0075**
under bandext. The loud minute-1 sub-population was therefore
mostly another symptom of the edge-band conversion error — those
windows' energy is concentrated where the clipped conversion
distorted most — not a separate missingness phenomenon. What
remains for the maskfix cell to fix is the mask *form* in
training, not a real-window anomaly.

## Workstream 1 — Passage conversion variants — COMPLETE

Machinery verification (**Measured**): the base configuration
(hann 2 s segments, constant detrend, mean aggregation) re-converted
all 542 clips and reproduced `passages.npz` with max |Δ rel_db| =
**0.00 dB** — the variant pipeline is bit-exact against the original
harvest, so every difference below is the variant's mathematics, not
pipeline drift. Variants (exactly one axis changed per variant; the
notebook's band grid and integration are kept unless the variant is
the band axis): **magmean** — magnitude aggregation, per minute and
band (mean over segments of √power)² [assessment priority 1];
**dbmean** — aggregation by averaging segment dB values [priority
2]; **median** — median of segment powers [priority 2 family];
**rect** — rectangular instead of hann window [priority 3];
**rawfft** — no constant detrend [priority 3 family]; **seg5 /
seg10 / min60** — 5 s / 10 s / 60 s segments with matching hop
[priority 3]; **welch1s** — 1 s segments, 0.5 s hop [priority 3];
**bandext** — band 0 extended to the lowest bin and band 24 to
Nyquist [priority 4]. A density-vs-spectrum scaling variant is
deliberately not run: any frequency-independent constant cancels
under the relative-dB normalisation (recorded reasoning, assessment
§2). Exact third-octave (2^(1/3)) edges differ from the notebook's
deci-decade edges by 0.04% in width — arithmetically negligible, not
run (recorded in `code/convert_variants2.py`).

Screening protocol: one model per variant on the variant's trainA
(40k windows, v3 split discipline, fresh recorded seed per variant),
scored on its own four holdouts, cross-scored against the v3 A model
in both directions on identical windows, with the variant
construction's oracle ceiling and the real-window prediction
distribution. Full A/B/main runs follow for the strongest variants
only. Reference points: v3 A model scores 0.9235 / 0.9194 / 0.9233 /
0.9085 on the v3 holdouts (random / site / period / long); the v3
family reads real windows at mean 0.335–0.339; the plausible prior
band for the real mean is 0.13–0.20.

**Screening results** (own holdouts in v3 order; "paired" = variant
model vs v3 A model on identical v3 windows, same order; real =
real-window prediction mean of the variant's screen model):

| variant | own holdouts | paired on v3 windows (variant vs v3 A) | ceiling | real mean |
|---|---|---|---|---|
| magmean | 0.9277 / 0.9240 / 0.9323 / 0.9186 | 0.9224 / 0.9193 / 0.9241 / 0.9069 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3277** |
| dbmean | 0.9263 / 0.9257 / 0.9307 / 0.9202 | 0.9200 / 0.9166 / 0.9192 / 0.9044 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3187** |
| rect | 0.9232 / 0.9146 / 0.9200 / 0.9129 | 0.9220 / 0.9124 / 0.9227 / 0.9082 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3360** |
| rawfft | 0.9280 / 0.9247 / 0.9292 / 0.9162 | 0.9223 / 0.9207 / 0.9230 / 0.9098 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3333** |
| seg5 | 0.9236 / 0.9188 / 0.9238 / 0.9094 | 0.9223 / 0.9158 / 0.9218 / 0.9072 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3342** |
| seg10 | 0.9239 / 0.9226 / 0.9247 / 0.9094 | 0.9231 / 0.9211 / 0.9238 / 0.9083 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3361** |
| welch1s | 0.9253 / 0.9219 / 0.9244 / 0.9105 | 0.9219 / 0.9172 / 0.9226 / 0.9087 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3320** |
| median | 0.9293 / 0.9267 / 0.9336 / 0.9209 | 0.9215 / 0.9195 / 0.9230 / 0.9037 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9983 | **0.3238** |
| bandext | 0.9445 / 0.8911 / 0.9459 / 0.9392 | 0.8384 / 0.8229 / 0.8393 / 0.7897 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9982 | **0.2042** |
| min60 | 0.9243 / 0.9200 / 0.9266 / 0.9133 | 0.9232 / 0.9200 / 0.9233 / 0.9079 vs 0.9235 / 0.9194 / 0.9233 / 0.9085 | 0.9982 | **0.3349** |

magmean (assessment priority 1, power vs magnitude): synthetic
behaviour is indistinguishable from the base conversion (paired
differences within ±0.002 on identical windows), and the real mean
moves from 0.337 to 0.328 — a shift of 0.009 against a gap of ≈0.16.
**Measured: magnitude aggregation is not the organisers' differing
choice in any way that matters; it does not explain the transfer
failure.**

dbmean (assessment priority 2, averaging in dB vs averaging linear
power): the largest real-mean movement of any variant so far, 0.337
→ **0.319**, with synthetic scores level with the base conversion on
its own holdouts. But its model is *worse* than v3 A on identical v3
windows (paired −0.003…−0.004 on every holdout), and 0.319 remains
far outside the 0.13–0.20 band. A −0.018 shift against a ≈0.16 gap,
bought with slightly worse synthetic fidelity, is a nudge in the
right direction, not an explanation: **Supported** as the only
conversion axis that moves the real distribution at all;
**Rejected** as a sufficient account of the failure.

rect (window function, hann → rectangular): real mean 0.336,
synthetic scores slightly below the base conversion on its own
holdouts. **Measured: the window function is not the differing
choice either.**

rawfft (no constant detrend): real mean 0.333, synthetic scores
level with the base conversion. **Measured: detrending is not the
differing choice either.**

seg5 (5 s segments): real mean 0.334, synthetic scores level.
**Measured: segment length at 5 s is not the differing choice.**

seg10 (10 s segments): real mean 0.336, synthetic scores level
(paired on v3 windows within ±0.002 on three of four holdouts).
**Measured: segment length at 10 s is not the differing choice
either. Across 1 s / 5 s / 10 s the STFT axis is exhausted.**

welch1s (1 s segments, 50% overlap — the classic Welch variant a
different implementer would plausibly choose): real mean 0.332,
synthetic scores level. **Measured: the overlap/Welch choice is
not the differing choice either.**

median (median across segments instead of mean): real mean
**0.324**, the second-largest movement after dbmean, with the
best own-holdout scores of any variant (0.9293 / 0.9267 / 0.9336 /
0.9209). Median and dB-mean aggregation share one property: both
down-weight loud short events relative to linear mean power. The
pattern — the two robust aggregators move the real mean by
0.013–0.018, everything else by ≤0.005 — is **Supported** evidence
that the organisers' aggregation, if it differs at all, differs in
the robust direction; but a movement of at most 0.018 against a
≈0.16 gap means aggregation choice explains at most about a tenth
of the failure. **Rejected as a sufficient account.**

bandext (band boundaries, priority 4) — **the differing choice.**
Exact math: identical to the base conversion except the two edge
bands. Band 0 integrates from 0 Hz (the lowest FFT bin) instead of
its deci-decade lower edge (≈56 Hz); band 24 integrates up to
Nyquist (16,000 Hz at the 32 kHz sample rate) instead of its
deci-decade upper edge (≈8.9 kHz). Interior bands unchanged.

Results: real-window mean **0.2042** (median 0.1634, fraction
below 0.05 = 0.209, 90th percentile 0.4565) against 0.337 for v3
and 0.319–0.336 for every other variant — a movement of 0.133
against a gap of ≈0.16, landing just above the 0.13–0.20 prior
band. Its own holdout scores are the best of any variant on three
of four holdouts (0.9445 random / 0.9459 period / 0.9392 long; the
site holdout dips to 0.8911, the one synthetic weakness, to be
re-examined in the full run). Cross-scores collapse in both
directions (v3 A on bandext windows 0.72–0.76; bandext model on
v3 windows 0.79–0.84): the two conversions define genuinely
different feature worlds, and only the bandext world reads the
real windows plausibly.

Interpretation (Supported, with the mechanism consistent with all
measurements): the released background windows are the organisers'
own conversion, so they already contain full-range edge bands —
real band 0 carries the large sub-56 Hz energy of ocean
recordings. Our base conversion clipped the added passages' edge
bands, so in synthetic windows an added vessel contributed far
less edge-band energy, relative to the interior, than a real added
vessel does in the released mixtures. The model calibrated on
that distortion systematically over-read real windows. Extending
the two edge bands to the spectrum edges removes most of the
distortion. The residual gap (0.204 against a band top of 0.20)
is the size the remaining corrections — the measured mask-form
mismatch (workstream 3) and robust aggregation (dbmean, above) —
would predict. **Measured: edge-band clipping was the principal
conversion error.** The full A/B/main protocol and the factorial
interaction test now run on the bandext base.

min60 (whole-minute single FFT, priority 5): real mean 0.335,
synthetic scores level. **Measured: minute alignment / segment
structure is not the differing choice.**

**Workstream 1 synthesis.** Ten conversion variants screened on
identical protocol. Nine leave the real-window mean within
0.319–0.336 (v3: 0.337): the aggregation axis moves it slightly
(dbmean 0.319, median 0.324, magmean 0.328), the STFT axis
(window, detrend, segment length, overlap) and minute alignment
do not move it at all. One variant, bandext, moves it to 0.204
and simultaneously produces the best synthetic holdouts of the
phase. The conversion question therefore has a definite answer:
**the organisers' conversion integrates the first band from the
bottom of the spectrum and the last band to Nyquist; the starter
notebook's deci-decade edge clipping — which our conversion
inherited — was the principal reason every model generation
over-read the real windows.** Channel handling (priority 6) was
tested as workstream 4's channel family rather than as a
conversion variant, because the passages are single-hydrophone
recordings; see there.

## Workstream 4 — Background channel (domain shift) — COMPLETE

Code: `code/simulate_bgchan.py`; record: `models/conv_bgchan_screen.json`.
Construction: after the v3 background draw, the background is
multiplied per band by a channel drawn from the release's own
inter-site spectral differences — an ordered pair of distinct sites'
median spectra, differenced and scaled by u ~ U(0.5, 1.0), constant
across the window's minutes. Measured size of that variation in the
release: inter-site spectral-difference RMS median **14.2 dB**, p90
26.0 dB; applied channel RMS averaged **11.2 dB** — far larger than
v3's ±2 dB jitter, deliberately, because that is the grounded size
of export variation the release itself exhibits.

Screen results (trainA model): own holdouts 0.9195 / 0.9264 /
0.9205 / 0.9048 (random / site / period / long) on channel-shifted
windows; the v3 A model cross-scores 0.9107 / 0.9137 / 0.9182 /
0.9035 on the same windows (the channel makes windows harder for a
model that never saw it). On identical v3 windows the channel-trained
model scores 0.9199 / **0.9265** / 0.9210 / 0.9023 against v3 A's
0.9235 / 0.9194 / 0.9233 / 0.9085 — it generalises back to clean
windows about as well as v3 itself, better on the site holdout.
Construction ceiling 0.9977.

**The decisive number: the channel-trained model reads the real
windows at mean 0.3350** (median 0.2738), indistinguishable from the
v3 family's 0.335–0.339. Training the model to be invariant to
release-sized channel responses did not move its real-window
readings at all.

**Answer to the workstream question.** Is the remaining discrepancy
attributable to recording or export differences of this kind?
**Rejected (Measured)** for the channel family as constructed: if
the real windows differed from ours by a per-band frequency response
of the size the release exhibits between its own sites, a model
trained to be invariant to exactly that family would read them
differently. It does not. A residual possibility — a *single fixed*
export difference combined with something else, rather than a
per-window random response — is **Unresolved**, but note the
channel-trained model is also robust to any single member of the
family it trained on, which is what a fixed response would be.

Addendum after workstream 1 completed: the channel test ran on
the base (edge-clipped) conversion, which workstream 1 has since
shown to be wrong. The channel factor is therefore not retested
on the bandext base in this phase: it showed no independent
transfer value of its own (0.3350 against 0.337), and the
interaction test combines only independently justified variants.
Whether channel robustness adds anything on the corrected
conversion is **Unresolved** and recorded as an open question,
not silently dropped.

### Bandext full run (the full A/B/main protocol)

Record: `models/conv_bandext_full.json`. Full mode generates the
identical windows to v3 (same RNG continuation), with bandext
passages substituted.

Holdouts (main / A / B), A–B agreement RMSE:

| holdout | main | A | B | agreement |
|---|---|---|---|---|
| random | 0.9488 | 0.9430 | 0.9440 | 0.0306 |
| site | 0.9021 | 0.8915 | 0.9009 | 0.0411 |
| period | 0.9465 | 0.9412 | 0.9415 | 0.0323 |
| long | 0.9415 | 0.9358 | 0.9377 | 0.0336 |

Against v3's 0.9271 / 0.9264 / 0.9280 / 0.9146: bandext gains
≈+0.02 on random, period and long, and loses ≈−0.024 on the site
holdout — the one synthetic weakness, consistent with the screen.
The site holdout's backgrounds are two unseen sites; under
bandext the first band carries each site's low-frequency
character, and the model evidently leans on it in a partly
site-specific way. A–B agreement (0.031–0.041) is tighter than
v3's (0.037–0.045). Cross-world scores confirm the worlds differ:
v3 models score 0.758–0.761 on bandext windows; the bandext model
scores 0.845–0.851 on v3 windows. True-zero windows read
0.014–0.018 (natural traffic still not counted). Error is
concentrated in the mid and high share bands (0.1–0.3 and 0.3–0.6
carry 28% and 31% of SSE on val_random) with small biases
(+0.014 at zero share, −0.018 at 0.3–0.6).

Real windows (main model): mean **0.2168**, median 0.1703,
fraction below 0.05 = 0.217, deciles 0.024 / 0.058 / 0.170 /
0.345 / 0.493. Confidence categories on the real windows (fixed
thresholds on main/A/B agreement): **clean 799 (53.3%), reading
0.130 on average — inside the prior band; ambiguous 623 (41.5%),
reading 0.298; unresolved 78 (5.2%), reading 0.457.** The
over-read that remains is concentrated in the windows the three
models disagree about; where the models agree, the readings are
already plausible.

Distance to aim on the best holdout: 0.0492 (0.9488 against the
0.998 aim; construction ceiling 0.9982). Bandext alone is
therefore **not a candidate**: it misses the 0.96 development
minimum on every holdout and the site holdout is its weakest
point. It is, however, a transformed base — the first simulator
generation whose real-window behaviour is in the neighbourhood
of the prior band — and it is the base for the interaction test.

## Interaction test — factorial on the bandext base

Design (fixed after the workstreams named their strongest
variants; only independently justified factors are combined):
base = bandext conversion (workstream 1's winner). Factor 1 =
aggregation: mean (base) vs dbmean — independently Supported on
the base conversion (real mean −0.018 alone). Factor 2 = mask
form: v3's Bernoulli process (base) vs the measured real form
(complete 0.517 / minute-1-only 0.413 / minute-15-only 0.065 /
both 0.005) — a Measured mismatch from workstream 3. The channel
factor (workstream 4) is excluded: it showed no independent
transfer value, and only independently justified variants are
combined. Cells: bandext (full run, above) = mean × Bernoulli;
bandextdb (screen) = dbmean × Bernoulli; bandext_maskfix (full)
= mean × corrected mask; the fourth cell (dbmean × maskfix) runs
only if both single-factor cells show independent value.

**Cell bandextdb (screen).** Own holdouts 0.9463 / 0.8883 /
0.9493 / 0.9421 — statistically indistinguishable from bandext's
screen (0.9445 / 0.8911 / 0.9459 / 0.9392). Real-window mean
**0.2130** against bandext screen's 0.2042 (model-to-model
variation in this phase is ≈±0.01: bandext's own screen and full
main models read 0.2042 and 0.2168). **Measured: dB aggregation
adds nothing on the corrected conversion.** Its small effect on
the clipped conversion was evidently another partial symptom of
the same distortion — dB averaging down-weights exactly the loud
segments the clipped edge bands mis-weighted. The dbmean ×
maskfix cell is therefore not justified unless maskfix alone
shows strong independent value and a specific reason to combine
emerges.

**Cell bandext_maskfix (full run).** Record:
`models/conv_bandext_maskfix_full.json`. Holdouts (main / A / B):
random 0.9482 / 0.9434 / 0.9436, site 0.9030 / 0.9002 / 0.9051,
period 0.9469 / 0.9439 / 0.9439, long 0.9422 / 0.9381 / 0.9383 —
statistically identical to bandext's full run (0.9488 / 0.9021 /
0.9465 / 0.9415); A–B agreement 0.0307 / 0.0402 / 0.0316 / 0.0332,
also identical. Construction ceiling 0.9981. Real windows: main
mean **0.2061**, A 0.2099, B 0.1962 (bandext family: main
0.2168) — a small, consistent improvement of ≈0.01 in the real
distribution at no synthetic cost. Confidence categories on the
real windows: clean 770 (51.3%, mean 0.124), ambiguous 641
(42.7%, 0.279), unresolved 89 (5.9%, **0.391**, down from
bandext's 0.457 — the corrected mask form mainly improves the
windows the models previously disagreed about most).

**Interaction verdict.** The two corrections do not compound
into a breakthrough, because there is little left for them to
fix once the conversion is right: dbmean adds nothing
(**Measured**), maskfix adds a small real-distribution
improvement and is worth keeping on principle — it trains on the
mask form the real windows actually have (**Measured**, small).
The fourth cell (dbmean × maskfix) is **not run**: dbmean showed
no independent value on the bandext base, and the design rule
was to combine only independently justified variants. No
interaction resurrects the channel factor (excluded by the same
rule; its status stays Unresolved on the corrected base).

## Gate judgement — NOT MET (no candidate)

Best construction of the phase: **bandext_maskfix** (bandext
conversion, corrected mask form, mean aggregation). Against the
binding standards:

- Development minimum 0.96: **not met.** Best holdout (main,
  period) 0.9469; site holdout 0.9030 is the weakest point.
- Submission standard ≈0.998 on the fixed validation AND both
  cross-checks: **not met.** Distance to aim on the best holdout
  is 0.0498; the A and B models sit at 0.943–0.944 where the main
  model sits at 0.948. The construction ceiling (0.9981) shows
  the remaining gap is model estimation error, not label
  construction.
- Real-window behaviour: **credibly improved** — mean 0.2061
  (family 0.196–0.210) against v3's 0.337, at the top edge of
  the 0.13–0.20 prior band, with the clean majority (51.3%)
  reading 0.124. This is a diagnostic, not ground truth, and it
  does not override the synthetic gate.

**No candidate submission file is produced and no Zindi slot is
spent.** The per-model test prediction CSVs written by the
training runs are diagnostics, not submission files. The Zindi
site was not opened during this phase.

**The five questions, answered.**
1. *Conversion* — **Yes, and it was the main problem (Measured).**
   The organisers' conversion integrates band 0 from 0 Hz and
   band 24 to Nyquist; the starter notebook's deci-decade edge
   clipping, inherited by our conversion, was the principal
   cause of the transfer failure (real mean 0.337 → 0.204 at
   screen, best synthetic holdouts of the phase). Aggregation
   (power/magnitude/dB/median), window function, detrending,
   segment length, overlap and minute alignment are each
   **Rejected** as sufficient explanations; robust aggregation
   retains a small **Supported** effect on the clipped
   conversion only, which vanishes on the corrected one.
2. *Target arithmetic* — **No (Measured).** v3's observed-cells
   sum is the only audited definition consistent with the real
   top score 0.9943 and the label moments; the mean/band/
   amplitude alternatives are **Rejected** (ceilings
   0.78–0.94). Rounding order is immaterial (RMSE 0.0008).
3. *Missingness* — **No label information (Rejected,
   Measured).** The real mask is an independent, edge-minute
   process; its form mismatch in simulation was **Measured**
   and is corrected by maskfix (small real improvement). The
   loud minute-1 sub-population was mostly a conversion
   symptom (excess +0.065 → +0.0075 under bandext, Measured);
   its residual origin is **Unresolved**.
4. *Domain shift (channel)* — **Rejected (Measured)** for the
   release-sized per-window channel family: channel-invariant
   training did not change real readings at all (0.3350 vs
   0.337). A single fixed export response on the corrected
   conversion is **Unresolved** (not retested; no independent
   justification).
5. *Interaction* — **Small and mostly null (Measured).** On
   the bandext base, dbmean adds nothing; maskfix adds ≈0.01
   of real-mean improvement at no synthetic cost; the
   combination cell was not justified and not run. The large
   effect was the conversion alone; the corrections were
   symptoms of it, not independent causes.

**What stands between here and the standard** (for the next
decision, not acted on in this phase): the binding constraint
is now ordinary model error on a correct simulator — SSE
concentrated in the 0.1–0.6 share bands — plus a specific site
generalisation weakness (0.903) plausibly tied to the first
band's site character. Both are model-side problems (capacity,
features, training-set size, site-robust training), a different
kind of work from this phase's measurement questions, which are
now closed.

**Seeds and configurations.** Master seed 20261116 (v3 split
discipline inherited exactly). Screen streams: seed 20261116 +
91000 + TAG_ORDER index (bandext index 9, bandextdb index 10,
appended last so earlier indices are unchanged). Full mode: the
identical RNG continuation as simulate_v3. Interaction cells:
maskfix stream seed = (20261116 + 94000) + abs(hash(tag)) mod
1000 (simulate_interact.py); the realised value is recorded in
`data/sim_cf_bandext_maskfix_meta.json` (generation_seed
20355124). Screen streams realised: bandext 20352125, bandextdb
20352126 (recorded in their meta files). Missingness audit: 12,000 windows per
arm, seed recorded in `models/audit_missing.json`. Label audit:
deterministic, `models/audit_labels.json`. All trainings:
LightGBM, v3 hyperparameters, 6000-round cap, identical
early-stopping rule; per-model best iterations and versions are
recorded in each `models/conv_*_*.json`. Conversion checkpoints:
`data/passages_conv_<tag>.npz` (542 clips each); the base
configuration reproduced `passages.npz` at max |Δ| = 0.00 dB.
