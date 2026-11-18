# Phase 3 diagnostic — background matching investigation (2026-11-16)

**Verdict in one paragraph.** The hypothesis that test backgrounds are
verbatim windows from the released South African acoustics is **refuted**
for the real test set: across 60 sampled test windows, with a median of
5,829 candidate backgrounds examined per window, **not one window's best
candidate contains a single exact minute**, while synthetic windows built
the organisers' way carry a median of 6 exact minutes in their true
background. Even after the best scalar gain alignment, the best release
candidates still violate the physical background constraint in 8% of
cells. And even in the synthetic world where the hypothesis is true by
construction, **no deployable matcher reaches Godson's 0.93 gate**
(best raw matcher 0.8195; the two independent matchers disagree with
each other at RMSE 0.3057). Only two things reach 0.93+: subtraction
with the *true* background (oracle, 0.9528 — unobtainable on real data)
and the matcher restricted to the 86% of synthetic windows where its
own exact-minute fingerprint fires (0.9493 — a fingerprint that fires
on **0 of 60** real windows). **Nothing from Phase 3 is
submission-ready, and nothing should be submitted from it.**

All scores are 1 − RMSE on `share_added_vessel`. Seed 20261116
everywhere. Test.csv was used for inference/diagnostic only; nothing
was fitted, calibrated or pseudo-labelled from it. No submission was
made. Code: `code/phase3_index.py`, `phase3_gen.py`, `phase3_match.py`,
`phase3_synth.py`, `phase3_analyze.py`, `phase3_prior.py`,
`phase3_test.py`. Records: `data/phase3_synth_records.json`,
`data/phase3_test_records.json`; logs in `code/phase3_*.log`.

## 1. What was built

- **Minute index** (`phase3_index.py`): every screened one-minute
  record of the release (same screen as the background pool: MinRes,
  dur ≥ 55 s, 25 common bands finite in [25, 160] dB), sorted by site
  and time, with a contiguity map of valid 15-minute window starts.
  Measured en route: released values sit exactly on the 0.1 dB grid,
  like Test.csv — which is what makes an *exact fingerprint* test
  possible at all.
- **Synthetic generator** (`phase3_gen.py`): mirrors the organisers'
  stated process. Background = a real screened pool window (identity
  recorded, withheld from the matchers); addition = a VTUAD passage
  segment (vessel 70% / ambient 30%, event lengths 1–15 min in the v2
  mixture), time-warped, channel-coloured, scaled in the linear domain
  to an exact target share, added, rounded to the 0.1 dB grid, then the
  measured test missingness pattern applied. The label is the realised
  share over the full window before missingness.
- **Two independent matchers** sharing one verification
  (`phase3_match.py`). A candidate background B is physically
  admissible only if B ≤ T (approximately, cell-wise): an addition can
  only add energy. Verification ranks candidates by (1) fewest
  violating cells (B > T + 0.06 dB), (2) most *exact minutes* —
  observed minutes where ≥24 of 25 bands sit on the same 0.1 dB grid
  value as T (the true background equals T exactly in every minute the
  addition does not touch; a merely similar window essentially never
  does), (3) highest exact-cell fraction, (4) smallest median
  high-band gap, (5) smallest mean excess.
  - **Matcher A (KD)**: candidate starts from k-nearest released
    minutes to T's anchor minutes (Manhattan distance on the ≥1 kHz
    bands), filtered by the one-sided constraint at the anchor.
  - **Matcher B (hash)**: candidate starts from released minutes whose
    two highest band values exactly equal (or sit just below) the
    anchor minute's — exact grid hashing, no distance ranking. A
    genuinely different generation mechanism.
  - **Union**: both candidate sets merged, one verification — the
    cross-check configuration.
- **Fallback estimators** (the plan's weaker variants): `quiet`
  (per-band 10th percentile across the window's own minutes as
  background), `stat_or` (true site's median spectrum, level-aligned
  on ≥1 kHz bands), `stat_guess` (same with the site guessed by
  profile fit — the deployable variant, since test rows carry no site
  labels).
- **Oracle**: subtraction with the *true* background. This is the
  ceiling of the entire matching idea.

Experiment design: 200 synthetic windows — 8 share levels
{0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9} × 15 reps on backgrounds from
the 8 training sites (split A, n=120) and × 10 reps on backgrounds
from the 2 holdout sites, Bird Island and Fish Hoek (split B, n=80).
Note: the matcher index contains all sites, holdout included (their
backgrounds *are* in the release), so split B tests generator realism
on unseen sites, not pool exclusion.

## 2. Synthetic results — the decisive experiment

### 2.1 Overall recovery score by estimator

| estimator | split A (n=120) | split B (n=80) | combined (n=200) |
|---|---|---|---|
| oracle (true background) | 0.9403 | 0.9851 | **0.9528** |
| matchB (hash) | 0.8131 | 0.8291 | **0.8195** |
| union (A+B, one verification) | 0.7616 | 0.7658 | 0.7633 |
| matchA (KD) | 0.7131 | 0.5265 | 0.6271 |
| stat_or (true-site spectrum) | 0.5881 | 0.5321 | 0.5648 |
| quiet (own-window percentile) | 0.5983 | 0.4848 | 0.5494 |
| stat_guess (guessed site) | 0.5818 | 0.4749 | 0.5361 |

### 2.2 RMSE by share level (combined)

| share | oracle | matchA | matchB | union | quiet | stat_or | stat_guess |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.4428 | 0.0000 | 0.0000 | 0.6961 | 0.7270 | 0.7512 |
| 0.05 | 0.0044 | 0.5144 | 0.2724 | 0.3191 | 0.6201 | 0.4860 | 0.5256 |
| 0.10 | 0.1164 | 0.5604 | 0.3434 | 0.4329 | 0.5258 | 0.3974 | 0.4441 |
| 0.20 | 0.0234 | 0.4246 | 0.1973 | 0.2937 | 0.4344 | 0.2588 | 0.2564 |
| 0.35 | 0.0090 | 0.2645 | 0.1599 | 0.2065 | 0.3870 | 0.3150 | 0.3289 |
| 0.50 | 0.0126 | 0.2361 | 0.0709 | 0.1352 | 0.2281 | 0.3145 | 0.3495 |
| 0.70 | 0.0579 | 0.1703 | 0.0836 | 0.1069 | 0.2397 | 0.4138 | 0.4499 |
| 0.90 | 0.0089 | 0.0519 | 0.0228 | 0.0233 | 0.1752 | 0.3950 | 0.4307 |

### 2.3 The bimodality: found vs missed

The averages hide the structure. Recovery is essentially binary:

| matcher | found true background | score \| found | score \| missed (RMSE) |
|---|---|---|---|
| matchA | 141/200 (70.5%) | 0.9441 | 0.3189 (0.6811) |
| matchB | 174/200 (87.0%) | 0.9496 | 0.3406 (0.6594) |
| union | 176/200 (88.0%) | 0.9499 | 0.3303 (0.6697) |

When the matcher finds the true background, subtraction is
oracle-quality (0.944–0.950 vs oracle 0.9528). When it misses, it
typically locks onto a much quieter unrelated window and reports a
share near 1 — errors of ~0.66 RMSE. **Estimation is solved;
candidate selection is the entire problem.** Found rate by share level
is flat-ish (union: 1.00 at share 0, 0.76–0.92 elsewhere) — misses are
not concentrated at high shares.

Two further structural findings:

- **Union is worse than matchB alone** (0.7633 vs 0.8195) despite
  finding the true background slightly more often (88% vs 87%).
  Merging candidate sets gives wrong candidates more chances to
  outrank the true one under the proxy ranking. More candidates is
  not more accuracy.
- **The oracle is not perfect** (0.9528, and only 0.9403 on split A).
  Its label is defined over the full window before missingness, while
  the estimate can only use observed minutes; the 0.1 dB rounding
  adds a little more. One share-0.10 window has oracle |error| 0.581:
  missingness dropped the very minutes carrying the event. So even
  *perfect* background recovery caps around ~0.95 under this label
  definition — the 0.93 gate leaves almost no room for selection
  error.

### 2.4 The cross-check (independent agreement)

Matcher A and Matcher B generate candidates by genuinely different
mechanisms, so their agreement is the plan's independent check:

- A vs B estimates on the same windows: mean |difference| 0.1219,
  **RMSE between estimates 0.3057** (n=187 windows where both
  returned an estimate).
- A vs union: 0.2875. B vs union: 0.0713 (union is dominated by B's
  candidates).

For the cross-check to certify a 0.93-standard method, the two
estimates would need to agree at roughly RMSE ≤ 0.07. They disagree
at 0.31 — exactly on the windows where one of them has locked onto a
wrong background. **The cross-check fails at the gate standard**, and
it fails informatively: disagreement is a reliable alarm, agreement
is not (both can be wrong together, and on real windows both are).

### 2.5 Confidence gating — the one bright synthetic result

The verification metrics are *observable* on any window, so we can
ask whether they certify their own answer. On synthetic windows, the
rule "accept if viol ≤ 0.05 and ≥ 1 exact minute" (the thresholds
`phase3_test.py` carried to the real data):

- accepts 86.0% of windows,
- **every accepted window's top candidate is the true background**
  (found | accepted = 1.000, n=172),
- score among accepted: **0.9493**.

A stricter rule (viol ≤ 0.01, ≥ 3 exact minutes) accepts 59% at
0.9398, also with zero false accepts. In the synthetic world the
exact-minute fingerprint is a perfect certificate: when it fires,
the answer is right at oracle quality. Hybrid pipelines (gated union
estimate, fallback elsewhere) still cap at **0.8549** (stat_guess
fallback) / 0.8523 (constant 0.1 fallback), because the 12–14% of
windows the gate rejects fall back to weak estimators. The gate
certifies; it does not rescue.

### 2.6 Reproducibility note (v1 run vs fresh run)

An earlier run of the same experiment (log `phase3_synth_v1.log`,
records `phase3_synth_records_v1.json`) used a verification ranking
*without* the exact-minute priority: matchB found 78%, scored
0.7613, A–B agreement RMSE 0.3167. The fresh run (exact-minute
ranking) on identical seeded windows improved found rates (A 64.5% →
70.5%, B 78% → 87%) and matchB's score (0.7613 → 0.8195). Every
conclusion above is unchanged across both rankings.

## 3. Real test windows — the hypothesis meets the data

`phase3_test.py`: 60 test windows spread evenly across Test.csv,
union candidate generation, same verification, classified with the
synthetic-derived thresholds (clean / ambiguous / none). Inference
only; nothing fitted.

**Result: 0 clean, 0 ambiguous, 60 none.**

| quantity (best candidate per window) | median | max |
|---|---|---|
| violation fraction | 0.0000 | 0.0053 |
| exact minutes | **0** | **0** |
| exact-cell fraction | 0.0057 | 0.0143 |
| residual share implied | 0.987 | 1.000 |
| residual low-band fraction | 0.940 | 0.999 |
| residual spectral correlation with vessel profile | 0.009 | 0.725 |
| gain-aligned violation (after best ±3 dB offset) | 0.080 | — |
| candidates examined per window | 5,829 | 6,877 |

Reading, point by point:

- Violation ≈ 0 is **not** evidence of a match — thousands of quiet
  release windows sit entirely below any test window. That is why
  the implied residual shares are ~0.99: the "best" candidates are
  just very quiet windows, and the residual spectrum does not
  resemble a vessel passage (spectral correlation ≈ 0.01).
- The decisive number is **exact minutes = 0 in all 60 windows**.
  In synthetic windows the true background shows a median of 6 exact
  minutes (untouched minutes reproduce the test window cell-for-cell
  on the 0.1 dB grid). On real windows, across ~5,800 candidates per
  window, no best candidate reproduces even one minute exactly, and
  the exact-cell fraction (~0.006) is chance level. If the test
  backgrounds were release windows passed through the same export,
  untouched minutes — especially in the high bands, where additions
  are weakest — would fingerprint themselves. They do not.
- Gain shift does not save it: the best scalar offset hits the
  +3 dB clip bound at the median and still leaves 8% of cells
  violating B ≤ T (synthetic true backgrounds: ~0%). The best
  release windows are not gain-shifted copies of the test
  backgrounds either.

**Conclusion: the test backgrounds are not in the released minutes**,
verbatim or gain-shifted. The plausible remaining explanations are
(a) backgrounds drawn from withheld periods or stations of the same
recording programme (the release need not contain the exact minutes
used), or (b) backgrounds passed through a different
averaging/export chain that destroys the exact grid fingerprint.
Either way, exact background recovery — the only route Phase 3 found
that reaches 0.93+ anywhere — has nothing to lock onto in the real
data. This also retro-explains the top-of-board 0.9943 differently:
whoever achieved it did not do it by matching against *this*
release, or knows the withheld source.

## 4. Prior diagnosis — what the two public scores pin down

From `phase3_prior.py` (arithmetic in the script and its log).
On the public split (~30% of the 1,500 windows), with s the true
shares, μ = E[s], Var = Var(s), and p our v2 predictions (known
exactly; full-file moments used as split estimates, sampling noise
~0.01 on the mean):

- (1) Var + (μ − 0.1)² = RMSE_b² = **0.060581** exactly. Hence
  sd(s) ≤ 0.2461 and μ ∈ [0, 0.3461]. The pair of scores does **not**
  pin μ and Var separately — one equation short.
- (2) E[(s−p)²] = 0.097426 with E[p] = 0.3299, Var(p) = 0.0584 gives,
  for each assumed μ, the implied correlation ρ(p, s):

| assumed μ | Var(s) | sd(s) | implied ρ | optimal shrunk RMSE | best shrunk score |
|---|---|---|---|---|---|
| 0.10 | 0.0606 | 0.2461 | +0.626 | 0.1920 | 0.808 |
| 0.13 | 0.0597 | 0.2443 | +0.514 | 0.2096 | 0.790 |
| 0.17 | 0.0557 | 0.2360 | +0.371 | 0.2192 | 0.781 |
| 0.20 | 0.0506 | 0.2249 | +0.262 | 0.2171 | 0.783 |
| 0.25 | 0.0381 | 0.1951 | +0.058 | 0.1948 | 0.805 |

(The upturn at μ = 0.25 is an artefact of the assumed μ approaching
the boundary where Var(s) shrinks; the plausible range, per the
benchmark cross-check in RESULTS_sim_v2.md, is μ ≈ 0.13–0.20.)

- In that range, the v2 predictions correlate with the truth at only
  ρ ≈ 0.26–0.51, and **even the optimal linear shrinkage of the v2
  predictions caps at ≈ 0.78–0.79** — above the 0.7539 benchmark,
  nowhere near 0.93. Shrinkage is a floor defence, not a route.
- Also exact from (1): the best possible constant predictor scores
  1 − sd(s); at μ = 0.17 that is 0.7640 (constant 0.17).

## 5. Judgement against the gate

Godson's gate: a candidate method is submission-ready only if it
proves **0.93 or above on known-answer validation**, with the
independent cross-check agreeing at the same standard, and per-window
confidence reported.

| candidate | known-answer score | cross-check | verdict |
|---|---|---|---|
| oracle subtraction | 0.9528 | n/a (uses ground truth) | ceiling only; not a method |
| gated matcher (synthetic, 86% coverage) | 0.9493 on accepted | fingerprint never fires on real windows (0/60) | not deployable on test |
| matchB raw | 0.8195 | A–B agreement RMSE 0.3057 | **falls short** |
| union raw | 0.7633 | — | falls short |
| hybrid gate + fallback | ≤ 0.8549 | — | falls short |
| quiet / stat fallbacks | 0.54–0.56 | — | falls short |
| shrunk v2 model (from §4) | caps ≈ 0.78–0.79 public | n/a | floor defence only |

**Nothing meets the gate. The verbatim-matching hypothesis is
refuted on real data; the weaker background estimators are not
competitive; the v2 model family caps far below the gate under any
shrinkage.** Per the standing instruction, the second submission
slot stays unspent.

## 6. Weaknesses of this investigation (stated plainly)

- The synthetic generator is *our* mirror of the organisers' process,
  not theirs. The refutation in §3 does not depend on the generator —
  it rests on the absence of exact minutes in real windows — but the
  synthetic *scores* (0.82, 0.95) inherit every simulator assumption,
  including the notebook conversion the organisers themselves call
  unverified.
- The test diagnostic sampled 60 of 1,500 windows (evenly spread).
  60/60 negative at this margin makes a different outcome on the
  full file implausible, but it is a sample, not a census.
- Matcher candidate generation is capped (6,000 starts per matcher
  per window); a true background outside the anchor neighbourhoods
  would be missed by construction. On synthetic windows this cap is
  part of why found rates are 70–88% rather than ~100%. The §3
  refutation is less exposed to this caveat than it may look: the
  fingerprint test was applied to ~5,800 candidates per window
  including hash-exact anchors on the two highest bands, and the
  gain-aligned check is independent of candidate ranking — but a
  exhaustive all-windows scan was not run and would be the strict
  form of the test.
- Split B's holdout sites contribute few background windows (Fish
  Hoek 296 in the pool); split-B numbers carry wide uncertainty.
- The prior diagnosis assumes the public split's prediction moments
  resemble the full file's; a non-random split would move the
  implied-ρ table somewhat. The headline (shrinkage caps ≈ 0.78)
  is robust across the whole plausible μ range.

## 7. Ranked next steps

1. **Attack the sim→real gap where it is measured to live: the
   backgrounds.** The v2 model over-reads real windows (predictions
   mean 0.330; implied ρ only 0.26–0.51), and Phase 3 now shows the
   real backgrounds are *not* release windows — consistent with
   withheld-period backgrounds whose character differs (e.g. more
   genuine vessel traffic, which a model reads as "added"). Build
   simulator v3 backgrounds that mimic that: include release windows
   containing real vessel-like events as backgrounds, and background×
   background mixtures, then re-measure transfer with one smoke
   submission. This is the only route with evidence behind it.
2. **Tolerant near-matching, honestly pre-validated.** Replace the
   exact fingerprint with per-band gain-aligned matching plus a
   residual vessel-shape test — but first validate it on synthetic
   windows whose backgrounds have been passed through
   fingerprint-destroying transforms (per-band gain jitter,
   re-rounding from jittered values). If it cannot beat ~0.85 there,
   drop the matching line entirely; §3's gain-aligned viol of 0.08
   already warns the ceiling is low.
3. **Exhaustive fingerprint census (cheap, closes §3 strictly).**
   Run the exact-minute scan over all 1,500 test windows against all
   contiguous release windows without candidate caps. Expected
   outcome is confirmation of the refutation; cost is compute, not
   submission slots. Worth doing once so the writeup's central claim
   is census-grade.
4. **Shrinkage floor, if a slot is ever to be spent defensively.**
   Optimally shrunk v2 predictions cap at ≈ 0.78–0.79 public —
   above benchmark, and a third public score would pin μ and Var(s)
   much more tightly (a third equation). This is a decision for
   Godson, not an agent action: the standing instruction is that the
   slot stays held.
5. **Writeup assets.** The Phase 0–3 record (null checks, pressure
   maps, site/period holdouts, this refutation with its fingerprint
   method) is strong material for the two $1,000 best-writeup prizes
   independent of leaderboard outcome. Keep packaging it as we go.
