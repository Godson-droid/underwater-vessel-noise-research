# Simulator v3 — results (2026-11-16)

**Verdict up front: the gate is NOT met, and no candidate submission file
has been produced.** Simulator v3 fixed the one thing that was provably
broken — the label construction (the construction's oracle ceiling rose
from 0.951 to 0.9983, so a 0.994-class score is no longer arithmetically
impossible under our mirror of the organisers' process) — and it improved
synthetic validation to 0.926–0.928. But it did **not** close the
sim-to-real gap: every v3-family model still reads the real test windows
at a mean of ≈ 0.335, against a plausible true mean of 0.13–0.20 — the
same over-reading as v2 (0.330). On identical validation windows (the
cross-matrix in §5), v3's apparent gains evaporate. Per Godson's
decision rule, the gate is not lowered, the criteria are unchanged, no
submission slot is spent, and this report ends with the gate judgement
and the diagnosis.

Code: `code/label_variants.py`, `code/simulate_v3.py`,
`code/train_sim_v3.py`, `code/cross_matrix_v2_v3.py`. Records:
`models/sim_v3_validation.json`, `models/cross_matrix_v2_v3.json`.
Seed 20261116 everywhere. V1, V2 and all Phase 3 code and results are
untouched; v3 lives entirely in new files. Nothing was fitted,
calibrated or tuned on Test.csv — real-window numbers below are
inference-only diagnostics, labelled as such. Background matching was
not pursued in any form (closed direction); the only subtraction used
anywhere is the oracle *ceiling diagnostic* on synthetic windows whose
backgrounds are known by construction.

## 1. What v3 changed, and why (each change tied to a measurement)

1. **Label / missingness construction** — the subject of §2. Label =
   realised share over the **observed cells** of the final window; the
   two generative orders sharing that label function (drop-first and
   mix-first) are mixed 50/50.
2. **Backgrounds with withheld character** (Phase 3 §7 step 1; Godson's
   brief points A–C):
   - *clean* (40%): screened pool window, level-stratified, as v2.
   - *busy* (30%): pool window from the top traffic quartile of its
     split pool (traffic score = the window's own B1-style low-band
     excess; pool median 0.427, p75 0.646) — recordings that already
     contain genuine vessel traffic. **Label rule (brief point A): the
     label is computed exclusively from the artificially added
     fraction** (the placed event's energy over observed cells);
     natural traffic in the background is never counted toward it.
   - *mixture* (30%): two pool windows combined in the energy domain,
     E = w·E1 + (1−w)·E2, w ~ U(0.3, 0.7), rescaled to the first
     component's total level — a blended character no single release
     window has (brief point B; its measured effect is §6).
   - *channel jitter* (p = 0.5): mild smooth per-band gain jitter
     (±2 dB) — a different export chain's channel.
   - *changing ambient conditions* (p = 0.5; brief point C): a smooth
     per-minute level envelope (±2.5 dB) plus a spectral character
     that drifts across the 15 minutes (interpolation between two
     smooth ±2 dB colourings). Structured smooth processes grounded in
     how real recordings evolve — no uniform-noise augmentation is
     used anywhere in the backgrounds.
3. **Passage variation per minute** (PLAN §4.7): the event colouring is
   now time-varying (interpolation between two independent smooth
   ±7 dB colourings across the event), and the gain profile is a smooth
   envelope widened to ±5 dB plus iid per-minute jitter ±1.5 dB plus,
   with p = 0.3, a step of ±2–4 dB at a random minute. (v2: one static
   colouring, smooth ±3.5 dB envelope only.)
4. **Grid rounding**: mixed windows are rounded to the 0.1 dB grid
   before missingness, matching Test.csv's measured quantisation
   (v1/v2 did not round).

Unchanged from v2: share prior (training label mean realised 0.182,
inside the plausible band), ambient-addition probability 0.30, ambient
segment slicing, event duration mixture, split discipline (holdout
sites Bird Island + Fish Hoek; period holdout = top 15% by start;
8% val_random carve-out; passage-clip split membership identical to
v1/v2 — pool sizes reproduce v2's exactly: 65,838 / 5,724 / 3,459 /
12,632), the 549-feature set (unchanged — nothing in it was shown to
be broken), manual hyperparameters identical to v2, no AutoML.

Models: **main** (80k mixes, full pools); **A** and **B** (40k each,
disjoint in BOTH background windows and passage clips — the gate's
independent cross-check); **NM** (ablation: the identical 80k training
examples minus mixture-mode ones, 56k, for brief point B).

## 2. Constraint 1 — the label construction, settled by experiment

`code/label_variants.py`: paired draws (same background, event, target
share and missingness mask for all three variants; 240 windows over the
8 Phase 3 share levels), scored by the oracle — subtraction with the
true background over observed cells:

| construction | oracle score |
|---|---|
| A — v1/v2 status quo: label = full-window share, computed before missingness | **0.9510** |
| B — drop-first: mask first, addition scaled on observed cells, label = observed share | **0.9995** |
| C — mix-first: v2-style mixing, label recomputed over observed cells | **0.9995** |

Variant A reproduces Phase 3's 0.9528 cap under an independent
generator, confirming the cap is a property of the label definition,
not of any matcher. Under A, a 0.994-class score is arithmetically
impossible; under B/C it is not. **Adopted: the observed-cells label
(B/C's shared label function), with both generative orders mixed**,
since the measurements cannot distinguish B from C and the label
function — the part that matters — is identical.

Measured again on the full v3 construction (2,000 oracle-set windows
with contaminated, mixture, jittered and drifted backgrounds):
**oracle ceiling 0.9983**. The construction is now consistent with
the real top score of 0.9943.

A visible consequence in the pressure maps (§7): missingness now
costs almost nothing (val_random RMSE 0.071 complete → 0.077 with
≥2 missing minutes; v2's map rose 0.074 → 0.118), because dropped
minutes no longer carry hidden label mass.

## 3. Validation on v3-generation holdouts (identical protocol to v1/v2)

Scores are 1 − RMSE. A/B agreement = RMSE between the two independent
models' predictions.

| set | B0 | v2 model | **v3 main** | A | B | NM | A–B agree |
|---|---|---|---|---|---|---|---|
| val_random | 0.7375 | 0.9108 | **0.9271** | 0.9235 | 0.9243 | 0.9262 | 0.0370 |
| val_site | 0.7513 | 0.9079 | **0.9264** | 0.9194 | 0.9238 | 0.9240 | 0.0450 |
| val_period | 0.7299 | 0.9046 | **0.9280** | 0.9233 | 0.9258 | 0.9260 | 0.0383 |
| val_long | 0.6960 | 0.8964 | **0.9146** | 0.9085 | 0.9113 | 0.9119 | 0.0407 |

Null behaviour (mean prediction on true-zero windows): main 0.016 /
0.030 / 0.016 on random / site / period — healthy, and better than the
v2 model's nulls on the same sets (0.022 / 0.052 / 0.025).

## 4. The real-window diagnostic (inference only)

Prediction distributions on the 1,500 real test windows, against the
Phase 3 prior diagnosis (plausible true mean share 0.13–0.20; the v3
training labels themselves average 0.182, inside the band):

| model | mean | median | frac < 0.05 |
|---|---|---|---|
| v2 model | 0.3299 | 0.2854 | 0.126 |
| **v3 main** | **0.3372** | 0.2825 | 0.144 |
| v3 A | 0.3348 | 0.2816 | 0.156 |
| v3 B | 0.3394 | 0.2859 | 0.145 |
| v3 NM (no mixtures) | 0.3352 | 0.2850 | 0.146 |

**The distribution did not move.** Four independently trained
v3-family models, with contaminated backgrounds, mixtures, channel
jitter, ambient drift, per-minute passage variation and the corrected
label, all read the real windows at 0.335–0.339 — statistically the
same over-reading as v2's 0.330. Whatever makes real windows look
"added-rich" to these models was not touched by any background-side
intervention in v3.

## 5. Cross-matrix — V2 vs V3 on identical windows (Godson's requirement)

Both frozen models scored on both generations' holdout windows (same
splits and seeds as each generation used):

| windows | v2 model | v3 model | Δ (v3 − v2) |
|---|---|---|---|
| v2 val_random | 0.9110 | 0.9033 | −0.0077 |
| v2 val_site | 0.9082 | 0.9184 | +0.0102 |
| v2 val_period | 0.9094 | 0.9017 | −0.0077 |
| v2 val_long | 0.8981 | 0.8901 | −0.0081 |
| v3 val_random | 0.9108 | 0.9271 | +0.0163 |
| v3 val_site | 0.9079 | 0.9264 | +0.0185 |
| v3 val_period | 0.9046 | 0.9280 | +0.0234 |
| v3 val_long | 0.8964 | 0.9146 | +0.0182 |

**Plain statement: v3's gains do not survive on identical windows.**
On v2-generation windows the v3 model is, on balance, indistinguishable
from the v2 model (three small negatives, one positive). Its +0.016 to
+0.023 advantage exists only on v3-generation windows — the world it
was trained on and the v2 model never saw. One asymmetry must be
recorded for fairness: v2-generation labels use the old pre-missingness
definition, which penalises the v3 model for predicting the
observed-cells share on windows with dropped minutes; that explains
part (not all) of the negative deltas there. Even granting that, there
is no identical-windows evidence that v3 is the better model, and the
one comparison that is not synthetic at all (§4) shows no improvement
either. The v3 synthetic task did not merely get easier — its labels
got *truer* (§2) — but the model's real-world reading did not improve.

## 6. Brief point B — did background mixtures help? Measured answer: no

The NM ablation (identical training examples minus mixture-mode ones)
vs main: validation deltas ≤ 0.002 on every set (0.9262 vs 0.9271
random; 0.9240 vs 0.9264 site; 0.9260 vs 0.9280 period; 0.9119 vs
0.9146 long); on mixture-mode validation windows specifically, main is
better by only ~0.003 RMSE (e.g. val_random mixture windows: main
0.068, NM 0.071); on the real-window diagnostic the means are 0.3372
(main) vs 0.3352 (NM). **Mixtures are neutral within measurement
noise on every diagnostic we have.** They neither measurably improve
realism nor measurably hurt; the hypothesis that background blending
was the missing realism ingredient is not supported.

## 7. Brief point A — label semantics, verified in behaviour

The label is computed solely from the placed addition's energy over
observed cells (see `simulate_v3.py:make_example`); background energy —
including genuine vessel traffic in busy windows — enters only the
denominator (total window energy), never the numerator. Behavioural
check, mean prediction on true-zero windows by background mode (main
model): val_random — busy 0.016, clean 0.018, mixture 0.013;
val_period — busy 0.019, clean 0.015, mixture 0.013; val_site — busy
0.030, clean 0.034, mixture 0.022. Inside the synthetic world the model
has learned the distinction: a background full of natural vessel
traffic is read as ≈ 0 added, at the same rate as a clean background.
The failure on real windows (§4) is therefore not "the model counts
natural traffic" in any simple sense — it does not do so on synthetic
busy backgrounds — but that real windows differ from *all* synthetic
modes in some further respect.

Pressure maps (main model, RMSE): by share band, val_random 0.036
(share 0) → 0.049 (0–0.1) → 0.075 (0.1–0.3) → 0.101 (0.3–0.6) → 0.131
(0.6–1.0); same monotone shape on all sets — high shares remain the
hardest, as in v1/v2. By construction order: drop-first 0.072 vs
mix-first 0.074 (val_random) — the two orders are equally learnable.
By site: Fish Hoek remains weakest (0.102 on val_site; Bird Island
0.070), on few windows as before. By mode: busy is the hardest
background type (0.080 val_random) but only slightly above clean
(0.071); mixture windows are the easiest (0.068).

## 8. Weaknesses and process notes (stated plainly)

- **All four trainings ran to the round cap** (best iterations 5997 /
  5932 / 5597 / 6000 of 6000; NM's early stopping never fired). The
  models may be slightly undertrained, as v1 was. Raising the cap is a
  legitimate knob for a future iteration, but a domain gap of this
  size (real mean 0.337 vs ≤ 0.20) is not a training-length problem.
- **Home-turf caveat**: §3's numbers are on v3-generator windows. §5
  exists precisely to discount them, and it does.
- The background-mode proportions (40/30/30), jitter magnitudes
  (±2 dB), drift magnitudes (±2.5 dB) and the traffic-quartile
  threshold are judgement calls. The withheld backgrounds are
  unobservable by definition (Phase 3), so these axes cannot be
  measured against the real thing — only their *effect* can, and §4
  measures it as nil.
- Ambient drift (point C) was folded into the v3 package as a whole
  and was **not separately ablated**; only mixtures were (brief point
  B asked for the mixture experiment explicitly). The package-level
  result (§4) bounds the combined effect of all background changes.
- Fish Hoek rests on 296 pool windows; its holdout numbers carry wide
  uncertainty in every generation.
- Process: the first v3 generation run produced 34 NaN labels
  (0.02% — an event landing entirely inside dropped minutes made the
  scaling divide by zero). Caught by a label-mean check, fixed with an
  explicit guard (such a window is a share-0 observation), all files
  regenerated and re-verified clean before any training. Separately,
  a first training run on pre-drift data was stopped and fully redone
  after Godson's written brief added the ambient-drift requirement;
  the superseded model files were deleted, not kept. The numbers in
  this report all come from the final data and final models.
- Versions: python 3.12.3, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1,
  lightgbm 4.7.0, scikit-learn 1.9.1. V3 main model file
  `models/lgbm_sim_v3.txt` (sha256 4c8d30b24722…). The v3 main model's
  test predictions exist only as a diagnostic artifact
  (`models/sim_v3_test_predictions.csv`); **no file was written to
  `submissions/` and nothing was submitted.**

## 9. Gate judgement

Godson's gate, unchanged: a candidate is submission-ready only at
**0.93 or above on known-answer validation**, with an **independent
cross-check agreeing**, and (this phase's addition) the real-window
prediction distribution **inside the plausible prior band**.

| criterion | required | measured | verdict |
|---|---|---|---|
| Main model, known-answer validation | ≥ 0.93 | 0.9271 / 0.9264 / 0.9280 / 0.9146 | **FAIL** (short by 0.002–0.015) |
| Cross-check models at the same standard | ≥ 0.93 | A 0.9194–0.9235, B 0.9238–0.9258 | **FAIL** on level |
| Cross-check agreement | agree closely | A–B RMSE 0.037–0.045 | PASS (agreement is strong; both agree below the gate) |
| Real-window distribution in prior band | mean ∈ [0.13, 0.20] | 0.3372 | **FAIL** (unchanged from v2's 0.330) |
| Construction admits 0.994-class recovery | oracle ≥ ~0.99 | 0.9983 | PASS |

**Judgement: NOT submission-ready. No candidate file, no submission.**
This is Godson's decision point; per his rule the response to a failed
gate is diagnosis and improvement, not a lowered bar.

## 10. Diagnosis — where the failure lives, and what it implies

What v3 *did* establish: (i) the organisers' label must be defined on
the observed window (§2) — our construction is now consistent with
the real 0.9943 ceiling, and missingness is nearly free in the
pressure maps; (ii) the model family is stable and reproducible —
two fully independent trainings agree at RMSE 0.037–0.045 and land
within 0.004 of each other on every set; (iii) inside the synthetic
world the model honours the label semantics on contaminated
backgrounds (§7). None of that is the bottleneck.

The bottleneck is unchanged and now much better localised. Four
models differing in training data (full, two disjoint halves,
no-mixtures) all read real windows at 0.335–0.339 while reading
synthetic windows — including deliberately contaminated, blended,
jittered and drifted ones — at the prior's 0.18. The real windows
therefore differ from **every** background mode we can generate from
the release, in a way that survives all background-side variation we
applied. With the background side exhausted this phase, the remaining
candidate locations for the gap, in the order the evidence points:

1. **The passage side.** Our additions are built with the starter
   notebook's `minute_band_power` conversion, which the notebook
   itself calls an unverified reconstruction of the organisers'
   export. If their VTUAD→band conversion differs systematically
   (segment length, windowing, band integration), our model has
   learned the wrong "added" signature, and no background treatment
   can compensate — consistent with everything measured here,
   including why busy-background training did not transfer: the model
   distinguishes *our* natural traffic from *our* additions perfectly,
   but the real additions may not look like ours.
2. **A background dimension not varied**: e.g. a per-band channel
   response larger or less smooth than ±2 dB, or a different minute
   averaging that changes within-window temporal statistics (our
   features lean heavily on per-band temporal std and minute shares).
   The release's own inter-site spectral differences are the grounded
   source from which such variation should be drawn (Godson's rule:
   variation grounded in the released recordings).
3. **The share prior's shape** (not its moments, which check out
   against both public scores): e.g. many real windows carrying small
   additions that our features read as moderate. This is the weakest
   candidate — it would not by itself produce a uniform +0.16 shift
   across four models — but it is not excluded.

The implied next iteration (for Godson's decision; this report
proposes no submission): attack location 1 first — treat the passage
conversion itself as a simulator axis (alternative segment lengths,
window functions and band-integration rules for `minute_band_power`),
and judge each variant by the only honest transfer diagnostic we
have: whether the trained model's real-window prediction distribution
moves toward the prior band *without anything being fitted on the
test windows*, backed by identical-windows cross-matrices so a
synthetic-easiness artefact cannot masquerade as progress again.
