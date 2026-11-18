# Step 3, Stages 0 and 1: physical energy decomposition screen, results (2026-11-18)

Authorisation and scope (Godson, 2026-11-18). This work was approved
under Option A of PROPOSAL_step3_decomposition.md: Stages 0 and 1
only. Stage 0 establishes an independent synthetic validation,
including its oracle and baseline checks. Stage 1 tests the two
minimal decomposition estimators, S1 (temporal envelope) and S2
(spectral family decomposition), against the predeclared null and
performance criteria of proposal Section 6.4. Any kill criterion ends
the work. An intermediate result means stop and report, not adjust
thresholds. Stage 2 requires a separate decision and was not started,
sketched into code, or pre-built. No candidate file was produced, no
Zindi submission was made, and no submission slot was spent (slots
remain 1 of 200). No inference was run on the real Zindi test set at
any point; all material below is validation material built for this
purpose. The frozen archive was not modified; all new work lives in
step3/.

Godson's qualification, binding on this record. The construction
below is independent in implementation and held-out material, but it
still uses the same underlying public data sources and synthetic
energy additive assumptions as the frozen programme. It is an
independent synthetic validation. It must never be represented as
independent real-world validation, and every I1 number in this record
carries that qualification.

Finding classifications: Measured / Supported / Unresolved /
Rejected, exactly one per finding.

## 1. The predeclared lines (proposal Section 6.4, unchanged)

The null check runs first. On pure backgrounds from held-out sites
under the independent construction (true share zero), an estimator
passes only if its mean estimated share is at most 0.03 and its
median at most 0.02.

Kill, if any of: the null check fails; or the I1 overall score is
below 0.93; or site zero-band bias is above +0.040; or the (0.6, 1]
share-band bias is more negative than -0.070.

Promise (the only outcome that may recommend Stage 2): I1 at least
0.96 overall, and the unseen-site subset at least 0.93, and zero-band
bias at most +0.030, and (0.6, 1] bias no more negative than -0.055,
and the null check passed.

The bias lines are applied to the unseen-site subset, matching the
frozen record's usage: the reference numbers behind these lines are
site-holdout statistics from Phases 6 to 8B.

## 2. Stage 0: the independent construction (I1)

### 2.1 Material split

Passage clips (VTUAD v1.3.2 strict set, 525 clips with hull and date
metadata). Hulls and dates were each assigned to a side (rectangle
assignment refined by seeded simulated annealing), and a clip was
kept only where its hull assignment and its date assignment agree.
Result: family side 212 clips (197 vessel label, 15 ambient label;
83 hulls; 15 dates); I1 side 211 clips (195 vessel label, 16 ambient
label; 94 hulls; 18 dates); 102 clips excluded because their hull
and date assignments disagreed; the 17 class-archive clips (sailing,
tanker, rescue, pilot vessel) were excluded because the workspace
holds no hull or date metadata for them. Shared hulls between the
kept sides: 0. Shared dates: 0. Stage 0's own kill criterion (hull
disjoint and date disjoint material of usable size cannot be
assembled) does not trigger. Measured.

Background windows (the released pool of 87,653 screened 15-minute
windows). Bird Island and Fish Hoek, the frozen programme's unseen
sites, are held out entirely: their windows carry only the roles
eval_unseen (2,909) and null (550, disjoint from eval_unseen). Within
the six family sites, the latest 15 percent of windows by start time
form a period holdout (12,629, excluded from all family fitting);
the remainder splits into disjoint fit (35,782) and eval_seen
(35,783) halves. Measured.

### 2.2 Construction

Fresh code in step3/ (i1_common.py, gen_corpus.py); no archived
simulator, converter or label code is imported or reused. The
converter implements the settled bandext definition (band 0
integrated from 0 Hz, band 24 to Nyquist, 2-second Hann segments,
periodogram density scaling, per-minute means). Verified against the
archived bandext conversion of an unvaried clip: maximum absolute
difference 0.0006 dB, mean difference 0.0000. Measured.

Variations are applied at the recording level, before conversion,
following the organisers' stated classes: event assembly from
same-side, same-kind clips with per-piece gain jitter of plus or
minus 3 dB; time warp by FFT resampling, factor log-uniform in
[0.5, 2.0]; reversal with probability 0.25; per-band colouring from
a smooth random walk clipped at plus or minus 7 dB; per-minute gain
envelope walk clipped at plus or minus 3.5 dB. One construction
property is recorded for honesty: FFT resampling discards source
energy above the resampled Nyquist, as any standard resampling
does, so strongly warped broadband events lose some top-band
energy; sine amplitude at retained frequencies is preserved.

Backgrounds are drawn from the pool under the role plan of Section
2.1, with the measured background realism modes (energy-sum mixture
of two same-site windows with probability 0.30, per-band channel
jitter within plus or minus 2 dB with probability 0.5). The share
prior is deliberately different from the frozen construction: 25
percent exact zero, otherwise uniform on [0, 0.9]. Event kind among
nonzero draws: vessel 0.7, ambient 0.3. Event durations follow the
empirical frozen-corpus duration distribution. Missingness follows
the measured maskfix form (whole minutes only). Mixing is linear in
energy under a single gain set to hit the drawn target share over
the settled scaling domain (observed cells for drop-first order,
all cells for mix-first, orders equally likely). Summaries are
rounded to the 0.1 dB grid. Labels are the realised share over
observed cells, computed from the unquantised component energies.

Corpora produced. Calibration set: 1,500 windows, family-side clips
and fit-partition backgrounds only (label mean 0.3445, zero fraction
0.246; kinds: vessel 777, ambient 355, none 368). Main set (the
scored I1 set): 2,000 windows, I1-side clips; backgrounds 1,192
eval_seen, 293 period holdout, 515 unseen-site (label mean 0.3346,
zero fraction 0.264; kinds: vessel 1,055, ambient 418, none 527).
Null set: the 550 null-partition windows with the mask applied and
nothing added. Estimator-facing files carry no truth; truth is held
in separate files read only by the scorer, the oracle check and the
hand-check.

Process note, recorded plainly: the first generation run kept its
windows in memory and was destroyed, work and all, by a virtual
machine replacement at about 1,250 of 1,500 calibration windows.
Generation was rewritten to save 250-window chunks with independent
per-chunk seeds and to skip chunks already on disk; the rerun
completed, and a second silent process death later cost nothing
beyond one chunk boundary. The corpora on disk are the chunked
build.

### 2.3 Verification

Disjointness, re-verified independently of the splitting code from
the saved split files: 0 shared hulls, 0 shared dates between the
kept clip sides; Bird Island and Fish Hoek carry only eval_unseen
and null roles; the null and main unseen-site window sets are
disjoint partitions. Measured.

Hand-check: labels for a random sample of 12 main windows were
recomputed from the stored component energies by a separate
explicit-loop implementation. Worst label difference: 2.7e-09.
Worst summary difference against the component sums: 6.1e-06 dB.
The harness arithmetic is exact. Measured.

Oracle check: the true-background residual estimator on the main
set (share from the stored true background energies against the
quantised mixture summaries) scores 0.9989 (oracle mean 0.3345
against label mean 0.3346). This lands at the construction ceiling
of about 0.998 stated in the proposal: under this construction the
labels are recoverable when the true background is known, so the
estimand is well posed here and the remaining difficulty is
estimation, exactly the question Stage 1 asks. Measured, with the
standing caveat from the proposal that a ceiling computed under a
construction's own label definition is partly an identity and does
not speak to simulator-to-real mismatch.

Baseline check: the frozen Phase 5 baseline (bandext_maskfix
booster and its 549-feature transform, run read-only, unmodified)
scored on I1 material: overall 0.8854; eval_seen subset 0.8909;
period holdout 0.8771; unseen-site subset 0.8783; unseen-site
zero-band bias +0.0579; prediction mean 0.3008 against a label mean
of 0.3346. On the I1 null set it reads mean 0.0471, median 0.0340,
against its in-family true-zero readings of 0.014 to 0.018 recorded
in the frozen programme. Classification, stated as the proposal
requires: the baseline is materially weaker on I1 than in family,
which is evidence that I1 is a genuinely different and harder test
rather than a restatement of the frozen construction; the baseline
score does not by itself validate I1, and it is reported as a
reference point only. It also sets an honest expectation: on this
material the frozen regressor would itself fall below the 0.93
floor of the Stage 1 lines. Supported.

Stage 0 stands: the independent synthetic validation exists, its
material is disjoint as specified, its harness is verified, its
oracle lands at the stated ceiling, and its baseline check is
recorded. No Stage 0 kill criterion fired.

## 3. Stage 1: the minimal estimators

### 3.1 Reference families (family-side material only)

Background family, from 6,000 fit-partition windows: pooled median
spectrum shape, per-site median shapes for the six family sites,
and the top 4 principal components of window log-shape variation.
Passage family, from family-side clips converted fresh without
variation: prototype minute-spectrum shapes from 764 vessel minutes
(pooled median, plus per-class medians for cargo, fishing, other,
passenger ship, pleasure craft and tug) and 61 ambient minutes.

### 3.2 S1 fit

S1 takes, per band, a low quantile of the window's minute energies
times the number of observed minutes, corrected by a per-band
factor fitted on the calibration corpus. The quantile grid
{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35} gave calibration RMSE
{0.3573, 0.3619, 0.3572, 0.3480, 0.3408, 0.3362, 0.3286}; the best
level, q = 0.35, sits at the grid edge, and even there the
calibration score is only 0.6714 on S1's own training side. The
minimal temporal envelope was already weak in family before any
held-out test. Measured.

### 3.3 Implementation incident, recorded before the results

The first null run produced an apparent S1 pass (mean 0.0132). That
number was an artefact: the prediction code summed a cell-level
mask where a minute count was required, inflating the background
estimate about 25-fold and driving predictions to near zero on pure
backgrounds. The defect was caught on review of the code path
before any reliance was placed on the number, the code was fixed,
the stale prediction file was deleted, and a smoke test on the
fixed code showed the opposite behaviour. Every Stage 1 number
below comes from the fixed code. The incident is recorded here
because the first figure existed and was wrong; it plays no part in
any verdict.

### 3.4 The null check (run first, as predeclared)

On the 550 pure-background windows from the held-out sites (true
share exactly zero; pass requires mean at most 0.03 and median at
most 0.02):

S1: mean estimated share 0.5747, median 0.6285. FAIL.
S2: mean estimated share 0.0999, median 0.0920. FAIL.

Both minimal estimators fail the null check. Under the binding
order, Stage 1 ends here: neither estimator was run on the main
set, and no I1 overall score, unseen-site score or share-band bias
was produced for either estimator, because the predeclared sequence
does not reach those numbers once the null fails. No thresholds
were adjusted, no variant was tried, and none will be under this
authorisation.

### 3.5 Reading of the failures

S1 mechanism. The per-band correction is fitted on calibration
windows in which events are present and typically occupy most of a
window's minutes, so the fitted envelope is event-contaminated and
the correction factor learns to discount it. On a pure background
there is no contamination to discount, the background is
under-restored, and the missing background energy is read as
vessel share, at more than half the window on average. The failure
is structural to fitting an envelope correction under contamination,
not a matter of the quantile level. Supported.

S2 mechanism. The background family is fitted on family sites. At
the held-out sites it cannot express the local background character
closely enough, and the joint fit attributes the unexplained
background energy to a passage component: about a tenth of pure
background energy at Bird Island and Fish Hoek is read as an added
passage that is not there. This is the same low-share hallucination
the frozen programme measured in the gradient-boosted regressor,
reproduced by an unrelated mechanism on independently built
material. Supported.

Context, Measured: the frozen Phase 5 regressor also fails this
null bar on the same 550 windows (mean 0.0471, median 0.0340), but
by a small multiple of the line, where S2 fails by about a factor
of three on the mean and S1 by about a factor of nineteen. The null
bar is strict by design; the decomposition estimators do not come
close to it.

Joint reading, Supported: three unrelated estimation mechanisms
(the frozen regressor, a temporal envelope, and a spectral family
decomposition) all over-read share on pure backgrounds from unseen
sites under the independent synthetic validation. The persistent
estimation difficulty under every model, feature and variation
strategy tested sits in unseen-site background characterisation.
That is a statement about the strategies tested, in the qualified
form Godson requires; split identifiability from the summaries
remains Unresolved, as it was in the frozen record.

## 4. Outcome

A kill criterion fired. Stage: Stage 1, at the null check, which
the predeclared order runs first. Numbers: S1 mean 0.5747 and
median 0.6285 against pass lines of 0.03 and 0.02; S2 mean 0.0999
and median 0.0920 against the same lines. Both minimal estimators
are killed. The promise line was not met, so no Stage 2
recommendation arises from this work, and Stage 2 remains a
separate decision for Godson. Nothing in this record argues for
Stage 2 on performance grounds.

The yield of Stages 0 and 1 is twofold. First, a verified
independent synthetic validation now exists (Stage 0 stands in
full: disjoint material, exact harness, oracle at the stated
ceiling, baseline check recorded), and it can be reused if Godson
ever authorises further work. Second, a clean negative result:
estimating the physical background contribution by either minimal
route does not clear the first honesty screen on held-out sites.
The decomposition hypothesis, in its minimal forms, does not offer
what the previous regression approach could not. That was the
question Option A was approved to answer, and it is answered.

Artefacts. All under step3/: splits.py, splits.json and
clip_sides.csv (material split); bg_roles.npz (background roles);
extract_wav.py and wav_cache/ (waveform cache); i1_common.py and
gen_corpus.py (construction); corpus/ (calibration, main and null
sets, chunked with merged estimator-facing and truth files);
verify_stage0.py and stage0_verification.json (Stage 0 checks);
families.py and families.json (reference families and the S1 fit);
estimators.py, run_stage1.py, stage1_results.json, null_pred_s1.npy
and null_pred_s2.npy (Stage 1). The frozen archive is untouched.
