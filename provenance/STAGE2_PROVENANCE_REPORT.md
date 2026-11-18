# Stage 2 Provenance Report — the reference conversion `minute_band_power`

**Date:** 18 November 2026 (Africa/Lagos)
**Scope:** Stage 2 Area A only — starter-notebook conversion provenance.
**Controlling references:** `RELEASE_AUDIT.md` §2, `release_docs_stage1/DATA_SOURCES.md` §3.
**Original file:** `code/build_passage_pool.py` — preserved unchanged.
**Stage 2 working files:** `stage2_work/clean_conversion.py`,
`stage2_work/equivalence_test.py`, `stage2_work/equivalence_results.txt`.

---

## 1. The historical implementation, documented

**Location.** `code/build_passage_pool.py`, lines 26–51 (function
`minute_band_power`), inside a module whose header docstring (lines 1–11)
states that the conversion is "the starter notebook's own reference
conversion (cell 11, `minute_band_power`)". The function's own docstring
reads, verbatim: *"Verbatim copy of the starter notebook's cell-11
reference conversion."* The starter notebook is organiser-supplied
competition material with no licence stated anywhere in the tree
(DATA_SOURCES.md §3). Attribution is present; redistribution permission
is not established. That is the provenance problem this Stage 2 area
addresses.

**Signature and contract.**

```
minute_band_power(samples, sample_rate_hz) -> ndarray, shape (n_minutes, 25)
```

- Input `samples`: one-dimensional, finite, cast to float64. Raises
  `ValueError` if not 1-D or non-finite.
- Input `sample_rate_hz`: must be a positive integer (the recording's
  actual sample rate); raises `ValueError` otherwise.
- Output: **linear** band power per complete minute. The function does
  **not** return dB. The relative-dB step (10·log10 of power over the
  clip's median power, floored at 1e-30) is performed separately by its
  caller, `convert_wav_bytes` in the same module.

**Mathematical operation, exactly as the historical code defines it.**

1. Segment length is `2 * fs` samples (non-overlapping 2-second
   segments). Only complete minutes are used:
   `minutes = samples.size // (30 * segment_size)`; trailing samples
   are discarded. Fewer than one complete minute raises `ValueError`.
2. The trimmed signal is reshaped to `(n_segments, segment_size)` and
   passed to `scipy.signal.periodogram` with `window="hann"`,
   `detrend="constant"` (each segment's mean is removed before
   windowing), `return_onesided=True`, `scaling="density"`, `axis=1`.
   The FFT length is the segment length (no zero-padding), so the bin
   spacing is `df = fs / (2 fs) = 0.5 Hz` at any sample rate.
3. Band centres are `10 ** (1.5 + 0.1 * k)` Hz for k = 0..24
   (31.62 Hz … 7943.28 Hz). Band edges are `centre / 10**0.05` and
   `centre * 10**0.05` (deci-decade bands, ±0.05 decade).
4. A bin belongs to a band when `low <= frequency < high`. Band power
   is the density summed over the band's bins, multiplied by `df`
   (a rectangle-rule integral of the density over the band).
5. Frequency limit: a band is computed only when its upper edge
   satisfies `high <= fs / 2` and at least one bin falls inside it;
   otherwise the band's entries remain **NaN**. At fs = 16 kHz, for
   example, band 24 (upper edge ≈ 8912 Hz > 8000 Hz) is NaN.
6. The per-segment powers are reshaped to `(minutes, 30, 25)` and
   averaged over the 30 segments **in linear units**, giving one
   25-band vector per minute.

Two properties worth stating plainly. First, band 0 in this historical
function is **clipped** at its deci-decade lower edge (≈ 28.18 Hz): it
does not integrate from 0 Hz. The extended edge-band treatment
("bandext": band 0 from 0 Hz, band 24 to Nyquist) that Phase 5 later
proved necessary is a different conversion, implemented in other
programme files — it is not what this function computes, and the clean
reimplementation below deliberately reproduces **this** function, not
bandext. Second, the function's role in the programme is limited to
the passage-pool harvest (`build_passage_pool.py`); the competition
windows themselves arrive pre-converted from the organisers.

## 2. The clean reimplementation — independent method

**File:** `stage2_work/clean_conversion.py`, function
`minute_band_power_clean(samples, sample_rate_hz)`, same input/output
contract as the historical function.

**Sources used:** the published mathematical definition of the
conversion as documented in DATA_SOURCES.md §3 and the programme
reports (non-overlapping 2 s segments, mean removed, one-sided Hann
periodogram with density scaling, power integrated over each
deci-decade band, 30 segments averaged in linear units per complete
minute), and the documented band grid (centres `10^(1.5 + 0.1 k)`,
k = 0..24, edges at ±0.05 decade — the same grid as the competition's
15×25 summaries, also recorded in `code/common.py`).

**How it is written (its own structure, not an adaptation of the
historical source):**

- The Hann taper is computed directly from its defining formula,
  w[n] = 0.5 − 0.5·cos(2πn/N) (the periodic form used for single-frame
  spectral analysis).
- The periodogram is computed directly with `numpy.fft.rfft`:
  density = |DFT(w·x)|² / (fs · Σw²), with the one-sided doubling
  applied to every bin except DC and (for even frame length) Nyquist.
  `scipy.signal.periodogram` is not called.
- The signal is grouped as `(minutes, 30, segment)` up front and
  demeaned per segment; band integration uses boolean bin masks and
  the bin width from `numpy.fft.rfftfreq`.
- The historical function was consulted only to document it (Section 1
  above) and as the comparison oracle in the equivalence test — not as
  a template for the new code.

**Assumptions made, each forced by the definition and each verified
numerically by the equivalence test:**

1. "Hann window" means the **periodic** Hann form (denominator N), the
   standard for frame-based spectral analysis — as opposed to the
   symmetric form (denominator N−1, e.g. `numpy.hanning`). This is the
   single assumption that could have changed the numbers materially;
   the measured deviations (Section 3) confirm it.
2. FFT length equals the segment length; no zero-padding.
3. Density scaling is 1 / (fs · Σw²), with one-sided doubling of all
   bins except DC and Nyquist.
4. Bin membership is `low <= f < high`; band power is the rectangle
   sum of density × bin width.
5. A band whose upper edge exceeds Nyquist yields NaN, per the
   documented frequency limits.
6. Segment means are removed before windowing; the 30 segments of a
   minute are averaged in linear units.

## 3. Equivalence test — method and results

**Harness:** `stage2_work/equivalence_test.py`, run with the project
environment (`.venv`, numpy 2.5.5, scipy 1.18.1 — the versions recorded
in the manifest). Full output: `stage2_work/equivalence_results.txt`.

**Loading the historical function safely.** The historical module was
**not imported**. Its source was parsed with `ast`, the single
`minute_band_power` definition extracted, and only that function
executed in a fresh namespace providing `np` and scipy's
`periodogram`. (The module level of `build_passage_pool.py` was first
assessed: it is inert — imports, pinned checksum constants and function
definitions, with `main()` guarded by `if __name__ == "__main__"`, and
its helper `common.py` likewise defines constants and functions only.
Isolation was still chosen so that no module-level code, archive
reassembly (`strict_vessel_zip`) or `main()` can run, and no bytecode
cache is written into the research tree.)

**Signals:** twelve synthetic signals generated in memory — silence;
pure tones at the centres of bands 0, 12 and 24; a tone exactly on the
shared edge of bands 10/11; a 10 Hz tone below band 0's lower edge;
seeded broadband white noise; a 20 Hz → 12 kHz chirp; a mixed
band-0 + band-24 + noise signal; a two-minute noise signal; and, at
fs = 16 kHz, white noise and a band-24-centre tone (the above-Nyquist
edge case). No competition data, no VTUAD audio, no caches.

**Results** (deviations are clean minus historical, on linear power):

| Signal | fs | Max abs dev | Mean abs dev | Band 0 max dev | Band 24 |
|---|---|---|---|---|---|
| silence | 32000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| tone, band 0 centre | 32000 | 8.771e-15 | 3.508e-16 | 8.771e-15 | 1.620e-30 |
| tone, band 12 centre | 32000 | 8.771e-15 | 3.508e-16 | 5.436e-28 | 2.316e-30 |
| tone, band 24 centre | 32000 | 8.549e-15 | 3.419e-16 | 1.070e-31 | 8.549e-15 |
| tone, band 10/11 edge | 32000 | 5.274e-15 | 3.531e-16 | 2.317e-27 | 1.087e-30 |
| tone, 10 Hz (below band 0) | 32000 | 1.109e-32 | 2.060e-33 | 3.637e-34 | 4.394e-33 |
| white noise (seeded) | 32000 | 1.985e-15 | 3.861e-16 | 8.023e-18 | 1.985e-15 |
| chirp 20→12000 Hz | 32000 | 1.207e-15 | 2.581e-16 | 8.470e-22 | 1.207e-15 |
| mix band0+band24+noise | 32000 | 4.496e-15 | 3.748e-16 | 4.191e-15 | 4.496e-15 |
| white noise, 2 minutes | 32000 | 2.082e-15 | 3.880e-16 | 7.373e-18 | 2.082e-15 |
| white noise | 16000 | 1.832e-15 | 3.699e-16 | 8.023e-18 | both NaN |
| tone, band 24 centre | 16000 | 1.157e-28 | 1.174e-29 | 1.214e-31 | both NaN |

**Summary:** worst maximum absolute deviation across all signals and
bands: **8.771e-15** (linear power units). Worst mean absolute
deviation: **3.880e-16**. Expressed against each signal's own peak band
power, the worst deviation is **1.812e-14** relative. NaN positions:
**identical in every signal (0 mismatches)**, including band 24 at
fs = 16 kHz, where both implementations return NaN because the band's
upper edge exceeds Nyquist. Output shapes matched in all twelve cases.

Two relative figures in the raw results need context so they are not
misread: for the 10 Hz tone and the fs = 16 kHz band-24 tone, the
"max abs dev / peak band power" ratio is larger (4.7e-06 and 1.8e-10)
only because those signals deposit essentially no power in any
computable band (their energy sits below band 0's edge, or in the
NaN band), so the peak band power is itself at leakage level
(~1e-27). The absolute deviations there are 1.1e-32 and 1.2e-28 —
roundoff on nothing.

## 4. Assessment — is this sufficient for a later proposed replacement?

**Assessment (Stage 2 author's, for owner review; it authorises
nothing):** yes, the evidence is sufficient to *support proposing* the
clean implementation as the release copy's replacement for the copied
function, for three reasons. (1) The deviations are at double-precision
roundoff level (≤ 9e-15 absolute, ≤ 2e-14 relative to peak) across
signals chosen to stress exactly the places two implementations of
this definition could diverge: band centres, a band edge, sub-band
energy, broadband energy, a sweep, silence, multiple minutes, and the
above-Nyquist edge band. (2) The NaN behaviour — the one place the
definition involves a discrete rule rather than arithmetic — matches
exactly. (3) The function is pure arithmetic on its input; there is no
state, randomness or data dependence in which a divergence could hide
that these signal classes would not exercise.

**Limits of the evidence, stated plainly.** Equivalence was shown on
synthetic signals at two sample rates (32 kHz and 16 kHz), not on real
VTUAD audio; running it on real audio was neither necessary (the
function is signal-agnostic arithmetic) nor permitted under Stage 2's
no-data-handling scope. Equivalence is claimed only for the documented
definition above — including the clipped band 0 — and not for the
programme's later bandext conversion, which lives in other files and
is untouched by this question.

**Outstanding concerns.**

- The starter notebook's licence remains unstated; this work removes
  the *need* to redistribute the copied code in a future release copy,
  but the historical function stays in the frozen record, attributed,
  exactly as DATA_SOURCES.md §3 describes. Any actual replacement in a
  release copy requires separate owner review and approval.
- One residual judgement call is recorded for the reviewer: the choice
  of the periodic Hann form (Assumption 1) is confirmed numerically
  against the historical implementation, which is the strongest check
  available, but it is a confirmation against the copied code, not
  against the notebook itself (which is organiser material and was not
  re-read for this task).

**Verification record for this area.** Original file path:
`code/build_passage_pool.py` (untouched; see the final archive
checksum verification). Working copies: `stage2_work/clean_conversion.py`
(new), `stage2_work/equivalence_test.py` (new),
`stage2_work/equivalence_results.txt` (captured stdout of the test
run, exit code 0). Change made to the original: **none**.
