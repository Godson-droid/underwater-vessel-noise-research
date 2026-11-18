# Koleshjr VTUAD declarations — transcription and audit

Source: screenshots supplied by Godson from his signed in Zindi session, 2026-11-17.
Two posts, both by Koleshjr. Discussion 35461 is the original pilot declaration
(20 Oct 2026, 23:29, title "External data declaration — VTUAD v1.3.2 bounded
validation pilot"). Discussion 36170 is the addendum ("External data declaration
addendum — expanded VTUAD strict-vessel source bank").

## Discussion 36170 (addendum), transcribed

Update to my earlier VTUAD declaration (https://zindi.world/discussions/35461):
I am expanding my training-side source-bank experiment using the same public
VTUAD (rebuilt) v1.3.2 strict-interval release by Hao Yuan, Wenbo Wang, Tian Li,
Guici Chen, Lingjiang Zeng and Sixian Liu:
https://zenodo.org/records/23037895, DOI 10.5281/zenodo.23037895, CC BY 4.0.
Upstream audio is from Ocean Networks Canada (Oceans 3.0). The record provides
vtuad_strict_wav_vessel.zip.part01–.part03, registry_anonymized.csv and
strict_interval_wav_SHA256SUMS.txt without a private account.

The additional tier comprises 36 strict vessel WAVs selected from the public
release using metadata and verified against the released per-WAV SHA-256
ledger. Each has at least three actual recorded minutes. They are from hulls
and dates disjoint from my original 18-clip evaluation pilot. I convert the
first three minutes of each mono 32 kHz recording into the challenge's 25
frequency bands using 2-second Hann periodograms and linear-energy averaging
over complete minutes. These profiles are used only to generate training-side
controlled additions with the released South African backgrounds. The source
audio remains available from the provider; I do not redistribute it.

The frozen evaluation windows and held-out pilot source groups remain the same
for the paired comparison. Test.csv is used only for inference, not fitting,
calibration, source matching or generator selection. This update adds no other
external dataset and does not use the VTUAD dedicated-background archive. I
retain the VTUAD and ONC attribution, exact file identities, conversion code
and split records for reproducibility.

## Discussion 35461 (pilot), transcribed

For reproducibility and equal access, I plan to investigate VTUAD v1.3.2, the
public ONC-derived acoustic dataset also cited in the official starter
notebook: https://zenodo.org/records/23037895 (DOI: 10.5281/zenodo.23037895).

The challenge starter attributes these components under CC BY 4.0. I will
verify the record and per-stream licence notices before using recordings,
retain the required attribution, and record the exact downloaded files and
checksums. Access is through the public Zenodo record; no private recordings
will be used.

For an initial bounded pilot, I will use a small documented recording subset,
convert one waveform channel into complete-minute mean-square frequency-band
summaries using the actual sample rate, and preserve unsupported bands as
missing. The initial conversion will follow the starter's 2-second Hann PSD
integration approach, averaging in linear energy before logarithms.
Uncalibrated digital samples will not be represented as calibrated physical
pressure.

I will keep source recordings and documented vessel groups separate across
development partitions before generating any mixtures with released South
African measurements. AIS-derived source labels remain weak supervision. I
will not infer hidden test source identities, use test inputs for training or
adaptation, or reconstruct test answers. Exact preprocessing, splits and seeds
will be retained with the solution; I am not redistributing the source files.

## Audit against Godson's five requirements (2026-11-17)

1. Recording inventory: PARTIAL. The addendum gives a count (36 strict vessel
   WAVs, each at least three recorded minutes) and states they are disjoint in
   hulls and dates from his original 18-clip pilot. It does not publish exact
   filenames. He states exact file identities are retained privately.
2. Source provenance: MET, as far as a public declaration can. Same public
   VTUAD (rebuilt) v1.3.2 strict-interval release, Zenodo record 23037895,
   DOI, CC BY 4.0, authors named, upstream Ocean Networks Canada (Oceans 3.0),
   archive parts named (vtuad_strict_wav_vessel.zip.part01 to part03,
   registry_anonymized.csv, strict_interval_wav_SHA256SUMS.txt), selection
   verified against the released per-WAV SHA-256 ledger, which also confirms
   the WAVs are unchanged.
3. Passage diversity: The 36 are new relative to his own 18-clip pilot only.
   They are drawn from the same public release that defines the pool for
   everyone. There are no scenarios or recordings beyond the VTUAD release.
   Our own harvest already converts 542 clips from this archive, so relative
   to our pool this adds nothing. By Godson's taxonomy this is previously
   unused recordings from the 509 scenarios, not new source material.
4. Split integrity: ASSERTED, not demonstrated. Hull and date disjointness
   from the pilot, vessel groups kept separate across development partitions,
   frozen evaluation windows and held-out pilot source groups unchanged. No
   split lists are published.
5. Incremental value: NONE OFFERED. The addendum is a declaration, not a
   results post. No comparison against any baseline, no holdout numbers.

## Verdict

Audit only, not approved for incorporation, and on the evidence now available
there is nothing to incorporate: the expansion is internal to Koleshjr's own
experiment (18 pilot clips growing by 36 recordings from the same public
release our simulator already draws on far more widely). His method is
careful and his provenance documentation is good. One notable detail: his
conversion follows the starter approach (2-second Hann, complete minutes),
with no mention of the edge band treatment our Phase 5 identified as the
transfer fault. Classification: Measured (from the post text itself).

## Godson's final verdict, 2026-11-17 (workstream PERMANENTLY CLOSED for Phase 7)

- Source expansion: genuine within Koleshjr's experiment.
- New material for our project: none.
- Conversion compatibility: NOT established. His declaration uses the
  starter-style two-second Hann conversion and does not document the corrected
  edge band treatment Phase 5 proved was a major source of transfer failure.
  A larger source pool processed through an uncorrected conversion is not
  assumed to improve our model.
- Performance benefit: untested.
- Decision: exclude from Phase 7 training additions. Do not spend time
  reproducing his 36 recordings. Remaining effort belongs on the three
  substantive experiments: background-relative features, held-out site
  character synthesis, and leave-one-site-out ensembling.
- Recorded as a negative result for the writeup: a competing data expansion
  claim was investigated, its provenance and source overlap verified, and
  incorporation rejected for a concrete reason.
- The Phase 5 baseline remains the comparison standard; the 0.96 gate and
  submission restrictions are unchanged.
