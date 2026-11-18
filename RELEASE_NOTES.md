# Release Notes

Package: Underwater Vessel Noise, Zindi research programme
Candidate assembled: 18 November 2026
Status: published release. The accepted private candidate was published by owner decision on 18 November 2026. This publication copy differs from the accepted candidate in exactly three respects, each recorded here: a LICENSE file has been added carrying the BSD 3 Clause licence the owner selected on 18 November 2026; this licence section and the status lines have been updated to state that decision; and no other file has been changed. Every other file in this release is byte identical to the accepted candidate, as verified against its checksum inventory before upload.

## 1. What this package is

This package is a documented research record with partial reproducibility. It is not a fully reproducible executable project. It preserves the research programme's documentation, reports, code, Step 3 records and machine readable result records so that the reported findings can be inspected and checked against their records. It does not contain the data, generated corpora or trained model boosters that full regeneration would require, and no reader should treat it as able to regenerate the programme's results end to end. The approved README, which ships unedited as the front door of this package, describes the reproduction roadmap as a partial reconstruction in its sections 6 and 7, and those limitations stand.

## 2. Contents and sourcing

The candidate was assembled file by file from a curated manifest of 211 rows: 179 inclusions and 32 exclusions after all owner decisions were applied. Files were copied byte for byte from the project working tree, from the approved Stage 1 documentation set, and from the Stage 2 working area. No file was copied from the frozen research archive. The archive was used only as a read only reference for the identity check described in section 9. No file content was edited during assembly; author attribution is retained throughout, by owner decision.

Contents in outline: the approved documentation (README, requirements, data sources, licence decision record); the consolidated research report and the phase and Step 3 reports; the programme code set; the Step 3 scripts and records; 51 model result records and 25 data records in machine readable JSON form; the Stage 2 provenance materials, including the clean conversion implementation and its equivalence evidence; and the five Stage 2 process reports, included by owner decision as the audit trail for the provenance and hygiene claims made here.

## 3. Exclusions and their reasons

The following are excluded, in summary:

- The South African competition acoustics, in source and derived form, including every derived pool or corpus that could reproduce them. Their terms grant no permission to redistribute them. They are obtainable only through participation in the competition.
- The VTUAD passage archives and audio caches. VTUAD is cited as an external prerequisite instead: version 1.3.2, CC BY 4.0, Zenodo DOI 10.5281/zenodo.23037895. See DATA_SOURCES.md.
- The generated simulation corpora and the large trained boosters. Results that depend on them can be checked against the included JSON records but cannot be regenerated from this package.
- The four historical competition submission files, excluded by owner decision. The historical submission record is preserved in SUBMISSIONS.md. No submission shaped file is included in this package.
- The two opaque ID Step 3 derivatives (the per clip assignment record and the split record), excluded by owner decision on privacy grounds. The Step 3 findings stand on the included JSON verification records and written reports.
- The historical pool building script, code/build_passage_pool.py, excluded by owner decision because it contains a verbatim copy of the competition organisers' starter notebook conversion function, whose source licence is not established. Attribution alone does not establish redistribution permission. The file remains preserved in the private research record.
- The two Step 3 null prediction arrays, step3/null_pred_s1.npy and step3/null_pred_s2.npy, excluded uninspected. No authorisation to inspect their contents was given, and under the governing sequence an uninspected binary file is excluded rather than included on trust. The null test results they relate to are preserved in text form in the Step 3 records.
- Nothing further. A LICENSE file is included in this published release; see section 5.

## 4. The pool building limitation, stated plainly

The historical pool building step cannot be run as originally written from the contents of this release. The script that performed it is excluded for the provenance reason given in section 3, and the roadmap reference to it in the approved README therefore points to a file that exists only in the private research record.

In its place, the provenance folder contains clean_conversion.py, an independent reimplementation of the one conversion function that the excluded script copied, written from the published mathematical definition. Its equivalence evidence is a synthetic signal test: across 12 synthetic signals the worst maximum absolute deviation from the historical function was 8.771e-15, with NaN positions identical. That evidence covers the tested synthetic signals only; it does not establish equivalence for all inputs or sample rates.

Three limits must not be lost: the clean implementation replaces only that one conversion function; it is not a replacement for the complete historical pool building script, which has not been replaced anywhere; and it does not implement the Phase 5 bandext conversion, which is a different conversion recorded elsewhere in the programme's code and reports.

A related limitation stands for the Step 3 verification script: verify_stage0.py loads the Phase 5 baseline booster, models/lgbm_conv_bandext_maskfix.txt, which is not in this package and which the approved roadmap cannot regenerate. A reproducer needs that booster from the historical research environment.

## 5. Licence position

The owner selected the BSD 3 Clause licence for his own code and documentation in this package on 18 November 2026, and the LICENSE file in this release carries its full text. Readers may use, copy, modify and redistribute that material, including for commercial purposes, subject to the licence conditions: retention of the copyright notice, retention of the licence conditions, and no use of the owner name to endorse derived products. The licence carries a warranty disclaimer, stated in the LICENSE file.

This position supersedes the one recorded at assembly time. When the candidate was assembled and accepted earlier on 18 November 2026, no licence had been granted and all rights were reserved, and several documents in this package, including LICENSE_DECISION.md, record that earlier position in their text. Those documents are preserved unedited as the historical record of the position at the time they were written. The governing position for this published release is the one stated in this section and in the LICENSE file.

Third party material is not relicensed by anything here. VTUAD is not bundled; it is cited under its own terms in DATA_SOURCES.md.

## 6. Leaderboard figures

The competition figures quoted in this package are externally sourced leaderboard records, preserved in SUBMISSIONS.md and the programme reports. No leaderboard export or receipt exists in the research record, and the figures cannot be verified from this package. They are: the benchmark, 0.753867825; the public top score, 0.994250562; and Submission 1's public score, 0.687868143. Submission 1 was the programme's only competition submission, made with the early simulator before the later phases of work; the far stronger later models were never submitted, because they did not meet the programme's own gate and no submission without owner approval was permitted.

## 7. The research record in brief

The programme's development gate was a minimum score of 0.96 on fixed validation and both independent cross checks across all four holdouts. It was never met. The best holdout ever recorded was 0.9488 and the best site holdout was 0.9250. The figure of approximately 0.998 that appears in the reports is a submission standard and an oracle ceiling under the programme's own label definition, the score attainable with the true background known. It is not achieved model performance, and nothing in this package should be read as claiming it.

The central positive finding is from Phase 5: the passage conversion was the principal measured source of the early transfer failure, and correcting it transformed the results. Phase 7 and Phase 8 work then characterised a site calibration problem and bounded what character training could correct. Phase 8B used independent drawn share assignment with small measured realised label variation across its strength levels; labels were not identical across strength levels, and the reports must be read with that qualification.

All validation in this programme is synthetic. Synthetic validation is not independent real world validation. Step 3's construction was independent in implementation and held out material, but it used the same underlying public data sources and synthetic energy additive assumptions as the rest of the programme. Real window prediction distributions reported in the programme are unvalidated diagnostics, not prevalence estimates and not proof of accuracy.

Step 3 tested a physical energy decomposition hypothesis in two minimal forms. Its Stage 0, an independent synthetic validation construction, passed its checks. Its Stage 1 was killed at its first predeclared test, the null check on 550 pure held out site backgrounds with true share zero: estimator S1 read a mean estimated share of 0.5747 and estimator S2 read 0.0999, against a pass requirement of mean at most 0.03 and median at most 0.02. Neither estimator was run on the main evaluation set, so no overall, unseen site or band bias scores for S1 or S2 exist, and none may be inferred. The hypothesis stands rejected in its two minimal forms, and the result is preserved as a formal negative finding, including the disclosed implementation incident in the first S1 run.

## 8. Reproduction position

A reader can check every reported score that has a JSON record against the records included here, together with the configurations, seeds and provenance the records carry. A reader cannot regenerate the programme end to end from this package: the restricted acoustics, the generated corpora and the trained boosters are excluded for the reasons given, the pool building script is excluded for its provenance reason, and one verification step needs a booster the roadmap cannot regenerate. Recorded exceptions to in package traceability are stated in the reports: the leaderboard figures in section 6 are externally sourced; the B1 null figures are prose and code only; and the Phase 0 figures trace to a data audit document that sits outside the frozen archive and is included here as a document with that status stated.

## 9. Assembly verification

At assembly, every one of the 76 record JSON files in this candidate (51 model records and 25 data records) was compared individually, by SHA 256, against its frozen archive counterpart. All 76 matched. Fresh hygiene scans were run on the candidate itself, covering vessel identifier strings, original clip keys, local machine paths, credential like patterns and the exact competition submission schema header. The results, the full candidate checksum list and the post assembly archive re verification are recorded in the dated assembly report held with the project's planning documents. Earlier Stage 2 scans describe the Stage 2 working area as it then stood; they are not this verification.

## 10. Status

This release is published. The candidate was assembled under a specific owner authorisation limited to assembly, was independently reconciled and was formally accepted by the owner as a private candidate. The owner subsequently recorded three separate decisions on 18 November 2026: the write up prize entry was closed under the existing research decisions; publication of the candidate as an independent scientific record was approved; and the BSD 3 Clause licence was selected. This publication copy was prepared and uploaded under the owner authorisation of those decisions. Binary inspection of the two excluded Step 3 arrays remains unauthorised and has not taken place.
