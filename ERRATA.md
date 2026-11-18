# Errata and Document Status

Repository: Underwater Vessel Noise Research
Applies to: the published release, initial commit 0dd1f9be9f12c840fa8be9c77fad9974bb1c891e
Errata date: 18 November 2026

## Purpose

This release preserves its historical documents unedited, as a matter of record. Several of those documents were written before later stages of work were completed and before the publication and licence decisions were taken, and their status wording is therefore out of date. This errata reconciles the public record. It changes no research finding, no result and no file other than by adding this note.

The governing documents for the published release are the LICENSE file, RELEASE_NOTES.md and this errata. Where any other document in this repository conflicts with them on publication status, licence status or the completion status of the Stage 2 provenance work, the governing documents prevail and the other document is to be read as the historical record of its date.

## Item 1. README status wording

The README opens by stating that no public release has been authorised yet and that no licence has been chosen. That was accurate when the README was approved on 18 November 2026, before the publication decisions were taken later the same day. It is not the current position. The owner authorised publication on 18 November 2026, this repository is that publication, and the licence was selected the same day. See Item 5. The README is preserved unedited by owner decision, as the approved front document of the research record.

## Item 2. DATA_SOURCES.md and the Stage 2 provenance work

DATA_SOURCES.md states that the clean reimplementation of the starter conversion and its equivalence check had not happened yet and were Stage 2 work. That statement is stale. Both were completed in Stage 2 and both are included in this release. The reimplementation is provenance/clean_conversion.py, whose function minute_band_power_clean was written from the published mathematical definition. The equivalence test is provenance/equivalence_test.py and its results are provenance/equivalence_results.txt: on twelve synthetic test signals the worst maximum absolute deviation from the historical implementation was 8.771e-15 and the NaN positions were identical. The scope of that result is exactly what it says: synthetic signals, and the clipped starter conversion only. It is not evidence about the Phase 5 bandext conversion, which the reimplementation does not implement. The result establishes numerical equivalence only for the tested synthetic signals and the historical clipped conversion. It does not establish equivalence on real acoustic recordings, nor does it validate the later Phase 5 bandext conversion.

## Item 3. The copied starter function is not in this repository

DATA_SOURCES.md documents that the historical pool building script contained a verbatim copy of the competition starter notebook conversion function, whose source licence is not established. That script, code/build_passage_pool.py, was excluded from this release by owner decision for that reason, and the copied function appears nowhere in this repository. The only conversion implementation published here is the clean reimplementation described in Item 2. The equivalence test loads the historical function from the private research record by source extraction, which is why the test cannot be rerun from the contents of this repository alone. That limitation is recorded in RELEASE_NOTES.md. Consequently, a reader can inspect the clean implementation and the recorded equivalence results, but cannot independently rerun the historical versus clean comparison using this repository alone.

## Item 4. The reproduction roadmap reference

The README reproduction roadmap refers to code/build_passage_pool.py, which is absent from this release for the reason given in Item 3. The roadmap is a partial reconstruction, as the README itself states in its later sections and as RELEASE_NOTES.md states in its contents section. No reader should expect the pool building step to run from this repository.

## Item 5. LICENSE_DECISION.md closing line

LICENSE_DECISION.md ends by recording that no licence was authorised yet. That was the position when the document was written. The owner subsequently selected the BSD 3 Clause licence for his own code and documentation on 18 November 2026, and the LICENSE file in this repository carries its full text. The governing licence position is stated in RELEASE_NOTES.md, section 5.

## Item 6. Historical planning and status documents

Several documents in this repository are planning or progress records of their dates and retain the status language of those dates. They include docs/SCOPE.md, which carries an earlier development gate figure and describes an investigation as running, the Stage 2 process reports in provenance/, some of which describe work as awaiting owner review, and the phase reports in docs/. The underlying research and release preparation stages have since been completed, reviewed and formally closed. The historical documents themselves have not been rewritten to reflect those later decisions. The consolidated research record is docs/WRITEUP_consolidated.md, and the closing status of the programme is stated in RELEASE_NOTES.md, section 10.

## Item 7. Initial commit message

The message on the initial commit of this repository, 0dd1f9be9f12c840fa8be9c77fad9974bb1c891e, does not describe this project. It is unrelated text carried by the publishing tool used for the upload. It has been retained rather than rewritten, because the commit is the published record, and its message is preserved as historical evidence of how the publication was made, not as a description of the repository contents. The provenance of this publication rests on RELEASE_NOTES.md, on the checksum verification recorded at publication, and on this errata, not on that message.

## Item 8. What this errata does not change

This errata corrects documentation status only. The release remains what its own documents describe: a documented research record with partial reproducibility. It does not satisfy the research write up qualification requirements identified in the project feasibility assessment. That is the project evidence based assessment, not an official Zindi eligibility ruling or judging decision. The release contains no demonstration of model performance on real South African acoustic data, no pressure maps, and no end to end runnable reproduction of the programme, and nothing in this errata should be read as claiming otherwise. The development gate of 0.96 was not met on any holdout. The validation scores throughout are synthetic. Real South African acoustic recordings were used as backgrounds within the synthetic construction and for the diagnostic readings on real windows, and those readings are unvalidated diagnostics, not performance evidence. Synthetic validation does not establish accuracy on natural vessel activity.

## Item 9. Competition code sharing rule

The general competition rules state that code shared outside a team must be made available to all challenge participants through the platform discussion boards. Whether an independent public research publication of this kind falls within that rule is not clear on the published wording, and this errata does not resolve the question. It is recorded here so that readers are aware the question exists and remains open.
