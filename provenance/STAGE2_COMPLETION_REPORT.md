# Stage 2 Completion Report — Underwater Vessel Noise, Zindi Research Programme

**Date:** 18 November 2026 (Africa/Lagos)
**Status:** Stage 2 work delivered for owner review. This report does
not declare Stage 2 formally complete; that decision is the owner's.
**Location of deliverables:** `stage2_work/` (plus one private file in
`step3/`, noted in §8).

---

## 1. Purpose and authorised scope

Stage 2 is provenance and release hygiene preparation for the
proposed public release of the vessel-noise research programme,
following the formally signed-off Stage 1 documentation set. It was
authorised for four areas only: (A) starter-notebook conversion
provenance, (B) data exclusion and redistribution hygiene,
(C) identifier hygiene, and (D) path and environment hygiene. A
completion round on the same date addressed the two items the owner's
review left open: the Area C clip-key linkage issue and an Area D
static path-resolution check of the sanitised drivers.

Stage 2 is not release assembly, not publication, and not research:
no modelling, no experiments, no test-set inference, no submission
activity, no licence selection or application was authorised or
performed.

## 2. The four areas and their deliverables

**Area A — Conversion provenance.** The historical
`minute_band_power` in `code/build_passage_pool.py` (a self-declared
verbatim copy of the organisers' starter-notebook conversion, no
licence established) was documented precisely, reimplemented
independently from the published mathematical definition in
`clean_conversion.py`, and compared against the historical function
in a controlled equivalence test (`equivalence_test.py`,
`equivalence_results.txt`). Worst maximum absolute deviation across
12 synthetic signals: 8.771e-15; NaN positions identical everywhere.
The historical function is the *clipped* starter conversion (band 0
from about 28.2 Hz), not the later bandext variant; the clean
implementation reproduces that function. No replacement has been
made; replacing the historical function in the research script
requires separate review and approval. Report:
`STAGE2_PROVENANCE_REPORT.md`.

**Area B — Data exclusion inventory.** A full inventory of material
that must not enter a public release was built by inspecting the
tree in place; nothing was downloaded, extracted, regenerated or
copied. Headline entries: the South African acoustics source
(~91 MB archive, ~92 MB extracted, no redistribution permission) and
its derived pools and feature corpora (289 `.npz` files, ~5.4 GB);
`Test.csv`, the sample submission and the starter notebook; the four
historical submission CSVs; the VTUAD archives (~5.1 GB of ZIPs) and
the extracted waveform cache (~15 GB) — VTUAD is CC BY 4.0 but
remains an external prerequisite, never bundled; the Step 3 corpus;
91 trained boosters (~2.0 GB). No prohibited material was copied
into `stage2_work/`; the only data-derived files there are the
clip-assignment derivatives of Area C, which the inventory and
reported scans found to contain no restricted data (an inventory
and scan finding, not a complete binary-content guarantee). Report: `STAGE2_DATA_HYGIENE_REPORT.md`. This is an
inventory assessment of the working tree, **not** an audit of a
future curated release package; no such package exists yet.

**Area C — Identifier hygiene.** The `MMSI` column of
`step3/clip_sides.csv` (178 distinct real identifiers over the 509
vessel rows, plus a placeholder on the 16 ambient rows) was removed
in a first derivative, `clip_sides_sanitised.csv` (525/525 rows,
retained fields row-identical, zero identifier strings, zero
nine-digit patterns). The completion round then resolved the linkage
issue the owner's review identified: the retained `clip_key` values
join directly to vessel identity through the upstream VTUAD metadata
(see §5), so that first derivative is **MMSI-stripped only — not
anonymised**. The treatment delivered is a further release-facing
derivative pair using stable opaque identifiers
(`clip_sides_release.csv` and `splits_release.json`, same mapping,
cross-file consistent), with the key-to-ID mapping held only in the
private research tree. **The opaque-ID derivatives preserve the
assignment record, not the independently reproducible
vessel-identity evidence**: a reader of the public files cannot
reconstruct the hull-disjointness check from them; that evidence
remains in the preserved original records, and the scientific
report must not imply otherwise. `clip_sides_sanitised.csv` is
classified as an **intermediate private working artefact, not a
release candidate**, because it still contains all original clip
keys; it must not be used in any future curated package. Full findings, join evidence, verification
numbers and the terminology correction are in the dated addendum to
`STAGE2_IDENTIFIER_AUDIT.md`.

**Area D — Path and environment hygiene.** No `/home/` or `/root/`
paths exist in any Python or shell source. Seven log files (23
lines) carried the original machine's absolute project prefix;
sanitised, captioned working copies were prepared for the two shell
drivers (diff-verified against the originals; final form described
below) and for the retained Step 3 incident log; the six excluded
`code/` logs are listed as proposed sanitisations only. An
independent credential scan found no credentials, tokens or private
URLs (one schematic, empty placeholder parameter in the
already-excluded `PLAN.md` was noted; it is not a credential). The
completion round added a static path-resolution trace of both
sanitised drivers, recorded as a dated addendum to
`STAGE2_PATH_HYGIENE_REPORT.md`: every relative path assumes the
working directory `<PROJECT_ROOT>/code`, and resolution is unchanged
provided `PROJECT_ROOT` names the project root. In their final
delivered state (§11), both sanitised drivers check that
`PROJECT_ROOT` is set, check that `${PROJECT_ROOT}/code` exists,
terminate with an error if either precondition fails, and use a
guarded `cd`; the pipeline driver's process-name pattern matches
`run_screens_sanitised.sh`. These changes apply only to the
sanitised working copies; the original drivers remain untouched.
The drivers were statically validated (`bash -n` and line-level
diffs against the originals); neither has been executed, and
successful end-to-end operation has not been established.

## 3. What was changed in working copies; what originals were preserved

Changed — working copies only, all inside `stage2_work/`:

- `clean_conversion.py` — new independent implementation (no original
  touched).
- `clip_sides_sanitised.csv` — column-drop derivative of
  `step3/clip_sides.csv` (MMSI removed).
- `clip_sides_release.csv`, `splits_release.json` — opaque-ID
  derivatives (completion round).
- `run_screens_sanitised.sh`, `run_full_pipeline_sanitised.sh` —
  sanitised driver copies: a generic caption block quoting no
  literal original path, the machine-specific directory change
  replaced by a project-root-relative one, an explicit
  `PROJECT_ROOT` preflight with a guarded `cd`, and, in the pipeline
  driver, the process-name pattern corrected to the sanitised
  screens driver's file name.
- `stage1_log_sanitised.log` — caption header prepended; the machine
  path prefix replaced by `<PROJECT_ROOT>` in 4 lines.
- The four Stage 2 reports — written new in this stage; the
  completion round appended dated addenda to the identifier and path
  reports without altering their original text.

Preserved unchanged: the frozen `archive/` in its entirety; the four
approved Stage 1 documents in `release_docs_stage1/`; every
historical research script, record and log, including
`code/build_passage_pool.py`, `step3/clip_sides.csv` and
`step3/splits.json`. One new file was created in the private research
tree by design: `step3/opaque_id_mapping_PRIVATE.csv` (the opaque-ID
mapping; see §8). No original file was overwritten anywhere.

## 4. Data and files explicitly excluded from any future release

Per the Area B inventory and the owner's Stage 1 release decisions:
the South African acoustics source data and every derived pool or
feature corpus that could reproduce it; `Test.csv`, the sample
submission and all other organiser competition files; the four
historical submission CSVs as release data; the VTUAD archives and
extracted audio caches (external prerequisite instead); all generated
`.npz` corpora and corpus chunks; all trained booster files; the
original `step3/clip_sides.csv` and `step3/splits.json` (private
evidence, superseded for release purposes by the opaque derivatives);
the opaque-ID mapping file; `PLAN.md` (account and scheduling
detail); and all run logs except the sanitised, captioned Step 3
incident log. This list records exclusions; assembling a release
package against it is Stage 3 work and is not authorised.

## 5. Remaining limitations

- **Equivalence (Area A)** was tested on synthetic signals only, over
  the tested signal classes and sample rates. It establishes
  agreement over those inputs, not exhaustive equivalence over every
  possible input, and it concerns the clipped starter conversion, not
  the bandext variant used by the programme's later phases.
- **Linkage (Area C).** `clip_key` joins to VTUAD metadata `path`
  (the key with a leading `./`), and the metadata carries MMSI:
  verified 525 of 525 keys join, 509 reaching a real MMSI, with the
  metadata and `clip_sides.csv` MMSI values in agreement on 525 of
  525 rows. The MMSI-stripped derivative therefore remains
  re-identifiable by anyone holding the public VTUAD metadata. The
  opaque derivatives remove both the identifier and the joinable
  reference; their residual risk is limited to the retained
  non-identifying fields (label, date, duration, side) as
  quasi-identifiers, recorded in the identifier addendum and not
  claimed to be zero.
- **Reproducibility trade-off (Area C).** The opaque derivatives can
  no longer be joined to VTUAD metadata, and neither derivative
  independently proves hull-disjointness; that evidence remains the
  preserved originals (`clip_sides.csv`, `splits.json`,
  `stage0_verification.json`) plus the private mapping, in the
  research tree.
- **Portability (Area D).** The sanitised drivers require
  `PROJECT_ROOT` to identify the project root. A missing or invalid
  `PROJECT_ROOT` now produces an immediate nonzero exit from the
  preflight checks added in §11, and all relative paths resolve from
  `<PROJECT_ROOT>/code`. Syntax was checked with `bash -n` and the
  changes verified by line-level diff against the originals. The
  pipeline has not been executed: successful end-to-end operation
  has not been established, and no claim of full portability or
  successful reproduction is made.
- **Inventory limit (Area B).** The exclusions are an assessment of
  the current working tree. A curated release package, once one
  exists under separate authorisation, will need its own audit
  against this inventory.
- **Authorship items.** The owner's Zindi username in the write-ups
  and proposal, and his first name in ten code docstrings, remain as
  recorded; both are owner authorship decisions, not leaks, and were
  not sanitised in this stage.

## 6. Status of every area (owner's review language)

- **Area A:** approved in substance.
- **Area B:** approved in substance, as an inventory assessment
  only — not a claim that a future curated package has been audited.
- **Area C:** approved in substance, with the clip-key linkage
  correction now delivered for owner decision (opaque-ID derivatives
  prepared; the MMSI-stripped file's description corrected).
- **Area D:** approved in substance; the final static
  path-resolution review and the requested corrections (generic
  captions, `PROJECT_ROOT` preflight with guarded `cd`, process-name
  correction) have been delivered in the sanitised working copies.

## 7. What Stage 2 does not authorise

Stage 2 does not constitute permission to assemble or publish a
release. Stage 3 (curated release assembly), repository creation,
licence selection or application, any replacement of the historical
conversion in the research scripts, and any publication or upload
each require separate, explicit owner authorisation. None of that
work has begun.

## 8. All Stage 2 deliverables and locations

In `stage2_work/`:

1. `STAGE2_PROVENANCE_REPORT.md` (Area A report)
2. `STAGE2_DATA_HYGIENE_REPORT.md` (Area B report)
3. `STAGE2_IDENTIFIER_AUDIT.md` (Area C report, with the
   18 November 2026 completion addendum)
4. `STAGE2_PATH_HYGIENE_REPORT.md` (Area D report, with the
   18 November 2026 completion addendum)
5. `STAGE2_COMPLETION_REPORT.md` (this report)
6. `clean_conversion.py`
7. `equivalence_test.py`
8. `equivalence_results.txt`
9. `clip_sides_sanitised.csv` (MMSI-stripped derivative; superseded
   for release purposes by item 10)
10. `clip_sides_release.csv` (opaque-ID release-facing derivative)
11. `splits_release.json` (opaque-ID derivative of `splits.json`)
12. `run_screens_sanitised.sh`
13. `run_full_pipeline_sanitised.sh`
14. `stage1_log_sanitised.log`

In the private research tree (not public-facing, listed for
completeness only; its contents are not reproduced in any report):

15. `step3/opaque_id_mapping_PRIVATE.csv` — the clip-key to opaque-ID
    mapping, held exclusively in the private tree by design.

## 9. Agent-reported checks versus independently reproduced checks

The equivalence test, the identifier scans and verifications, the
path and credential scans, and the completion round's consistency
checks (§10) are **agent-reported**: they were performed and recorded
by the agents doing this stage's work, with methods and scopes stated
in the respective reports. One check was **independently
reproduced** outside that work: the frozen archive's checksum
verification of 18 November 2026 (242 of 242 files matching the
manifest SHA-256 list) was independently re-run by Muse, the lead
agent in the main session, on 18 November 2026, after the Stage 2
working agent's own run of the same check. It was not performed by
the owner. (Attribution stated explicitly in the final corrections,
§11.)
In the completion round, archive checksums were deliberately **not**
recalculated (owner's instruction); preservation in this round was
verified by modification-time scan instead (§10a).

## 10. Final consistency checks (completion round, bounded and static)

- **(a) Originals unmodified in this round.** Method: modification-
  time scan of the whole project tree (excluding `.venv/`) for files
  changed after the completion round's session start (14:51 UTC,
  18 November 2026). Result: the only file outside `stage2_work/`
  changed is `step3/opaque_id_mapping_PRIVATE.csv`, the authorised
  new private mapping file. No original research file, no Stage 1
  document and no archive file was modified. (Modification times can
  in principle be altered by tools that preserve timestamps; no such
  tool was used, and the scan is reported as an mtime check, not a
  content proof — content-level archive verification remains the
  earlier checksum result in §9.)
- **(b) No restricted content in the public-facing derivatives.** A
  scan of the text-readable files in `stage2_work/` for all 178 MMSI
  digit strings found zero occurrences. Binary artefacts were not
  content-scanned for identifiers and are excluded from any proposed
  release. Original clip-key strings occur in exactly one file,
  `clip_sides_sanitised.csv`; all 525 occur there by design. The
  opaque-ID derivatives contain none. The private mapping is absent
  from `stage2_work/` and present at
  `step3/opaque_id_mapping_PRIVATE.csv`, as verified. No South
  African competition audio, derived acoustic datasets, or
  competition Test data were identified in `stage2_work/`. The
  data-bearing files in that directory are assignment-metadata
  derivatives only (the three clip-assignment CSVs and the splits
  derivative). This finding is based on the reported file inventory
  and scans; it is not a binary-content audit of every artefact.
  Credential-pattern hits occur only in `STAGE2_PATH_HYGIENE_REPORT.md`,
  where they are the report's own enumeration of the patterns it
  searched for — no credential value exists in any Stage 2 file.
- **(c) Machine-specific paths in the sanitised scripts and log.**
  Final result after the §11 corrections, within the scope of the
  three sanitised files (the two drivers and the retained incident
  log): zero literal original machine paths remain in any of them.
  The executable directory changes use `${PROJECT_ROOT}`; the
  retained log uses `<PROJECT_ROOT>` placeholders. The historical
  paths remain only in the preserved original research files. This
  result covers the three sanitised files only and is not
  generalised to the entire research tree. (As first delivered, two
  caption occurrences remained; they were removed in §11, which
  preserves the record of that earlier state.)
- **(d) Deliverables exist.** Every file named in §8 was confirmed
  present on disk at the stated location in this round.
- **(e) Derivatives distinguishable from originals.** The sanitised
  drivers and log carry caption headers naming their originals and
  their changes, and all derivatives carry distinct file names
  (`_sanitised`, `_release`) in a separate directory. The CSV/JSON
  derivatives carry no embedded caption (a plain format limitation);
  they are distinguishable by file name, location, and — for the
  release CSV — by the `clip_id` header replacing `clip_key`.
- **Checks not performed.** Archive checksums were not recalculated
  in this round (owner's instruction; see §9). Binary artefacts were
  not content-scanned for identifiers, as in the earlier rounds;
  they are excluded release material regardless. No curated release
  package exists, so no package audit was performed or is claimed.

## 11. Final corrections (18 November 2026, owner's Stage 2 completion review)

Applied to the Stage 2 working copies and to this report only. No
original research file, Stage 1 document or archive file was changed,
and Areas A to C were not reopened.

1. **Driver captions no longer disclose the original machine path.**
   Both sanitised drivers quoted the original `cd` line, including
   the local username and project directory, inside their caption
   comments. The captions now describe the change generically (an
   original machine-specific project-directory command replaced with
   a project-root-relative command). Verified: no literal machine
   path (no `/home/`, no `~/workspace`, no username) remains in
   `run_screens_sanitised.sh`, `run_full_pipeline_sanitised.sh` or
   `stage1_log_sanitised.log`. The historical path remains in the
   private research record, in the original drivers.
2. **Both sanitised drivers now fail safely when `PROJECT_ROOT` is
   missing or invalid.** An explicit preflight was added ahead of the
   directory change: exit 1 with an error if `PROJECT_ROOT` is unset,
   exit 1 if `${PROJECT_ROOT}/code` does not exist, and
   `cd "${PROJECT_ROOT}/code" || exit 1`. This corrects the behaviour
   recorded in §5, which described the copies as first delivered
   (continuing in the wrong directory; the screens driver could wait
   indefinitely in its poll loop). Both files pass `bash -n`. A
   line-level diff against the originals confirms the only
   executable changes are the preflight block, the guarded `cd`, and
   item 3 below; logic, parameters and seeds are untouched.
3. **Pipeline driver process-name coupling corrected.** The wait
   loop in `run_full_pipeline_sanitised.sh` now matches
   `run_screens_sanitised.sh`, the sanitised screens driver's file
   name, so the sanitised pair is self-consistent. The original
   drivers are unchanged.
4. **Checksum-verification attribution clarified (§9).** The
   independent 242 of 242 archive checksum re-verification of
   18 November 2026 was performed by Muse, the lead agent in the
   main session, separately from the Stage 2 working agent's own
   run; §9 now states this explicitly. It was not performed by the
   owner.
5. **Area C documentation distinctions made prominent (§2).** The
   opaque-ID derivatives preserve the assignment record, not the
   independently reproducible vessel-identity evidence; and
   `clip_sides_sanitised.csv` is classified as an intermediate
   private working artefact, not a release candidate, and must not
   be used in any future curated package.

With these corrections, Area D's minor outstanding items are
addressed in the working copies. Formal Stage 2 sign-off remains
the owner's decision; nothing in this section authorises Stage 3,
release assembly or publication.
