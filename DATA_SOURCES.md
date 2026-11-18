# Data Sources — provenance and redistribution conditions

**Status of this document:** proposed release documentation, prepared as
Stage 1 (18 November 2026). It records where every external dataset used
by the programme came from and the conditions under which it may and may
not be redistributed. No dataset is bundled with the proposed release;
this document is the acquisition and attribution record.

Two external datasets were used: VTUAD and the South African
competition acoustics. The competition's starter notebook and other
supplied files are treated as organiser-provided competition materials,
not as additional datasets. No other external dataset contributed to
model fitting, simulation or evaluation. A separate audit examined one
competing participant's data-expansion claim but did not incorporate
that material into the programme.

---

## 1. VTUAD — the vessel-passage corpus

**Exact release used:** VTUAD (rebuilt) **v1.3.2**.

**Canonical source:** Zenodo, DOI **10.5281/zenodo.23037895**.

**Licence:** CC BY 4.0.

**Upstream source:** the recordings originate from **Ocean Networks
Canada (Oceans 3.0)**; VTUAD is the rebuilt, AIS-qualified passage corpus
derived from that source, and it is the passage source the competition
organisers themselves name in the starter notebook.

**Dataset authors** (as listed in the v1.3.2 Zenodo record at DOI
10.5281/zenodo.23037895): Hao Yuan, Wenbo Wang, Tian Li, Guici Chen,
Lingjiang Zeng and Sixian Liu (School of Mathematics and Systems
Science, Wuhan University of Science and Technology; CSSC-LINCOM
Electronics (Wuhan) Co., Ltd.).

**CC BY 4.0 attribution requirements.** Any use or redistribution of
VTUAD material must: give appropriate credit to the dataset authors and
the upstream source (Ocean Networks Canada); provide a link to the
CC BY 4.0 licence; provide the DOI above; and indicate if changes were
made. This programme's use involved substantial transformation:
recordings were converted to one-minute × 25-band relative-dB summaries
(the conversion is described in the phase reports), and only those
derived summaries — never the audio — fed the simulator. The proposed
release does not redistribute VTUAD audio; a reproducer fetches the
archives from the Zenodo record under the same licence.

**Version note — v1.3.0 draft card vs the v1.3.2 release.** The dataset
card bundled inside the local download (`dataset_card.md`) labels itself
**v1.3.0** and describes that version as a candidate/draft. The programme
nevertheless used, and this document cites, the **v1.3.2 Zenodo release**
at DOI 10.5281/zenodo.23037895: that is the release the research record
(`ARCHIVE_MANIFEST.md` §§4–6, the phase reports and the write-up) cites
throughout, and the strict vessel archive's SHA-256 pinned in the
conversion code is the checksum of the archive actually used. A
reproducer should fetch the v1.3.2 record from the DOI and should not be
misled by the bundled card's draft labelling.

**Three-part archive reassembly and checksum verification.** The strict
vessel archive (509 clips) is distributed as three byte-split
parts, `vtuad_strict_wav_vessel.zip.part01`, `.part02` and `.part03`,
which must be concatenated in order to restore the single ZIP
(3,712,337,699 bytes). The programme's conversion code
(`code/build_passage_pool.py`, function `strict_vessel_zip()`) performs
this automatically on first run and verifies the result before use:

1. Each of the first two parts must be exactly 1,237,445,900 bytes, and
   the third exactly one byte less; if any part is the wrong size, the
   archive is not assembled.
2. The parts are concatenated in order to
   `vtuad_strict_wav_vessel.zip`.
3. The SHA-256 of the reassembled archive is computed and must equal the
   value pinned in the code,
   `96564721a917d8b4dd161fdb90c66db917136786e6f4680028be6c6e03cf8ac6`;
   the archive is used only on a match.
4. Per-clip integrity can additionally be verified after extraction
   against `strict_interval_wav_SHA256SUMS.txt`, which lists the SHA-256
   of every WAV inside the archives.

The strict background archive (16 clips) ships as a single ZIP and needs
no reassembly.

**Clip inventory, reconciled.** The strict vessel archive contains
**509 vessel clips** and the strict background archive **16 ambient
clips**: the per-clip checksum ledger
(`strict_interval_wav_SHA256SUMS.txt`) lists exactly 509 vessel and 16
background entries, and every clip in both archives was converted and
retained. The programme's passage pool is larger than the strict
archives alone, because the same VTUAD v1.3.2 download also includes
four per-class archives — `vtuad_wav_pilotvessel.zip`,
`vtuad_wav_rescue.zip`, `vtuad_wav_tanker.zip` and
`vtuad_wav_sailing.zip` — which contributed a further **17 vessel
clips** (2, 4, 5 and 6 respectively). The harvest
(`code/build_passage_pool.py`) converted all of them, so the pool that
fed the simulator is **526 vessel clips + 16 ambient clips, 542 in
total** (recorded in `data/passages.npz`, `RESULTS_sim_v1.md` and
`ARCHIVE_MANIFEST.md`). The two figures are therefore consistent:
526 = 509 strict-archive clips + 17 per-class clips, all from the same
v1.3.2 release.

**Step 3 inventory note.** Step 3 (consolidated report, §11) worked
from a separate inventory of the same pool: the 525 clips carrying
usable hull and date metadata, which are exactly the strict-archive
clips (509 vessel + 16 ambient). Its split records
(`step3/splits.json`, `step3/clip_sides.csv`) cover only those 525:
212 family side, 211 I1 side, and 102 excluded because their hull and
date assignments disagreed. The 17 clips Step 3 excluded for missing
hull and date metadata are the same 17 per-class clips counted above
(sailing 6, tanker 5, rescue 4, pilotvessel 2). This is established by
the records, not inferred from the counts: the Step 3 results record
names those four classes as its metadata exclusions, its split file
records the count under excluded class-archive clips with no metadata,
the pool's name list shows the 17 per-class entries are the only pool
clips outside the strict archives, and the Step 3 clip list contains
strict-archive keys only. The two inventories are consistent views of
the same 542-clip pool, not separate harvests.

---

## 2. South African acoustic data — NOT redistributable

**Source:** the competition package for the Underwater Vessel Noise
Challenge on Zindi: 1,715,779 unlabelled one-minute band-energy records
across ten South African sites, supplied as the station review package
(one Parquet file per station, with station and recorder metadata).

**Redistribution condition — read carefully.** The package carries an
explicit notice, recorded verbatim in substance here: *"Access is for
challenge organiser review. This packaging grants no new licence or
permission to redistribute the South African source data."*

Consequences for this release and for any reproducer:

- The South African acoustic data is **not included** in the proposed
  release, in any form.
- **Neither the original acoustic data nor derived pools reproducing it
  may be redistributed.** The programme's screened background pool
  (87,653 windows) and every simulator corpus and feature file built
  from these backgrounds are therefore also excluded from the release
  (see the README, §7). Describing the screening and construction
  procedures in code and prose is permitted; shipping the data or its
  derivatives is not.
- Competition participants must obtain the data through the authorised
  Zindi competition access route, under the competition's own terms.
  Full reproduction of this programme is possible only with that access.
- The competition's `Test.csv`, sample submission and starter notebook
  are likewise organiser material obtained through the same route and
  are not redistributed here.

---

## 3. Competition starter notebook — provenance of `minute_band_power`

The competition starter notebook is organiser-supplied material with **no
licence stated** in the competition package.

The existing implementation of the reference conversion,
`minute_band_power` in `code/build_passage_pool.py`, is — as its own
docstring states — a **verbatim copy of the starter notebook's cell-11
reference conversion** (non-overlapping 2 s segments, mean removed,
one-sided Hann periodogram with density scaling, power integrated over
each deci-decade band, 30 segments averaged in linear units per complete
minute). It is attributed in the docstring, but attribution alone does
not establish permission to redistribute copied code, and no licence for
the notebook has been identified.

**Agreed resolution (not yet carried out):** the proposed public release
will use a **clean reimplementation** of this conversion, written from
the published mathematical definition rather than copied from the
notebook, so that the release does not depend on an unclear licence for
organiser-supplied code. An **equivalence check** of the reimplementation
against the historical implementation is a **required Stage 2 task**.

**Neither the reimplementation nor the equivalence check has happened
yet.** Both are Stage 2 work, gated on a separate owner authorisation.
Until then, this provenance note is the accurate statement of the
position: the historical code in the frozen record contains the copied
function, attributed as described above, and the frozen archive is not
modified to change that — the fix belongs to the curated release copy,
not to the evidence vault.

---

## 4. Summary table

| Source | Version / route | Licence / terms | In the release? |
|---|---|---|---|
| VTUAD (rebuilt) | v1.3.2, Zenodo DOI 10.5281/zenodo.23037895 | CC BY 4.0, attribution required | No — fetch from Zenodo |
| South African acoustics | Zindi competition package | No redistribution permitted | No — competition access only |
| Competition files (Test.csv, sample submission, starter notebook) | Zindi competition package | Organiser material, no licence stated | No — competition access only |
