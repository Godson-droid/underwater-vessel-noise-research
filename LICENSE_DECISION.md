# Licence Decision Note — the project's original code

**Status:** decision note prepared as Stage 1 release documentation
(18 November 2026). **No licence has been chosen, and no licence grant
is made by this document.** Until the owner chooses one, the project's
code remains all rights reserved by default, whatever the research
reports say about reproducibility.

**Scope.** This decision concerns only the project's **original code**
(the programme scripts under `code/` and `step3/`). It does not and
cannot cover:

- the `minute_band_power` function in `code/build_passage_pool.py`,
  which is a verbatim copy of the competition starter notebook's
  reference conversion (organiser material, no licence stated). The
  agreed fix is a clean reimplementation from the published mathematical
  definition, as a Stage 2 task; see `DATA_SOURCES.md` §3.
- any data. VTUAD is CC BY 4.0 under its own terms; the South African
  acoustic data may not be redistributed at all. A code licence grants
  no rights in either.

This recommendation assumes the owner has the necessary rights to
license all original code included in the release and that no
third-party contributions carry separate licensing obligations.

## The three candidate licences compared

| | MIT | BSD-3-Clause | Apache-2.0 |
|---|---|---|---|
| **Type** | Permissive | Permissive | Permissive |
| **Attribution** | Copyright and licence notice must be kept in copies | Copyright notice, licence text and disclaimer retained; the third clause additionally bars using contributors' names to endorse derived products without permission | Copyright and licence notice kept; a NOTICE file, if present, must be preserved; modified files must carry prominent notices stating that they were changed |
| **Redistribution** | Use, copy, modify, merge, publish, distribute, sublicense and sell all permitted | Redistribution in source and binary form permitted under the notice conditions | Redistribution permitted under the licence conditions, including the NOTICE and change-marking requirements |
| **Patent provisions** | None — silent on patents | None — silent on patents | Express patent grant from every contributor, with a retaliation clause: a licensee who sues over the work loses the patent licence |
| **Warranty** | Disclaimed ("as is") | Disclaimed | Disclaimed |
| **Practical character** | Shortest and simplest; the most widely recognised permissive licence; maximal reuse with minimal friction | Near-identical practical effect to MIT, plus the non-endorsement clause, which suits a named researcher's work | Longer and more formal; the patent grant and retaliation clause give downstream users the clearest legal position; the NOTICE mechanism suits a release that carries attribution records |

## Practical implications for this release

The code is a research record whose value is inspection, verification
and reuse of method. All three licences serve that purpose; none of them
imposes copyleft conditions, and none affects the data position, which
is governed separately (see `DATA_SOURCES.md`). The real differences are
narrow:

- **MIT** minimises friction and reader effort. Its silence on patents
  is unlikely to matter for acoustic-signal research code, but it is a
  silence.
- **BSD-3-Clause** adds the non-endorsement clause: nobody may use the
  owner's or contributors' names to promote a derived product without
  permission. For a solo researcher's named work, that is a small, real
  protection at no practical cost.
- **Apache-2.0** provides an express contributor patent licence and a
  defined patent-termination mechanism, offering additional clarity on
  patent-related rights and obligations. It does so at the cost of a
  longer licence text and slightly more formality than a small research
  release usually carries.

## Recommendation

**BSD-3-Clause** is recommended: it keeps the release as frictionless as
MIT for anyone verifying or building on the method, and its
non-endorsement clause is a proportionate protection for a named solo
research record. Apache-2.0 is an equally defensible choice if the owner
prefers the express patent grant. MIT is appropriate if the owner
prioritises the shortest, simplest licence and does not consider the
additional non-endorsement provision necessary.

This recommendation is advisory only. The choice belongs to the owner,
and no licence file will be created until the owner makes it.

OWNER DECISION REQUIRED — NO LICENCE AUTHORISED YET
