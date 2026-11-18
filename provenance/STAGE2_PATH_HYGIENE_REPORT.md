# Stage 2 Path and Environment Hygiene Report

**Date:** 18 November 2026 (Africa/Lagos)
**Scope:** Stage 2 Area D.
**Originals preserved unchanged:** `code/run_screens.sh`,
`code/run_full_pipeline.sh`, `step3/stage1.log`, and all run logs.
**Stage 2 working copies:** `stage2_work/run_screens_sanitised.sh`,
`stage2_work/run_full_pipeline_sanitised.sh`,
`stage2_work/stage1_log_sanitised.log`.

---

## 1. Machine-specific absolute paths — findings

**Method.** Pattern search for `/home/<name>` and `/root/` over:
`code/*.py`, `code/*.sh`, `step3/*.py` (source scope), and every
`.log` file in the working tree excluding `archive/` and `.venv/`
(98 log files: 90 under `code/`, 4 under `step3/`, 4 under
`data/vtuad/`). A second search covered tilde-form home paths (`~/workspace/...`)
in the shell drivers.

**Source files (`*.py`, `*.sh` in `code/`, `step3/*.py`): zero
`/home/` or `/root/` paths.** Programme code resolves its data paths
relative to the tree (`code/common.py` derives `DATA` from
`__file__`), so the code itself is portable.

**Shell drivers:** each of `code/run_screens.sh` and
`code/run_full_pipeline.sh` contains one machine-specific line,
`cd ~/workspace/zindi-vessel/code` (tilde form: the original machine's
home directory plus the project path). Sanitised copies were prepared;
see §2.

**Logs:** exactly **7 files, 23 lines**, contain an absolute home
path, and every occurrence is the single prefix
`/home/hatch/workspace/zindi-vessel` (the original machine's local
username and project directory), appearing in Python traceback frames
and run headers:

| File | Lines with the path | Disposition |
|---|---|---|
| `code/bg_pool.log` | 2 | Exclude from release (audit); no copy prepared |
| `code/train.log` | 1 | Exclude; no copy prepared |
| `code/train_v2.log` | 3 | Exclude; no copy prepared |
| `code/train_v2_eval.log` | 1 | Exclude; no copy prepared |
| `code/sim_v3_gen.log` | 3 | Exclude; no copy prepared |
| `code/p6_ws1_screens.log` | 9 | Exclude; no copy prepared |
| `step3/stage1.log` | 4 | **Retain only as sanitised, captioned copy — prepared, see §2** |

The remaining 91 log files contain no `/home/` or `/root/` path. No
`/tmp`, drive-specific or other VM-specific paths were found in the
scripts or drivers, and the `data/vtuad/download*.log` files contain
no absolute home paths. Classification: the log paths are a genuine
but low-grade disclosure (a local username and directory structure,
nothing more); the frozen duplicates of the six `code/` logs inside
`archive/` are part of the evidence vault and stay exactly as they
are — exclusion and sanitisation apply only to a future release copy.

## 2. Sanitisations prepared (working copies)

### 2.1 `stage2_work/run_screens_sanitised.sh` and `stage2_work/run_full_pipeline_sanitised.sh`

- **Original paths:** `code/run_screens.sh`, `code/run_full_pipeline.sh`.
- **Exact change (one line in each):**
  `cd ~/workspace/zindi-vessel/code` → `cd "${PROJECT_ROOT}/code"`,
  plus a caption comment block after the shebang recording the
  original line, the replacement, and that `PROJECT_ROOT` must be set
  to the project root before running.
- **Rationale:** removes the machine-specific home path so the drivers
  can run on a reproducer's machine; path-prefix substitution only.
- **Verification:** `diff` against each original shows exactly two
  differences — the added caption block and the single `cd` line. No
  execution logic, parameters, seeds, tags, ordering or output paths
  were changed (the drivers' remaining paths are relative:
  `../.venv/bin/python`, `../data/...`, `../models/...`). Both copies
  pass `bash -n` syntax checking.
- **Remaining uncertainty:** with `PROJECT_ROOT` unset the `cd` fails
  loudly rather than silently running in the wrong directory; this is
  documented in the caption and is the intended behaviour of the
  substitution form the owner specified.

### 2.2 `stage2_work/stage1_log_sanitised.log`

- **Original path:** `step3/stage1.log` — the incident log for the
  Step 3 Stage 1 first (buggy) run, which the owner decided is
  retained only as a sanitised, clearly captioned copy.
- **Exact change:** a caption header was prepended stating that this
  is a sanitised copy of the incident log, what it evidences
  (RESULTS_step3_stage01.md §3.3), and what was changed; and the
  prefix `/home/hatch/workspace/zindi-vessel` was replaced by
  `<PROJECT_ROOT>` in all 4 traceback lines.
- **Verification:** the sanitised copy contains zero occurrences of
  `/home/hatch`; its body after the caption is byte-identical to the
  original with the prefix substitution applied (checked
  programmatically). No log content was added, removed or reworded.

### 2.3 Proposed sanitisations not copied

The six `code/*.log` files in §1 were **not** copied: the audit's
release plan excludes all logs except the Step 3 log, so sanitised
copies of excluded files would be dead weight. If a future release
decision ever retains one of them, the required sanitisation is the
same single-prefix substitution (`/home/hatch/workspace/zindi-vessel`
→ `<PROJECT_ROOT>`) on the lines counted in §1.

## 3. Credential and disclosure scan

**Scope:** the §1 source and log scope, plus `step3/*.json` and all
root-level `*.md`. **Patterns:** `api_key`/`apikey`, `password`,
`passwd`, `secret`, `bearer`, `access_token`, `auth_token`, `sk-…`,
`ghp_…`, `AKIA…`, `private_key`, `credentials`, and URLs with embedded
credentials (`https?://user:pass@…`).

**Result: no credentials, tokens, passwords, private URLs or wallet
addresses found.** Every pattern hit was inspected and is benign:

- The `password` / `secret` / `credentials` hits are sentences in
  `RELEASE_AUDIT.md` itself, stating that none were found.
- The single `sk-` hit is the substring inside the word "mask-form" in
  `RESULTS_conversion.md`.
- **One item recorded for completeness:** `PLAN.md` (line 40) contains
  a *schematic* URL pattern for the competition file host,
  `api.zindi.world/.../files/<file>?auth_token=…`, in a historical
  prose note. The parameter value in the file is empty placeholder
  punctuation (two characters), **not a token** — no credential is
  present. `PLAN.md` is in any case recommended for exclusion from any
  release (audit Fix 7) for its account and scheduling detail.
- URLs with embedded credentials: **0** in scope. Email addresses:
  **0** in scope (see the identifier audit §5).

This independently confirms the audit's "no secrets found" result
within the scope actually scanned.

## 4. Remaining uncertainties

- The scan covered text files in the scopes named above. Binary
  artefacts (`.npz`, boosters, `.npy`) were not content-scanned; all
  are excluded from any release on data-hygiene grounds regardless.
- `.venv/` was excluded (third-party environment code, never release
  material), as was `archive/` (immutable vault).
- The VTUAD download helper scripts under `data/vtuad/` (`download*.sh`)
  sit outside the driver scope named for this area; `data/` is
  excluded from release wholesale, so no sanitisation was prepared for
  them. Their logs were included in the path scan and are clean.

---

## Addendum — 18 November 2026, Stage 2 completion: static path-resolution check of the sanitised drivers

This addendum records a static trace of the two sanitised drivers
(`stage2_work/run_screens_sanitised.sh`,
`stage2_work/run_full_pipeline_sanitised.sh`). Nothing was executed;
the earlier `bash -n` syntax validation stands. The original §2.1
finding is confirmed and made precise.

### D.1 Every relative path, and the working directory each assumes

Both drivers execute exactly one `cd`, the substituted line
`cd "${PROJECT_ROOT}/code"` (line 16 in each sanitised copy), before
any relative path is used. Every relative path in both scripts is
therefore resolved against **`<PROJECT_ROOT>/code`**:

`run_screens_sanitised.sh`:

| Path | Where | Assumes cwd |
|---|---|---|
| `../.venv/bin/python` | `PY=` assignment, used for every invocation | `<PROJECT_ROOT>/code` |
| `../models/conv_${tag}_screen.json` | completion test, per tag | `<PROJECT_ROOT>/code` |
| `../data/passages_conv_${tag}.npz` | inside the embedded Python probe | `<PROJECT_ROOT>/code` |
| `simulate_conv.py`, `train_conv.py` | bare file names | `<PROJECT_ROOT>/code` (the scripts live there) |
| `sim_cs_${tag}.log`, `train_cs_${tag}.log` | output redirections | written into `<PROJECT_ROOT>/code` |

`run_full_pipeline_sanitised.sh`:

| Path | Where | Assumes cwd |
|---|---|---|
| `../.venv/bin/python` | `PY=` assignment | `<PROJECT_ROOT>/code` |
| `../models/conv_min60_screen.json` | wait-loop test | `<PROJECT_ROOT>/code` |
| `../data/sim_cf_bandext_meta.json`, `../data/passages_conv_bandextdb.npz`, `../data/sim_cf_bandext_maskfix_meta.json` | artifact tests | `<PROJECT_ROOT>/code` |
| `../models/conv_bandextdb_screen.json` | artifact test | `<PROJECT_ROOT>/code` |
| `simulate_conv.py`, `train_conv.py`, `convert_bandextdb.py`, `simulate_interact.py` | bare file names | `<PROJECT_ROOT>/code` |
| `sim_cf_*.log`, `train_cf_*.log`, `conv_p5_*.log`, `sim_cs_*.log`, `train_cs_*.log` | output redirections | written into `<PROJECT_ROOT>/code` |

In every case `../` denotes the project root itself, i.e. the
directory containing `code/`, `data/`, `models/` and `.venv/` as
siblings.

### D.2 Resolution is unchanged by the substitution — with one condition

The original line was `cd ~/workspace/zindi-vessel/code`, landing in
`<project root>/code` on the original machine. With `PROJECT_ROOT`
set to the project root, `cd "${PROJECT_ROOT}/code"` lands in exactly
the same directory relative to the tree, and every path in D.1
resolves identically. **Condition: `PROJECT_ROOT` must identify the
project root — the parent of `code/` — not the `code/` directory
itself.** If it were set to the `code/` directory, the `cd` target
would be `<code>/code`, which does not exist, with the consequences
in D.3.

### D.3 Required environment variable and behaviour when it is unset

- **Requirement.** `PROJECT_ROOT` must be set (and exported, if the
  drivers are invoked from a parent shell) to the absolute path of
  the project root before either driver is run. This is stated in
  each sanitised copy's caption block.
- **Unset behaviour, traced statically.** With `PROJECT_ROOT` unset
  or empty, the line expands to `cd "/code"`. On any machine without
  a literal `/code` directory the `cd` fails with an error message.
  **Neither script uses `set -e` (or `set -u`), so execution does
  not stop**: the script continues in whatever working directory the
  invoking shell started in, and every relative path in D.1 then
  resolves against that wrong directory. Concretely: in
  `run_screens_sanitised.sh`, `../.venv/bin/python` will normally not
  exist there, the embedded probe yields an empty count, the
  completion test never succeeds, and the script waits indefinitely
  in its poll loop rather than failing. In
  `run_full_pipeline_sanitised.sh`, the artifact tests evaluate
  against the wrong tree, the sequenced steps invoke a
  `../.venv/bin/python` that normally does not exist, each invocation
  fails, and the script runs through to its completion message having
  done no work. If a `/code` directory did exist, the `cd` would
  succeed into the wrong tree with the same class of consequences.
  This corrects the looser wording in §2.1 ("fails loudly"): the
  failure is loud at the `cd` line itself, but neither driver halts
  on it, and the follow-on behaviour is a hang (screens) or a
  no-work run-through (pipeline), not a clean stop. A future user of
  the sanitised drivers should set `PROJECT_ROOT` and, ideally, run
  them under `set -e` discipline of their own; changing the scripts'
  error handling is beyond this stage's path-substitution-only
  mandate and was not done.

### D.4 Other machine-specific assumptions in the two drivers

- **No other absolute or home paths.** Beyond the substituted `cd`
  line, neither sanitised driver contains any `/home/`, `/root/` or
  tilde path in its executable body. (The quoted original line
  survives once in each file, inside the caption comment block that
  records the change; see the completion report's consistency check.)
- **Driver-name coupling (not a path, recorded for completeness).**
  `run_full_pipeline_sanitised.sh` waits on `pgrep -f run_screens.sh`,
  the *original* driver name. If the sanitised copies are ever used
  under their sanitised file names, that pattern will not match a
  running `run_screens_sanitised.sh` process; the wait loop then
  relies on its 120-iteration cap and the screen-artifact test. This
  is a pre-existing property of the original script, preserved
  unchanged by the path substitution.
- **Environment layout assumptions (portable, not machine-specific).**
  Both drivers assume a project `.venv` at `<root>/.venv` with a
  `bin/python`, a POSIX shell environment with `pgrep` and `seq`
  available, and the Phase 5 artifact names and clip count (542)
  hard-coded in the screens driver. These assumptions travel with the
  project layout rather than with the original machine.
