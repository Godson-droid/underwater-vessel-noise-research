#!/bin/bash
# ---------------------------------------------------------------------
# Stage 2 sanitised working copy (18 November 2026).
# Original: code/run_full_pipeline.sh (preserved unchanged in the research tree).
# Changes relative to the original:
#   1. The original machine-specific project-directory command was
#      replaced with a project-root-relative command:
#      cd "${PROJECT_ROOT}/code"
#   2. An explicit preflight was added (final Stage 2 corrections,
#      18 November 2026): the driver exits immediately if
#      PROJECT_ROOT is unset or does not contain a code directory,
#      instead of continuing in the wrong directory.
#   3. The wait loop's process-name pattern now matches the
#      sanitised screens driver's file name
#      (run_screens_sanitised.sh), so the two sanitised drivers
#      remain consistent as a pair.
# PROJECT_ROOT must be set to the project root before running.
# Execution logic, parameters, seeds and behaviour are otherwise
# unchanged. See STAGE2_PATH_HYGIENE_REPORT.md.
# ---------------------------------------------------------------------
# Phase 5 follow-on pipeline (sequenced, resumable by artifact):
# waits for the screen driver, then bandext full, bandextdb
# conversion + screen, bandext+maskfix full.
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "ERROR: PROJECT_ROOT must point to the project root." >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}/code" ]]; then
    echo "ERROR: PROJECT_ROOT does not contain a code directory." >&2
    exit 1
fi

cd "${PROJECT_ROOT}/code" || exit 1
PY=../.venv/bin/python

for i in $(seq 1 120); do
  [ -f ../models/conv_min60_screen.json ] && break
  pgrep -f run_screens_sanitised.sh >/dev/null || break
  sleep 60
done
echo "[pipe] screen phase settled, starting bandext full generation"

if [ ! -f ../data/sim_cf_bandext_meta.json ]; then
  $PY simulate_conv.py bandext full > sim_cf_bandext_gen.log 2>&1
fi
echo "[pipe] bandext full training"
$PY train_conv.py bandext full > train_cf_bandext.log 2>&1

if [ ! -f ../data/passages_conv_bandextdb.npz ]; then
  echo "[pipe] bandextdb conversion"
  $PY convert_bandextdb.py > conv_p5_bandextdb.log 2>&1
fi
if [ ! -f ../models/conv_bandextdb_screen.json ]; then
  echo "[pipe] bandextdb screen generation"
  $PY simulate_conv.py bandextdb screen > sim_cs_bandextdb.log 2>&1
  echo "[pipe] bandextdb screen training"
  $PY train_conv.py bandextdb screen > train_cs_bandextdb.log 2>&1
fi

if [ ! -f ../data/sim_cf_bandext_maskfix_meta.json ]; then
  echo "[pipe] bandext+maskfix generation"
  $PY simulate_interact.py bandext maskfix > sim_cf_bandext_maskfix_gen.log 2>&1
fi
echo "[pipe] bandext+maskfix full training"
$PY train_conv.py bandext_maskfix full > train_cf_bandext_maskfix.log 2>&1
echo "[pipe] ALL DONE"
