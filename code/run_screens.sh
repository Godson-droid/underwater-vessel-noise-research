#!/bin/bash
# ---------------------------------------------------------------------
# Stage 2 sanitised working copy (18 November 2026).
# Original: code/run_screens.sh (preserved unchanged in the research tree).
# Changes relative to the original:
#   1. The original machine-specific project-directory command was
#      replaced with a project-root-relative command:
#      cd "${PROJECT_ROOT}/code"
#   2. An explicit preflight was added (final Stage 2 corrections,
#      18 November 2026): the driver exits immediately if
#      PROJECT_ROOT is unset or does not contain a code directory,
#      instead of continuing in the wrong directory.
# PROJECT_ROOT must be set to the project root before running.
# Execution logic, parameters, seeds and behaviour are otherwise
# unchanged. See STAGE2_PATH_HYGIENE_REPORT.md.
# ---------------------------------------------------------------------
# Phase 5 screening driver: for each conversion tag (assessment
# priority order), wait for its passages file to reach 542 clips,
# then generate screening sets and train the screening model.
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
for tag in magmean dbmean rect rawfft seg5 seg10 welch1s median bandext min60; do
  if [ -f ../models/conv_${tag}_screen.json ]; then
    echo "[driver] $tag screen already complete, skipping"
    continue
  fi
  # wait for the conversion of this tag to complete (542 clips)
  while true; do
    n=$($PY -c "
import numpy as np
try:
    d = np.load('../data/passages_conv_${tag}.npz', allow_pickle=True)
    print(len(d['names']))
except Exception:
    print(0)
" 2>/dev/null)
    [ "$n" = "542" ] && break
    sleep 30
  done
  echo "[driver] $tag passages complete, generating screen sets" 
  $PY simulate_conv.py $tag screen > sim_cs_${tag}.log 2>&1
  echo "[driver] $tag training screen model"
  $PY train_conv.py $tag screen > train_cs_${tag}.log 2>&1
  echo "[driver] $tag screen done"
done
echo "[driver] ALL SCREENS DONE"
