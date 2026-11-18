"""B0 — constant baseline for the Zindi Underwater Vessel Noise Challenge.

Reproduces the starter notebook's constant baseline exactly:
    share_added_vessel = 0.1   for every window
    score_added_vessel = 0.5   for every window
(maritime_starter_notebook.ipynb, cells 5-6; the notebook states these are
"arbitrary starting values, not learned estimates").

Rows are emitted in sample_submission.csv order, with its IDs.
Deterministic; no randomness, no seed needed. Nothing is fitted anywhere.

Usage: python3 b0_constant.py
Writes: ../submissions/submission_b0_constant.csv
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SHARE_CONSTANT = 0.1   # starter notebook cell 6
SCORE_CONSTANT = 0.5   # starter notebook cell 6


def main() -> None:
    sample = pd.read_csv(BASE / "data" / "sample_submission.csv", dtype={"ID": str})
    submission = sample.copy()
    submission["share_added_vessel"] = SHARE_CONSTANT
    submission["score_added_vessel"] = SCORE_CONSTANT
    out = BASE / "submissions" / "submission_b0_constant.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print(f"Wrote {out}: {len(submission)} rows, share={SHARE_CONSTANT}, score={SCORE_CONSTANT}")


if __name__ == "__main__":
    main()
