"""Mechanical validator for Zindi Underwater Vessel Noise submission files.

Checks, against data/Test.csv and data/sample_submission.csv:
  - header is exactly ID,share_added_vessel,score_added_vessel
  - exactly 1,500 rows
  - ID set matches Test.csv exactly (no missing, extra, or duplicated IDs)
  - both prediction columns are finite numbers in [0, 1]

Usage: python3 validate_submission.py <submission.csv> [<submission2.csv> ...]
Exit code 0 if every file passes, 1 otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REQUIRED_COLUMNS = ["ID", "share_added_vessel", "score_added_vessel"]


def validate(path: Path, test_ids: pd.Index) -> list[str]:
    problems = []
    sub = pd.read_csv(path, dtype={"ID": str})
    if sub.columns.tolist() != REQUIRED_COLUMNS:
        problems.append(f"header is {sub.columns.tolist()}, expected {REQUIRED_COLUMNS}")
        return problems
    if len(sub) != len(test_ids):
        problems.append(f"{len(sub)} rows, expected {len(test_ids)}")
    if not sub["ID"].is_unique:
        problems.append("duplicate IDs present")
    missing = set(test_ids) - set(sub["ID"])
    extra = set(sub["ID"]) - set(test_ids)
    if missing:
        problems.append(f"{len(missing)} test IDs missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unknown IDs present, e.g. {sorted(extra)[:3]}")
    preds = sub[REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(preds).all():
        problems.append("non-finite prediction values present")
    if ((preds < 0) | (preds > 1)).any():
        problems.append("prediction values outside [0, 1]")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    test = pd.read_csv(DATA_DIR / "Test.csv", dtype={"ID": str}, usecols=["ID"])
    ok = True
    for arg in sys.argv[1:]:
        path = Path(arg)
        problems = validate(path, test["ID"])
        if problems:
            ok = False
            print(f"FAIL {path}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"PASS {path} (1500 rows, IDs match Test.csv, values finite in [0,1])")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
