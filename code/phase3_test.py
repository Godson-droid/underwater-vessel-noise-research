"""Phase 3: matcher applied to 60 real test windows (diagnostic only).

Sample: 60 rows spread evenly across Test.csv (linspace indices).
Generation: union of Matcher A (KD) and Matcher B (hash) candidate
starts, one verification with the exact-fingerprint ranking.
Confidence classes (thresholds set from the synthetic controls in
phase3_synth.py — what found vs not-found candidates look like there —
never from test outcomes):
  clean     best viol <= 0.01, >= 3 exact minutes, and no rival
            candidate with a materially different resid_share
  ambiguous a plausible candidate exists (viol <= 0.05, >= 1 exact
            minute) but a rival disagrees on resid_share by > 0.05
  none      no candidate passes viol <= 0.05 with an exact minute
Also reported per window: gain-aligned violation of the best candidate
(diagnostic for 'in the release but gain-shifted').
Test.csv is used for inference/diagnostic only; nothing is fitted.
"""
import json
import numpy as np
import pandas as pd
from common import DATA
from phase3_match import Matcher, HashMatcher, full_metrics, gain_aligned_viol

N_SAMPLE = 60


def load_test():
    df = pd.read_csv(DATA / "Test.csv")
    ids = df["ID"].to_numpy()
    X = df.drop(columns=["ID"]).to_numpy(dtype=np.float64).reshape(-1, 15, 25)
    return ids, X


def classify(top):
    if not top:
        return "none"
    best = top[0]
    if best["viol"] > 0.05 or best["n_exact_min"] < 1:
        return "none"
    rivals = [c for c in top[1:]
              if abs(c["resid_share"] - best["resid_share"]) > 0.05
              and c["viol"] <= 0.05 and c["n_exact_min"] >= 1]
    if best["viol"] <= 0.01 and best["n_exact_min"] >= 3 and not rivals:
        return "clean"
    return "ambiguous"


def main():
    ids, X = load_test()
    mA, mB = Matcher(), HashMatcher()
    sel = np.linspace(0, len(X) - 1, N_SAMPLE).astype(int)
    records = []
    for i in sel:
        T = X[i]
        starts = mA.candidate_starts(T) + mB.candidate_starts(T)
        cands = mA.verify(T, starts)[:5]
        top = []
        for c in cands:
            B = mA.V[c["start_idx"]:c["start_idx"] + 15].astype(np.float64)
            c.update(full_metrics(T, B))
            top.append(c)
        rec = dict(row=int(i), id=str(ids[i]), n_candidates=len(set(starts)),
                   cls=classify(top), top=top[:3])
        if top:
            B = mA.V[top[0]["start_idx"]:top[0]["start_idx"] + 15].astype(np.float64)
            off, gv = gain_aligned_viol(T, B)
            rec["gain_offset_db"] = off
            rec["gain_aligned_viol"] = gv
        records.append(rec)
        b = top[0] if top else {}
        print(i, rec["cls"], "viol", b.get("viol"), "nexact", b.get("n_exact_min"),
              "eq", None if not b else round(b.get("eq_frac", np.nan), 3),
              "share", None if not b else round(b.get("resid_share", np.nan), 3),
              "ncand", rec["n_candidates"], flush=True)
    with open(DATA / "phase3_test_records.json", "w") as f:
        json.dump(records, f)
    cls = [r["cls"] for r in records]
    print("\nclasses:", {c: cls.count(c) for c in ("clean", "ambiguous", "none")})
    for c in ("clean", "ambiguous", "none"):
        rs = [r for r in records if r["cls"] == c and r["top"]]
        if rs:
            print(c, "best viol median", round(float(np.median([r["top"][0]["viol"] for r in rs])), 4),
                  "| nexact median", float(np.median([r["top"][0]["n_exact_min"] for r in rs])),
                  "| resid_share median", round(float(np.median([r["top"][0]["resid_share"] for r in rs])), 3),
                  "| gain-aligned viol median",
                  round(float(np.median([r["gain_aligned_viol"] for r in rs])), 4))


if __name__ == "__main__":
    main()
