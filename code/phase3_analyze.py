"""Phase 3 analysis of phase3_synth_records.json (fresh run, with union).

Adds what the raw tables do not show:
  * recovery score conditional on the matcher having found the TRUE
    background vs having missed it (the bimodality behind the averages);
  * whether OBSERVABLE confidence metrics (viol, n_exact_min) can gate
    the matcher: coverage and score among accepted windows, for a small
    grid of rules fixed by the phase3_test.py class definitions;
  * hybrid pipelines: matcher estimate where the gate accepts, a named
    fallback elsewhere — the deployable shape of a matching submission.
All thresholds are evaluated on synthetic ground truth only.
"""
import json
import numpy as np
from common import DATA

recs = json.load(open(DATA / "phase3_synth_records.json"))
ESTS = ["oracle", "matchA", "matchB", "union", "quiet", "stat_or", "stat_guess"]


def rmse(rs, key):
    v = np.array([x[key] for x in rs], dtype=np.float64)
    t = np.array([x["true_share"] for x in rs])
    ok = np.isfinite(v)
    return float(np.sqrt(np.mean((v[ok] - t[ok]) ** 2))) if ok.any() else np.nan


def score(rs, key):
    return 1 - rmse(rs, key)


print(f"n records: {len(recs)}")
for split in ("A", "B", "AB"):
    rs = recs if split == "AB" else [x for x in recs if x["split"] == split]
    print(f"\n== split {split} (n={len(rs)}) ==")
    for e in ESTS:
        print(f"  {e:10s} RMSE {rmse(rs, e):.4f}  score {score(rs, e):.4f}")

print("\n== per share level (combined) ==")
print("share  " + "  ".join(f"{e:>9s}" for e in ESTS))
for s in sorted({x["target_share"] for x in recs}):
    rs = [x for x in recs if x["target_share"] == s]
    print(f"{s:5.2f}  " + "  ".join(f"{rmse(rs, e):9.4f}" for e in ESTS))

print("\n== conditional on finding the true background ==")
for e in ("matchA", "matchB", "union"):
    f = [x for x in recs if x[e + "_found"]]
    nf = [x for x in recs if not x[e + "_found"]]
    print(f"{e}: found {len(f)}/{len(recs)} ({len(f)/len(recs):.1%}) "
          f"score|found {score(f, e):.4f} | score|missed {score(nf, e):.4f} "
          f"(RMSE {rmse(nf, e):.4f})")

print("\n== found rate by share level ==")
for e in ("matchA", "matchB", "union"):
    row = []
    for s in sorted({x["target_share"] for x in recs}):
        rs = [x for x in recs if x["target_share"] == s]
        row.append(f"{s:.2f}:{np.mean([x[e + '_found'] for x in rs]):.2f}")
    print(f"  {e:7s} " + "  ".join(row))

both = [x for x in recs if np.isfinite(x["matchA"]) and np.isfinite(x["matchB"])]
d = np.array([x["matchA"] - x["matchB"] for x in both])
print(f"\nA vs B agreement: n={len(both)}, mean|diff| {np.abs(d).mean():.4f}, "
      f"RMSE between estimates {np.sqrt((d ** 2).mean()):.4f}")
for e in ("matchA", "matchB"):
    o = [x for x in recs if np.isfinite(x[e]) and np.isfinite(x["union"])]
    dd = np.array([x[e] - x["union"] for x in o])
    print(f"{e} vs union: n={len(o)}, RMSE between estimates "
          f"{np.sqrt((dd ** 2).mean()):.4f}")

print("\n== confidence gating on observable metrics (union matcher) ==")
print("rule                        coverage  score|accepted  found|accepted")
for vmax, nmin in ((0.01, 3), (0.01, 1), (0.02, 2), (0.05, 1), (0.05, 3), (0.10, 1)):
    acc = [x for x in recs if np.isfinite(x["union"])
           and x["union_viol"] <= vmax and x["union_nexact"] >= nmin]
    if acc:
        print(f"viol<={vmax:<5} nexact>={nmin:<3}      {len(acc)/len(recs):7.1%}  "
              f"{score(acc, 'union'):13.4f}  {np.mean([x['union_found'] for x in acc]):.3f}")

print("\n== hybrid pipelines (gate -> union estimate, else fallback) ==")
for vmax, nmin in ((0.01, 3), (0.05, 1)):
    for fb in ("const0.1", "quiet", "stat_guess"):
        errs = []
        for x in recs:
            if (np.isfinite(x["union"]) and x["union_viol"] <= vmax
                    and x["union_nexact"] >= nmin):
                v = x["union"]
            elif fb == "const0.1":
                v = 0.1
            else:
                v = x[fb]
            errs.append(v - x["true_share"])
        r = float(np.sqrt(np.mean(np.square(errs))))
        print(f"gate viol<={vmax} nexact>={nmin}, fallback {fb:10s}: "
              f"RMSE {r:.4f}  score {1 - r:.4f}")
