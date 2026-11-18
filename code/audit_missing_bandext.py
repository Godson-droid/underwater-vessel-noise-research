"""Workstream 3 follow-up: real-window prediction means by
missingness pattern under the bandext full main model, joined by ID
to its saved test predictions, using audit_missing.pattern_of.
Compares against the frozen v3 main model's group means recorded
in models/audit_missing.json.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, ROOT
from audit_missing import pattern_of

test = pd.read_csv(DATA / "Test.csv", dtype={"ID": str})
fcols = [c for c in test.columns if c != "ID"]
Xt = test[fcols].to_numpy(dtype=np.float32).reshape(-1, 15, 25)
pats = np.array([pattern_of(Xt[i]) for i in range(len(Xt))])

pred = pd.read_csv(ROOT / "models" / "conv_bandext_test_predictions.csv",
                   dtype={"ID": str})
m = test[["ID"]].merge(pred, on="ID", how="left")
assert m["share_added_vessel"].notna().all()
p = m["share_added_vessel"].to_numpy()

out = {}
for g in ["complete", "m01_only", "m15_only", "other_single", "ge2"]:
    sel = pats == g
    if sel.sum():
        out[g] = dict(n=int(sel.sum()), pred_mean=float(p[sel].mean()),
                      pred_median=float(np.median(p[sel])),
                      pred_frac_below_0p05=float((p[sel] < 0.05).mean()))
out["overall"] = dict(n=int(len(p)), pred_mean=float(p.mean()))
(ROOT / "models" / "audit_missing_bandext.json").write_text(
    json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
