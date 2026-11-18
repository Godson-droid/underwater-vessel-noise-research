"""Phase 8B: strength x share independence verification.

Godson's central experimental requirement: every character-
strength level is crossed with the SAME added-vessel share
distribution; stronger transplants must not systematically
receive smaller or larger additions. Generation is paired
(identical draw sequence at every level), so this script
verifies the property on the generated trainA corpora and
reports it before any training:

- draw sequences (targets drawn, would-transplant flags)
  identical across levels;
- draw / transplant / target counts per level;
- realised-label distribution per level: mean, std, zero
  fraction, share-band fractions - overall, among would-
  transplant windows, and among the rest;
- per-window label differences across levels (the label is
  computed against observed-minute background energy, which
  the transplant preserves only to second order, so a
  microscopic wobble is possible in principle): max and mean
  |dy|, fraction exactly equal, and the mean signed dy among
  would-transplant windows (a systematic direction would show
  here).

Record: models/phase8b_independence.json.
"""
import json
import numpy as np
from common import DATA
from phase6_lib import MODEL_DIR, SHARE_BANDS

LEVELS = [0, 25, 50, 75, 100]


def band_fracs(y):
    out = {}
    for lo, hi, lab in SHARE_BANDS:
        m = (y == 0) if hi == 0.0 else ((y > lo) & (y <= hi))
        out[lab] = float(m.mean())
    return out


def label_stats(y):
    return dict(mean=float(y.mean()), std=float(y.std()),
                zero_frac=float((y == 0).mean()),
                band_fracs=band_fracs(y))


def main():
    corp = {}
    for lv in LEVELS:
        d = np.load(DATA / "sim_p8b_s{}_trainA.npz".format(lv),
                    allow_pickle=True)
        corp[lv] = d
        print("s{}: y mean {:.4f}, zero {:.4f}, would {}"
              .format(lv, d["y"].mean(), (d["y"] == 0).mean(),
                      int(d["draw_would"].sum())), flush=True)

    out = dict(levels={})
    ref_t = corp[100]["draw_target"]
    ref_w = corp[100]["draw_would"]
    for lv in LEVELS:
        d = corp[lv]
        y = d["y"].astype(np.float64)
        w = d["draw_would"]
        entry = dict(
            draws_identical_to_s100=bool(
                np.array_equal(d["draw_target"], ref_t)
                and np.array_equal(d["draw_would"], ref_w)),
            n_draw=int(len(y)),
            n_would_transplant=int(w.sum()),
            labels_all=label_stats(y),
            labels_transplanted=label_stats(y[w]),
            labels_natural=label_stats(y[~w]))
        out["levels"][str(lv)] = entry

    pairs = {}
    for a, b in [(0, 100), (0, 25), (25, 50), (50, 75), (75, 100)]:
        ya = corp[a]["y"].astype(np.float64)
        yb = corp[b]["y"].astype(np.float64)
        dy = yb - ya
        w = corp[a]["draw_would"]
        pairs["{}->{}".format(a, b)] = dict(
            max_abs_dy=float(np.abs(dy).max()),
            mean_abs_dy=float(np.abs(dy).mean()),
            frac_exactly_equal=float((dy == 0).mean()),
            mean_signed_dy_transplanted=float(dy[w].mean()),
            mean_signed_dy_all=float(dy.mean()))
    out["label_diffs"] = pairs
    (MODEL_DIR / "phase8b_independence.json").write_text(
        json.dumps(out, indent=1))
    for k, v in pairs.items():
        print("{}: max|dy| {:.2e}, mean|dy| {:.2e}, exact {:.4f}, "
              "signed dy (transplanted) {:+.2e}".format(
                  k, v["max_abs_dy"], v["mean_abs_dy"],
                  v["frac_exactly_equal"],
                  v["mean_signed_dy_transplanted"]), flush=True)
    print("wrote models/phase8b_independence.json", flush=True)


if __name__ == "__main__":
    main()
