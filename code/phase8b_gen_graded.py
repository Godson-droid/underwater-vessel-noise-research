"""Phase 8B: graded character transplants x independent shares.

Godson's authorised Phase 8B experiment (2026-11-17): Phase 7's
character transplant applied at a predeclared graded strength
s in {0, 0.25, 0.5, 0.75, 1.0}, crossed with the SAME added-
vessel share distribution at every level.

Method: the Phase 7 generation (phase7_gen_char.py /
phase7_gen_char_full.py) is rerun with identical seeds, pools,
P_TRANSPLANT, target family and draw sequence. The single
change: when a transplant fires, the applied shape is the
graded mixture (1-s)*source_shape + s*target_shape in per-band
energy-share space; per-band energies are multiplied by
mixture/source and rescaled to the original total energy
(Phase 7's energy preservation, unchanged). At s=1.0 the
mixture IS the target shape, so the s=1.0 corpora must
reproduce the Phase 7 corpora bit-exactly (NaN-aware) - the
pipeline validation gate, checked in main(). At s=0 no
transplant is applied.

Paired generation: the random stream is consumed draw-for-draw
identically at every strength (same r.random() and
r.integers(len(targets)) calls in the same order), so drawn
shares, masks, passages and transplant targets are identical
across levels; only the interpolation factor differs. Per-draw
records (target drawn, whether a transplant would fire) are
stored in each corpus file for the independence verification.

Usage: python phase8b_gen_graded.py <levels csv> <sets csv>
e.g.:  python phase8b_gen_graded.py 0,25,50,75,100 trainA
       python phase8b_gen_graded.py 50 train,trainB
Levels are given in percent. Existing corpora are skipped
(checkpointing). Prefix sim_p8b_s{level}.
"""
import json
import sys
import numpy as np
import simulate_v3
from common import DATA, SEED
from simulate_conv import build_pools
from simulate_interact import draw_mask_fixed
from simulate_v3 import traffic_scores
from phase7_gen_char import site_shapes, CharPool, P_TRANSPLANT

SETS = {"trainA": (40000, SEED + 97000 + 2),
        "train": (80000, SEED + 97000 + 3),
        "trainB": (40000, SEED + 97000 + 4)}


class GradedCharPool(CharPool):
    """CharPool with the transplant applied at graded strength.

    Draw consumption is byte-identical to CharPool's; the
    strength scales only the applied shape mixture. Records,
    per draw, the target drawn (or None when the p-draw did not
    fire) and whether a transplant would fire (target drawn and
    != source site) - identical across strengths by design.
    """

    def __init__(self, *a, strength=1.0, **kw):
        super().__init__(*a, **kw)
        self.strength = strength
        self.draw_target = []
        self.draw_would = []

    def draw_background(self, r):
        # V3Pool.draw_background, exactly as CharPool invokes it.
        E, site, mode = super(CharPool, self).draw_background(r)
        self.n_draw += 1
        tgt = None
        would = False
        if r.random() < self.p:
            tgt = self.targets[int(r.integers(len(self.targets)))]
            if tgt != site:
                would = True
                if self.strength > 0.0:
                    src = self.shapes[site]
                    mix = ((1.0 - self.strength) * src
                           + self.strength * self.shapes[tgt])
                    ratio = mix / np.maximum(src, 1e-30)
                    tot0 = E.sum()
                    E = E * ratio[None, :]
                    E *= tot0 / max(E.sum(), 1e-30)
                    self.n_transplant += 1
                    self.target_counts[tgt] += 1
        self.draw_target.append(tgt)
        self.draw_would.append(would)
        return E, site, mode


def gen_level_set(level_pct, setname, pools, shapes, targets):
    n, gseed = SETS[setname]
    tag = "p8b_s{}_{}".format(level_pct, setname)
    out_path = DATA / "sim_{}.npz".format(tag)
    if out_path.exists():
        print("{}: exists, skipped".format(tag), flush=True)
        return None
    pool, clips, fl, wb = pools[setname]
    cpool = GradedCharPool(pool.X, pool.site,
                           traffic_scores(pool.X), shapes, targets,
                           P_TRANSPLANT, None,
                           strength=level_pct / 100.0)
    r = np.random.default_rng(gseed)
    d = simulate_v3.gen(n, cpool, clips, r, force_long=fl,
                        want_bg=wb)
    d["draw_target"] = np.array(cpool.draw_target, dtype=object)
    d["draw_would"] = np.array(cpool.draw_would, dtype=bool)
    np.savez_compressed(out_path, **d)
    info = dict(level_pct=level_pct, set=setname, n=n,
                generation_seed=gseed,
                n_draw=cpool.n_draw,
                n_would_transplant=int(sum(cpool.draw_would)),
                n_applied=cpool.n_transplant,
                target_counts={k: v for k, v in
                               cpool.target_counts.items()},
                label_mean=float(d["y"].mean()),
                zero_frac=float((d["y"] == 0).mean()))
    print("{}: {}; label mean {:.4f}; would-transplant {}/{}; "
          "applied {}".format(tag, d["X"].shape, d["y"].mean(),
                              info["n_would_transplant"],
                              cpool.n_draw, cpool.n_transplant),
          flush=True)
    if level_pct == 100:
        # Pipeline validation gate: bit-identity (NaN-aware)
        # against the Phase 7 corpus for this set.
        ref = np.load(DATA / "sim_p7char_{}.npz".format(setname),
                      allow_pickle=True)
        same_x = np.array([np.array_equal(a, b, equal_nan=True)
                           for a, b in zip(d["X"], ref["X"])])
        info["verify_vs_p7char"] = dict(
            rows_identical_X=int(same_x.sum()),
            rows_differing_X=int((~same_x).sum()),
            y_max_abs_diff=float(np.abs(
                d["y"].astype(np.float64)
                - ref["y"].astype(np.float64)).max()))
        print("{}: rows identical to Phase 7 corpus: {}/{}; "
              "y max |diff| {:.2e}".format(
                  tag, int(same_x.sum()), n,
                  info["verify_vs_p7char"]["y_max_abs_diff"]),
              flush=True)
    return info


def main():
    levels = [int(v) for v in sys.argv[1].split(",")]
    sets = sys.argv[2].split(",")
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    shapes = site_shapes(bg["X"], bg["site"])
    targets = sorted(shapes.keys())
    pools, _ = build_pools("bandext")
    simulate_v3.draw_mask = draw_mask_fixed
    meta_path = DATA / "sim_p8b_meta.json"
    meta = (json.loads(meta_path.read_text())
            if meta_path.exists() else
            dict(tag="p8b_graded", p_transplant=P_TRANSPLANT,
                 strength_scale=[0, 0.25, 0.5, 0.75, 1.0],
                 note="Paired generation: identical seeds, pools, "
                      "draw sequence and target family as Phase 7 "
                      "at every strength; only the applied shape "
                      "mixture differs.", corpora={}))
    for lv in levels:
        for s in sets:
            info = gen_level_set(lv, s, pools, shapes, targets)
            if info is not None:
                meta["corpora"]["s{}_{}".format(lv, s)] = info
                meta_path.write_text(json.dumps(meta, indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
