"""Phase 8 opening experiment: the Bird Island exclusion test.

Godson's direction (2026-11-17): Phase 7's site-character
transplant family included Bird Island's own aggregate shape,
so its site-holdout correction could be either (a) genuine
learning of a transferable class of low-frequency-heavy natural
backgrounds, or (b) partial adaptation to Bird Island's
particular distribution. This experiment separates the two.

Method: the Phase 7 generation (phase7_gen_char.py /
phase7_gen_char_full.py) is rerun with the IDENTICAL seeds,
pools, split discipline, P_TRANSPLANT and frozen bandext_maskfix
construction, with ONE change: Bird Island's aggregate shape is
removed from the transplant targets. A background whose
transplant draw selects Bird Island keeps its natural character
instead — Bird Island's shape is never applied to any training
window.

Controlled-comparison design: the target list passed to the
draw keeps its Phase 7 length and order, so the random stream
is consumed identically to Phase 7 draw-for-draw; the exclusion
acts only on whether the drawn target is applied. Every other
site's transplant count is therefore expected to match Phase 7
exactly, and the only difference between the sim_p7char_* and
sim_p8excl_* corpora is the treatment of the windows that drew
Bird Island. The verification block in main() checks this
against the recorded Phase 7 counts and by direct row identity
against the Phase 7 files.

Sets and seeds (identical to Phase 7): trainA 40k seed
20358118 (SEED+97000+2), train 80k seed 20358119 (+3), trainB
40k seed 20358120 (+4). Prefix sim_p8excl.
"""
import json
import numpy as np
import simulate_v3
from common import DATA, SEED
from simulate_conv import build_pools
from simulate_interact import draw_mask_fixed
from simulate_v3 import traffic_scores
from phase7_gen_char import site_shapes, CharPool, P_TRANSPLANT

EXCLUDED = "4 Bird Island"

SETS = [("trainA", 40000, SEED + 97000 + 2),
        ("train", 80000, SEED + 97000 + 3),
        ("trainB", 40000, SEED + 97000 + 4)]


class ExclCharPool(CharPool):
    """CharPool with one target removed from application.

    The draw sequence (r.random(), r.integers(len(targets))) is
    byte-identical to CharPool's; when the drawn target is the
    excluded site, no transplant is applied and the draw is
    counted separately. The excluded site's shape is never read.
    """

    def __init__(self, *a, excluded=EXCLUDED, **kw):
        super().__init__(*a, **kw)
        self.excluded = excluded
        self.n_excluded = 0

    def draw_background(self, r):
        # V3Pool.draw_background, exactly as CharPool invokes it.
        E, site, mode = super(CharPool, self).draw_background(r)
        self.n_draw += 1
        if r.random() < self.p:
            tgt = self.targets[int(r.integers(len(self.targets)))]
            if tgt == self.excluded:
                self.n_excluded += 1
            elif tgt != site:
                ratio = self.shapes[tgt] / np.maximum(
                    self.shapes[site], 1e-30)
                tot0 = E.sum()
                E = E * ratio[None, :]
                E *= tot0 / max(E.sum(), 1e-30)
                self.n_transplant += 1
                self.target_counts[tgt] += 1
        return E, site, mode


def main():
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    shapes = site_shapes(bg["X"], bg["site"])
    targets = sorted(shapes.keys())
    assert EXCLUDED in targets, targets
    print("target family (draw list):", targets, flush=True)
    print(f"excluded from application: {EXCLUDED}", flush=True)
    pools, _ = build_pools("bandext")
    simulate_v3.draw_mask = draw_mask_fixed
    meta = dict(tag="p8excl", excluded_target=EXCLUDED,
                p_transplant=P_TRANSPLANT,
                note="Bird Island shape never applied; draw "
                     "sequence identical to Phase 7 (same seeds, "
                     "same target list length/order).",
                sets={})
    for name, n, gseed in SETS:
        pool, clips, fl, wb = pools[name]
        cpool = ExclCharPool(pool.X, pool.site,
                             traffic_scores(pool.X), shapes, targets,
                             P_TRANSPLANT, None, excluded=EXCLUDED)
        r = np.random.default_rng(gseed)
        d = simulate_v3.gen(n, cpool, clips, r, force_long=fl,
                            want_bg=wb)
        np.savez_compressed(DATA / f"sim_p8excl_{name}.npz", **d)
        meta["sets"][name] = dict(
            n=n, generation_seed=gseed, n_draw=cpool.n_draw,
            n_transplant=cpool.n_transplant,
            n_excluded=cpool.n_excluded,
            target_counts={k: v for k, v in
                           cpool.target_counts.items()},
            label_mean=float(d["y"].mean()),
            zero_frac=float((d["y"] == 0).mean()))
        print(f"{name}: {d['X'].shape}; label mean "
              f"{d['y'].mean():.4f}; transplants "
              f"{cpool.n_transplant}/{cpool.n_draw}; excluded "
              f"draws {cpool.n_excluded}", flush=True)

        # Verification against the Phase 7 corpus: row identity.
        ref = np.load(DATA / f"sim_p7char_{name}.npz",
                      allow_pickle=True)
        same_x = np.array([np.array_equal(a, b) for a, b in
                           zip(d["X"], ref["X"])])
        meta["sets"][name]["verify_vs_p7char"] = dict(
            rows_identical_X=int(same_x.sum()),
            rows_differing_X=int((~same_x).sum()),
            y_identical_on_identical_X=bool(
                np.array_equal(d["y"][same_x], ref["y"][same_x])))
        print(f"{name}: rows identical to Phase 7 corpus: "
              f"{int(same_x.sum())}/{n}; differing "
              f"{int((~same_x).sum())}", flush=True)
    (DATA / "sim_p8excl_meta.json").write_text(json.dumps(meta,
                                                         indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
