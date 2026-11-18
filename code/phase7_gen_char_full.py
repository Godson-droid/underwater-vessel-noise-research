"""Phase 7 workstream 2, full protocol generation: the train
(80k) and trainB (40k) companions to sim_p7char_trainA.npz,
same character-synthesis method, same pools and split
discipline (simulate_conv.build_pools('bandext')), maskfix
mask, fresh recorded seeds per set. trainA already exists
(seed 20358118); this script writes train (seed 20358119) and
trainB (seed 20358120). Prefix sim_p7char.
"""
import json
import numpy as np
import simulate_v3
from common import DATA, SEED
from simulate_conv import build_pools
from simulate_interact import draw_mask_fixed
from simulate_v3 import traffic_scores
from phase7_gen_char import site_shapes, CharPool, P_TRANSPLANT

SETS = [("train", 80000, SEED + 97000 + 3),
        ("trainB", 40000, SEED + 97000 + 4)]


def main():
    bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
    shapes = site_shapes(bg["X"], bg["site"])
    targets = sorted(shapes.keys())
    pools, _ = build_pools("bandext")
    simulate_v3.draw_mask = draw_mask_fixed
    meta = dict(tag="p7char_full", generation_seeds={},
                p_transplant=P_TRANSPLANT, sets={})
    for name, n, gseed in SETS:
        pool, clips, fl, wb = pools[name]
        cpool = CharPool(pool.X, pool.site, traffic_scores(pool.X),
                         shapes, targets, P_TRANSPLANT, None)
        r = np.random.default_rng(gseed)
        d = simulate_v3.gen(n, cpool, clips, r, force_long=fl,
                            want_bg=wb)
        np.savez_compressed(DATA / f"sim_p7char_{name}.npz", **d)
        meta["generation_seeds"][name] = gseed
        meta["sets"][name] = dict(
            n=n, n_transplant=cpool.n_transplant,
            n_draw=cpool.n_draw,
            label_mean=float(d["y"].mean()),
            zero_frac=float((d["y"] == 0).mean()))
        print(f"{name}: {d['X'].shape}; label mean "
              f"{d['y'].mean():.3f}; transplants "
              f"{cpool.n_transplant}/{cpool.n_draw}", flush=True)
    (DATA / "sim_p7char_full_meta.json").write_text(
        json.dumps(meta, indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
