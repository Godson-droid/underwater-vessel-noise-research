"""Phase 5 interaction test: combine independently justified
corrections with the strongest conversion variant, factorial style.

Cells (the conversion variant's own full run is the fourth cell,
produced by simulate_conv.py full):
  <tag>+chan     variant passages + background channel drawn from
                 the release's inter-site spectral differences
                 (workstream 4's family, simulate_bgchan.ChanPool).
  <tag>+maskfix  variant passages + mask process corrected to the
                 real mask's measured FORM (workstream 3 audit:
                 real pattern fractions are complete 0.517,
                 minute-1-only 0.413, minute-15-only 0.065,
                 >=2 0.005 with zero 'other single' drops in 1,500
                 windows; v3's Bernoulli gives 0.005 other-single
                 and 0.034 >=2). The corrected draw drops only minute
                 1, only minute 15, or both, at the real fractions
                 (the >=2 composition {1,15} is an assumption; the
                 audit's n=8 cannot resolve it).
  <tag>+both     both corrections together (tests additivity).

Memberships are simulate_conv.build_pools(tag)'s (v3 split
discipline); generation uses a fresh recorded seed per cell, so
pairing against v3 and against the variant's own full run is by
cross-scoring on identical files, as elsewhere in Phase 5. Sizes are
the full v3 sizes; train_conv.py <cell-tag> full trains and
evaluates them unchanged (files are sim_cf_<cell-tag>_*.npz).

Usage: python simulate_interact.py <tag> <chan|maskfix|both>
"""
import json
import sys
import numpy as np
import simulate_v3
from common import DATA, SEED
from simulate_bgchan import ChanPool, site_spectra
from simulate_conv import build_pools, FULL_SIZES
from simulate_v3 import traffic_scores

CELL_SEEDS = {"chan": SEED + 93000, "maskfix": SEED + 94000,
              "both": SEED + 95000}
ORDER = ["train", "trainA", "trainB", "val_random", "val_site",
         "val_period", "val_long", "oracle"]


def draw_mask_fixed(r):
    """Mask with the real windows' measured pattern fractions."""
    u = r.random()
    mask = np.ones(15, dtype=bool)
    if u < 0.517:
        pass
    elif u < 0.517 + 0.413:
        mask[0] = False
    elif u < 0.517 + 0.413 + 0.065:
        mask[14] = False
    else:
        mask[0] = False
        mask[14] = False
    return mask


def main():
    tag, cell = sys.argv[1], sys.argv[2]
    assert cell in CELL_SEEDS, cell
    pools, _ = build_pools(tag)
    if cell in ("chan", "both"):
        bg = np.load(DATA / "bg_windows.npz", allow_pickle=True)
        specs = site_spectra(bg["X"], bg["site"])
        # V3Pool does not retain the traffic array; traffic_scores is
        # a per-window function, so recomputing it on the pool's own
        # windows reproduces the original slice exactly.
        pools = {k: (ChanPool(p.X, p.site, traffic_scores(p.X), specs),
                     c, fl, wb)
                 for k, (p, c, fl, wb) in pools.items()}
    if cell in ("maskfix", "both"):
        simulate_v3.draw_mask = draw_mask_fixed
    gseed = CELL_SEEDS[cell] + abs(hash(tag)) % 1000
    r = np.random.default_rng(gseed)
    cell_tag = f"{tag}_{cell}"
    prefix = f"sim_cf_{cell_tag}"
    meta = dict(tag=cell_tag, base_tag=tag, cell=cell, mode="full",
                seed=SEED, generation_seed=gseed,
                pools={k: int(len(v[0].X)) for k, v in pools.items()},
                sets={})
    for name in ORDER:
        pool, pc, fl, wb = pools[name]
        d = simulate_v3.gen(FULL_SIZES[name], pool, pc, r,
                            force_long=fl, want_bg=wb)
        np.savez_compressed(DATA / f"{prefix}_{name}.npz", **d)
        meta["sets"][name] = dict(
            label_mean=float(d["y"].mean()),
            zero_frac=float((d["y"] == 0).mean()))
        print(f"{name}: {d['X'].shape}; label mean {d['y'].mean():.3f}",
              flush=True)
    (DATA / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=1))
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
