"""Phase 7 workstream 2: held-out site-character synthesis.

Godson's direction (Phase 7, item 2) and the Phase 6 report's
recommendation (4): synthesise training backgrounds that
reproduce the variation observed across sites — especially Bird
Island's character — while the added-vessel signal construction
stays independently controlled and unchanged.

Method. Each released site's CHARACTER is summarised by one
aggregate statistic: the median per-band energy-share vector of
its screened pool windows (a 25-vector, its typical spectral
shape). A training background drawn by the frozen V3Pool is then,
with probability P_TRANSPLANT, given another site's character:
its per-band energies are multiplied by target_shape /
source_shape and the window is rescaled to its original total
energy. Temporal structure, total level, traffic content and the
passage/mask/label machinery are untouched (bandext passages,
maskfix mask, observed-cells label — the frozen construction).
No site holdout WINDOW is trained on; what crosses the holdout
boundary is only the two holdout sites' aggregate median shapes,
which are part of the transplant target family by Godson's
explicit direction. That caveat is recorded in the results: the
site holdout remains a test of unseen windows and unseen
temporal/traffic character, but its spectral-shape dimension is
no longer fully blind for this variant.

Pools and split discipline: simulate_conv.build_pools('bandext')
(v3 memberships). Generation: fresh recorded seed; trainA only
(the screen set). Prefix sim_p7char.
"""
import json
import numpy as np
import simulate_v3
from common import DATA, SEED, db_to_lin
from simulate_conv import build_pools
from simulate_interact import draw_mask_fixed
from simulate_v3 import V3Pool, traffic_scores

P_TRANSPLANT = 0.6
GSEED = SEED + 97000 + 2          # 20358118, recorded in meta


def site_shapes(BX, BS):
    """Median per-band energy-share vector per site."""
    E = db_to_lin(BX.astype(np.float64))
    band_tot = E.sum(axis=1)                       # (n,25)
    share = band_tot / np.maximum(band_tot.sum(axis=1, keepdims=True),
                                  1e-30)
    shapes = {}
    for s in np.unique(BS):
        sh = np.median(share[BS == s], axis=0)
        shapes[str(s)] = sh / sh.sum()
    return shapes


class CharPool(V3Pool):
    def __init__(self, X, site, traffic, shapes, targets, p, rng_probe):
        super().__init__(X, site, traffic)
        self.shapes = shapes
        self.targets = list(targets)
        self.p = p
        self.n_draw = 0
        self.n_transplant = 0
        self.target_counts = {t: 0 for t in self.targets}

    def draw_background(self, r):
        E, site, mode = super().draw_background(r)
        self.n_draw += 1
        if r.random() < self.p:
            tgt = self.targets[int(r.integers(len(self.targets)))]
            if tgt != site:
                ratio = self.shapes[tgt] / np.maximum(self.shapes[site],
                                                      1e-30)
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
    print("sites:", targets, flush=True)
    # Bird Island character, for the record: band shares vs the
    # all-site median shape.
    med = np.median(np.stack([shapes[s] for s in targets]), axis=0)
    for s in ("4 Bird Island", "9 Fish Hoek"):
        print(f"{s} shape / median shape, bands 0-5: "
              f"{(shapes[s][:6] / med[:6]).round(3).tolist()}", flush=True)

    pools, _ = build_pools("bandext")
    pool, clips, fl, wb = pools["trainA"]
    cpool = CharPool(pool.X, pool.site, traffic_scores(pool.X),
                     shapes, targets, P_TRANSPLANT, None)
    simulate_v3.draw_mask = draw_mask_fixed
    r = np.random.default_rng(GSEED)
    d = simulate_v3.gen(40000, cpool, clips, r, force_long=fl,
                        want_bg=wb)
    np.savez_compressed(DATA / "sim_p7char_trainA.npz", **d)
    meta = dict(tag="p7char", base="bandext_maskfix pools",
                generation_seed=GSEED, p_transplant=P_TRANSPLANT,
                n_draw=cpool.n_draw, n_transplant=cpool.n_transplant,
                transplant_frac=cpool.n_transplant / max(cpool.n_draw, 1),
                target_counts=cpool.target_counts,
                shapes={s: shapes[s].round(6).tolist() for s in targets},
                label_mean=float(d["y"].mean()),
                zero_frac=float((d["y"] == 0).mean()))
    (DATA / "sim_p7char_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"trainA: {d['X'].shape}; label mean {d['y'].mean():.3f}, "
          f"zero {(d['y'] == 0).mean():.3f}; transplants "
          f"{cpool.n_transplant}/{cpool.n_draw}", flush=True)
    print("meta written", flush=True)


if __name__ == "__main__":
    main()
