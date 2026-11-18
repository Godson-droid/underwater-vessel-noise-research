"""Phase 6 workstream 5 generation: two extra TRAIN sets from the
frozen train pool (build_pools('bandext') memberships, maskfix mask),
differing only in the training-data axis under test.

  moresame — identical distribution to the frozen bandext_maskfix
             generation, fresh RNG stream. Tests VOLUME.
  diverse  — broadened variation axes (share prior flattened toward
             mid/high shares and the hard low band, wider colouring,
             wider warp, wider gain envelope, more mixture/busy
             backgrounds, wider background jitter/drift, more
             ambient-carrying additions). Tests DIVERSITY at equal
             volume (80k).

Evaluation of both is on the identical frozen holdout files.
Usage: python phase6_gen_coverage.py <moresame|diverse>
"""
import json
import sys
import numpy as np
import simulate_v3
from common import DATA, SEED
from simulate_conv import build_pools
from simulate_interact import draw_mask_fixed

N = 80000


def gen_moresame():
    simulate_v3.draw_mask = draw_mask_fixed
    pools, _ = build_pools("bandext")
    pool, clips, fl, wb = pools["train"]
    gseed = SEED + 120000
    r = np.random.default_rng(gseed)
    d = simulate_v3.gen(N, pool, clips, r, force_long=False,
                        want_bg=False)
    np.savez_compressed(DATA / "sim_p6ms_train.npz", **d)
    meta = dict(kind="moresame", generation_seed=gseed, n=N,
                label_mean=float(d["y"].mean()),
                zero_frac=float((d["y"] == 0).mean()))
    (DATA / "sim_p6ms_meta.json").write_text(json.dumps(meta, indent=1))
    print("moresame done", meta, flush=True)


def diverse_share(r):
    if r.random() < 0.20:
        return 0.0
    u = r.random()
    if u < 0.25:
        return r.uniform(0.0, 0.05)
    if u < 0.50:
        return r.uniform(0.05, 0.20)
    if u < 0.75:
        return r.uniform(0.20, 0.50)
    return r.uniform(0.50, 0.98)


def diverse_colour(r):
    steps = r.normal(0, 2.2, size=25).cumsum()
    steps -= np.linspace(steps[0], steps[-1], 25)
    return np.clip(steps, -10, 10)


def diverse_gain_profile(seg_lin, r):
    L = seg_lin.shape[0]
    if L <= 1:
        return seg_lin
    steps = r.normal(0, 1.5, size=L).cumsum()
    steps -= np.linspace(steps[0], steps[-1], L)
    g = np.clip(steps, -7, 7) + r.uniform(-2.5, 2.5, size=L)
    if r.random() < 0.4:
        k = int(r.integers(1, L))
        g[k:] += r.choice([-1.0, 1.0]) * r.uniform(2, 5)
    g = np.clip(g, -10, 10)
    return seg_lin * (10.0 ** (g / 10.0))[:, None]


def diverse_jitter(self, E, r):
    steps = r.normal(0, 1.4, size=25).cumsum()
    steps -= np.linspace(steps[0], steps[-1], 25)
    g = np.clip(steps, -4, 4)
    return E * (10.0 ** (g / 10.0))


def diverse_drift(self, E, r):
    env = r.normal(0, 1.1, size=15).cumsum()
    env -= np.linspace(env[0], env[-1], 15)
    env = np.clip(env, -4, 4)

    def colouring():
        s = r.normal(0, 1.3, size=25).cumsum()
        s -= np.linspace(s[0], s[-1], 25)
        return np.clip(s, -3.5, 3.5)
    c1, c2 = colouring(), colouring()
    t = np.linspace(0.0, 1.0, 15)[:, None]
    g = env[:, None] + (1.0 - t) * c1[None, :] + t * c2[None, :]
    return E * (10.0 ** (g / 10.0))


def diverse_build_event(blocks, r, force_long=False):
    from simulate_v2 import build_event_lin, draw_duration, warp_time
    D = r.uniform(12.0, 15.01) if force_long else draw_duration(r)
    w = r.uniform(0.4, 2.5)
    src_len = int(np.clip(round(D / w), 1, 40))
    seg = build_event_lin(blocks, src_len, r)
    if r.random() < 0.30:
        seg = seg[::-1].copy()
    new_len = int(np.clip(round(seg.shape[0] * w), 1, 15))
    seg = warp_time(seg, new_len, r)
    seg = simulate_v3.colour_tv(seg, r)
    seg = simulate_v3.gain_profile(seg, r)
    off = int(r.integers(0, 15 - new_len + 1))
    placed = np.zeros((15, 25))
    placed[off:off + new_len] = seg
    return placed, new_len


def gen_diverse():
    simulate_v3.draw_mask = draw_mask_fixed
    simulate_v3.draw_share = diverse_share
    simulate_v3.smooth_colour = diverse_colour
    simulate_v3.gain_profile = diverse_gain_profile
    simulate_v3.build_event_v3 = diverse_build_event
    simulate_v3.BG_MODE_P = (0.25, 0.55)
    simulate_v3.AMBIENT_P = 0.40
    simulate_v3.V3Pool._jitter = diverse_jitter
    simulate_v3.V3Pool._drift = diverse_drift
    pools, _ = build_pools("bandext")
    pool, clips, fl, wb = pools["train"]
    gseed = SEED + 121000
    r = np.random.default_rng(gseed)
    d = simulate_v3.gen(N, pool, clips, r, force_long=False,
                        want_bg=False)
    np.savez_compressed(DATA / "sim_p6dv_train.npz", **d)
    meta = dict(kind="diverse", generation_seed=gseed, n=N,
                label_mean=float(d["y"].mean()),
                zero_frac=float((d["y"] == 0).mean()),
                changes="share prior flattened (zero 0.20; quartiles "
                        "over (0,.05)/(.05,.2)/(.2,.5)/(.5,.98)); "
                        "colour clip +-10 (sigma 2.2); gain envelope "
                        "+-7, iid +-2.5, step p=0.4 (2..5 dB); warp "
                        "U(0.4,2.5); BG modes clean/busy/mixture "
                        "0.25/0.30/0.45; bg jitter clip +-4; drift "
                        "env +-4, colour +-3.5; ambient p=0.40")
    (DATA / "sim_p6dv_meta.json").write_text(json.dumps(meta, indent=1))
    print("diverse done", meta, flush=True)


if __name__ == "__main__":
    kind = sys.argv[1]
    assert kind in ("moresame", "diverse")
    (gen_moresame if kind == "moresame" else gen_diverse)()
