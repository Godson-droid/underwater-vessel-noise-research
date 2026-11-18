"""Phase 5 conversion variants, second batch (same machinery as
convert_variants.py, added after the assessment's priority order):

  magmean — magnitude aggregation (assessment priority 1, power vs
            magnitude): per minute and band, the value is
            (mean over 2 s segments of sqrt(segment band power))^2,
            instead of the mean of segment powers.
  bandext — band boundary definitions (priority 4): the notebook's
            deci-decade edges for interior bands, but band 0
            integrates from the lowest FFT bin and band 24 up to
            Nyquist, instead of centre x 10^(+-0.05) at the edges.
            (Exact 2^(1/3) third-octave edges differ from the
            deci-decade edges by 0.04% in width — arithmetically
            negligible, recorded in RESULTS_conversion.md, not run.)

Usage: python convert_variants2.py [magmean] [bandext]
"""
import sys
import zipfile
import numpy as np
import convert_variants as cv
from common import DATA

_orig_aggregate = cv.aggregate


def aggregate2(seg_power, how):
    if how == "magmean":
        return np.mean(np.sqrt(np.maximum(seg_power, 0.0)), axis=0) ** 2
    return _orig_aggregate(seg_power, how)


def run(tag, cfg, edges=None):
    cv.aggregate = aggregate2
    if edges is not None:
        orig_edges = cv.EDGES
        cv.EDGES = edges
    try:
        clips = cv.clip_list()
        print(f"[{tag}] {len(clips)} clips", flush=True)
        start, rel_list, names, kinds, fss = cv._resume(tag, clips)
        rel_list, names = list(rel_list), list(names)
        kinds, fss = list(kinds), list(fss)
        for i, (zp, entry, kind, disp) in enumerate(clips):
            if i < start:
                continue
            with zipfile.ZipFile(zp) as z:
                buf = z.read(entry)
            powers, fs = cv.convert_clip(buf, cfg)
            assert powers is not None, f"clip {disp} failed under {tag}"
            if edges is not None:
                # bandext hard-codes the 32 kHz Nyquist in band 24's edge
                assert fs == 32000, f"bandext assumes fs=32000, got {fs}"
            rel_list.append(cv.to_rel_db(powers[cfg[4][0]]))
            names.append(disp)
            kinds.append(kind)
            fss.append(fs)
            if (i + 1) % 50 == 0 or i + 1 == len(clips):
                cv._save_partial(tag, rel_list, names, kinds, fss)
                print(f"[{tag}] {i + 1}/{len(clips)} (checkpointed)",
                      flush=True)
        print(f"[{tag}] wrote passages_conv_{tag}.npz ({len(names)} clips)",
              flush=True)
    finally:
        cv.aggregate = _orig_aggregate
        if edges is not None:
            cv.EDGES = orig_edges


if __name__ == "__main__":
    want = sys.argv[1:] or ["magmean", "bandext"]
    if "magmean" in want:
        run("magmean", ("hann", "constant", 2, 2, ("magmean",)))
    if "bandext" in want:
        edges = list(cv.EDGES)
        edges[0] = (0.0, edges[0][1])
        edges[24] = (edges[24][0], 16000.0)  # 32 kHz Nyquist (all clips)
        run("bandext", ("hann", "constant", 2, 2, ("mean",)), edges=edges)
    print("ALL DONE", flush=True)
