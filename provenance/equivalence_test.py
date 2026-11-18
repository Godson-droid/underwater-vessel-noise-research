"""Equivalence test: historical minute_band_power vs the Stage 2 clean
reimplementation (clean_conversion.minute_band_power_clean).

Method and safety notes
-----------------------
* The historical function is loaded IN ISOLATION: its source is parsed
  with `ast`, the single `minute_band_power` function definition is
  extracted, and only that function is executed in a fresh namespace
  providing `np` and scipy's `periodogram`. The module
  code/build_passage_pool.py is never imported, so none of its
  module-level code runs (it was assessed as inert -- imports, pinned
  constants and function definitions, with main() guarded -- but
  isolation removes any doubt, and also avoids writing bytecode caches
  into the research tree). The archive-reassembly function
  strict_vessel_zip() and main() cannot run under this harness.
* All test signals are SYNTHETIC, generated in memory below. No
  competition data, no VTUAD audio and no extracted caches are used.
* Nothing is written by this script; its stdout is the report (the
  caller redirects it to equivalence_results.txt).

Run with the project environment:  .venv/bin/python equivalence_test.py
"""

import ast
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import periodogram

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from clean_conversion import minute_band_power_clean  # noqa: E402


def load_historical():
    src = (PROJECT / "code" / "build_passage_pool.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "minute_band_power":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"np": np, "periodogram": periodogram}
            exec(compile(module, "build_passage_pool.py:minute_band_power", "exec"),
                 namespace)
            return namespace["minute_band_power"]
    raise RuntimeError("minute_band_power not found in build_passage_pool.py")


# ---------------------------------------------------------------- signals
def tone(freq, fs, seconds, amp=1.0):
    t = np.arange(int(fs * seconds), dtype=np.float64) / fs
    return amp * np.sin(2.0 * np.pi * freq * t)


def chirp(f0, f1, fs, seconds, amp=1.0):
    t = np.arange(int(fs * seconds), dtype=np.float64) / fs
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2.0 * seconds))
    return amp * np.sin(phase)


def build_signals():
    signals = []
    fs = 32000
    centre = lambda k: 10.0 ** (1.5 + 0.1 * k)          # noqa: E731
    edge_10_11 = centre(11) / 10.0 ** 0.05              # shared edge, bands 10/11
    rng = np.random.default_rng(20261118)
    noise60 = rng.standard_normal(60 * fs)
    rng2 = np.random.default_rng(99)
    rng3 = np.random.default_rng(7)
    signals.append(("silence", np.zeros(60 * fs), fs))
    signals.append(("tone_band0_centre", tone(centre(0), fs, 60), fs))
    signals.append(("tone_band12_centre", tone(centre(12), fs, 60), fs))
    signals.append(("tone_band24_centre", tone(centre(24), fs, 60), fs))
    signals.append(("tone_band_edge_10_11", tone(edge_10_11, fs, 60), fs))
    signals.append(("tone_below_band0_10Hz", tone(10.0, fs, 60), fs))
    signals.append(("white_noise_seed20261118", noise60, fs))
    signals.append(("chirp_20_to_12000Hz", chirp(20.0, 12000.0, fs, 60), fs))
    mix = (0.7 * tone(centre(0), fs, 60) + 0.7 * tone(centre(24), fs, 60)
           + 0.3 * rng3.standard_normal(60 * fs))
    signals.append(("mix_band0_band24_noise", mix, fs))
    signals.append(("white_noise_2minutes", rng2.standard_normal(120 * fs), fs))
    fs16 = 16000
    rng4 = np.random.default_rng(20261118)
    signals.append(("fs16000_white_noise", rng4.standard_normal(60 * fs16), fs16))
    signals.append(("fs16000_tone_band24_centre", tone(centre(24), fs16, 60), fs16))
    return signals


# ---------------------------------------------------------------- compare
def compare(name, samples, fs, hist_fn):
    hist = hist_fn(samples, fs)
    clean = minute_band_power_clean(samples, fs)
    assert hist.shape == clean.shape, (name, hist.shape, clean.shape)
    hist_nan = np.isnan(hist)
    clean_nan = np.isnan(clean)
    nan_mismatch = int(np.count_nonzero(hist_nan != clean_nan))
    finite = ~hist_nan & ~clean_nan
    diff = np.abs(hist - clean)[finite]
    scale = np.max(np.abs(hist[finite])) if finite.any() else 0.0
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    max_rel_to_scale = max_abs / scale if scale > 0 else 0.0
    # edge bands, per signal (over minutes)
    edge = {}
    for band in (0, 24):
        hb = hist[:, band]
        cb = clean[:, band]
        both_nan = bool(np.isnan(hb).all() and np.isnan(cb).all())
        if both_nan:
            edge[band] = "both NaN (band above Nyquist)"
        else:
            d = np.abs(hb - cb)
            edge[band] = f"max abs dev {np.nanmax(d):.3e}"
    print(f"signal: {name}  (fs={fs}, minutes={hist.shape[0]})")
    print(f"  shape {hist.shape} == {clean.shape}; NaN positions: "
          f"historical {int(hist_nan.sum())}, clean {int(clean_nan.sum())}, "
          f"mismatched {nan_mismatch}")
    print(f"  max abs deviation  : {max_abs:.3e}")
    print(f"  mean abs deviation : {mean_abs:.3e}")
    print(f"  max abs dev / signal peak band power: {max_rel_to_scale:.3e}")
    print(f"  band 0 : {edge[0]}")
    print(f"  band 24: {edge[24]}")
    return max_abs, mean_abs, nan_mismatch


def main():
    print("EQUIVALENCE TEST - historical minute_band_power vs clean reimplementation")
    print(f"numpy {np.__version__}, scipy {scipy.__version__}")
    print("Historical function loaded by AST extraction of the single function")
    print("from code/build_passage_pool.py (module never imported; no module-level")
    print("code, reassembly or main() executed). Signals are synthetic, in memory.")
    print()
    hist_fn = load_historical()
    worst_abs, worst_mean, total_nan_mismatch = 0.0, 0.0, 0
    for name, samples, fs in build_signals():
        a, m, nm = compare(name, samples, fs, hist_fn)
        worst_abs = max(worst_abs, a)
        worst_mean = max(worst_mean, m)
        total_nan_mismatch += nm
        print()
    print("SUMMARY")
    print(f"  worst max abs deviation across all signals : {worst_abs:.3e}")
    print(f"  worst mean abs deviation across all signals: {worst_mean:.3e}")
    print(f"  total NaN-position mismatches              : {total_nan_mismatch}")


if __name__ == "__main__":
    main()
