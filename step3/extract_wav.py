"""Step 3, Stage 0: extract kept-side clip waveforms from the VTUAD strict
archives into a local .npy cache (mono float32), one file per clip.

Fresh code for Step 3. Reads the zips directly; no archived code is used.
Only clips assigned to the family or i1 side in clip_sides.csv are
extracted. Run once before corpus generation.
"""
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import io

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
CACHE = HERE / "wav_cache"
ZIPS = {
    "vessel": ROOT / "data" / "vtuad" / "vtuad_strict_wav_vessel.zip",
    "background": ROOT / "data" / "vtuad" / "vtuad_strict_wav_background.zip",
}


def main():
    CACHE.mkdir(exist_ok=True)
    sides = pd.read_csv(HERE / "clip_sides.csv")
    kept = sides[sides["side"].isin(["family", "i1"])]
    entries = {}
    for kind, zp in ZIPS.items():
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.lower().endswith(".wav"):
                    entries[f"{kind}/{Path(n).name}"] = (zp, n)
    done = 0
    for key in kept["clip_key"]:
        out = CACHE / (key.replace("/", "_") + ".npy")
        if out.exists():
            done += 1
            continue
        zp, entry = entries[key]
        with zipfile.ZipFile(zp) as z:
            buf = z.read(entry)
        data, fs = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
        assert fs == 32000, (key, fs)
        np.save(out, data[:, 0])
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(kept)}", flush=True)
    print("cached", done, "clips")


if __name__ == "__main__":
    main()
