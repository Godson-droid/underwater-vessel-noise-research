"""Independent reimplementation of the one-minute x 25-band reference
conversion, written for Stage 2 provenance review (18 November 2026).

This file was written from the published mathematical definition of the
conversion, as documented in the programme's reports and in
release_docs_stage1/DATA_SOURCES.md section 3:

  * the recording is divided into non-overlapping 2-second segments;
  * each segment has its mean removed and is tapered with a Hann window;
  * the one-sided periodogram of each segment is computed with density
    scaling (power spectral density, units of signal^2 per Hz);
  * the density is integrated over each of 25 deci-decade bands whose
    centre frequencies are 10^(1.5 + 0.1 k) Hz for k = 0..24 and whose
    edges lie a factor 10^0.05 either side of the centre;
  * the 30 segments of each complete minute are averaged in linear
    power units, giving one 25-band power vector per minute;
  * a band whose upper edge lies above the Nyquist frequency receives
    no estimate (NaN), matching the documented frequency limits.

This is an independent implementation written for provenance purposes.
It does NOT replace the historical implementation in
code/build_passage_pool.py (which remains untouched as the research
record), and it must not be substituted for it without separate review
and approval. Its only purpose is to be compared against the historical
implementation by stage2_work/equivalence_test.py.

Note on scope: this function returns LINEAR band power per minute,
exactly the quantity the historical function returns. The conversion of
clip power to relative dB (normalising by the clip's median power) is a
separate step performed by the historical caller, not by this function.
"""

import numpy as np

# Definition constants (from the documented specification).
SEGMENT_SECONDS = 2          # length of one analysis segment
SEGMENTS_PER_MINUTE = 30     # 30 x 2 s = one complete minute
N_BANDS = 25                 # deci-decade bands per minute
CENTRE_EXPONENT_BASE = 1.5   # first band centre is 10^1.5 Hz
CENTRE_EXPONENT_STEP = 0.1   # centres step by 0.1 decade (deci-decade)
EDGE_EXPONENT = 0.05         # band edges at centre x// 10^0.05


def band_centres_hz():
    """The 25 deci-decade band centre frequencies, in Hz."""
    exponents = CENTRE_EXPONENT_BASE + CENTRE_EXPONENT_STEP * np.arange(N_BANDS)
    return 10.0 ** exponents


def band_edges_hz():
    """(low, high) edge frequencies for each band, in Hz."""
    centres = band_centres_hz()
    factor = 10.0 ** EDGE_EXPONENT
    return centres / factor, centres * factor


def _periodic_hann(length):
    """Hann taper of the given length, w[n] = 0.5 - 0.5 cos(2 pi n / N).

    This is the standard periodic form of the Hann window used for
    spectral analysis of a single frame (the frame is treated as one
    period of a periodic signal).
    """
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


def _onesided_density_periodogram(frames, sample_rate_hz):
    """One-sided power spectral density of each row of `frames`.

    frames: (n_frames, n_samples) float array, already demeaned.
    Returns (frequencies_hz, psd) with psd of shape
    (n_frames, n_samples // 2 + 1), in signal^2 per Hz, using density
    scaling: |DFT(w x)|^2 / (fs * sum(w^2)), with every interior bin of
    the one-sided spectrum doubled (all bins except DC and, for even
    frame length, the Nyquist bin).
    """
    n_samples = frames.shape[1]
    taper = _periodic_hann(n_samples)
    spectrum = np.fft.rfft(frames * taper, axis=1)
    density = (spectrum.real ** 2 + spectrum.imag ** 2) / (
        sample_rate_hz * float(np.dot(taper, taper))
    )
    if n_samples % 2 == 0:
        density[:, 1:-1] *= 2.0   # keep DC and Nyquist single
    else:
        density[:, 1:] *= 2.0     # no exact Nyquist bin for odd length
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    return frequencies, density


def minute_band_power_clean(samples, sample_rate_hz):
    """Per-minute 25-band linear power of a mono recording.

    samples: one-dimensional array of finite samples.
    sample_rate_hz: the recording's integer sample rate, in Hz.
    Returns an array of shape (n_complete_minutes, 25) of linear band
    powers; bands whose upper edge exceeds the Nyquist frequency are
    NaN. Raises ValueError on unusable input.
    """
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("Use one channel with finite samples.")
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0 or not fs.is_integer():
        raise ValueError("Use the actual integer sample rate from the recording.")
    fs = int(fs)

    segment_len = SEGMENT_SECONDS * fs
    minute_len = SEGMENTS_PER_MINUTE * segment_len
    n_minutes = signal.size // minute_len
    if n_minutes == 0:
        raise ValueError("At least one complete minute is needed.")

    # Group the usable signal as (minutes, segments per minute, samples).
    grouped = signal[: n_minutes * minute_len].reshape(
        n_minutes, SEGMENTS_PER_MINUTE, segment_len
    )
    demeaned = grouped - grouped.mean(axis=2, keepdims=True)
    flat = demeaned.reshape(n_minutes * SEGMENTS_PER_MINUTE, segment_len)
    frequencies, psd = _onesided_density_periodogram(flat, fs)

    bin_width = frequencies[1] - frequencies[0]
    lows, highs = band_edges_hz()
    per_segment = np.full((flat.shape[0], N_BANDS), np.nan)
    for band in range(N_BANDS):
        in_band = (frequencies >= lows[band]) & (frequencies < highs[band])
        if highs[band] <= fs / 2.0 and in_band.any():
            per_segment[:, band] = psd[:, in_band].sum(axis=1) * bin_width

    per_minute = per_segment.reshape(
        n_minutes, SEGMENTS_PER_MINUTE, N_BANDS
    ).mean(axis=1)
    return per_minute
