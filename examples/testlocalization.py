#!/usr/bin/env python3
"""
Standalone STORM-simulation + localization demo.

• Synthesises a series of noisy frames with random Gaussian emitters
• Localises using microEye (BandpassFilter + CV_BlobDetector + localize_frame)
• Saves a simple rendered reconstruction TIFF (sum of localisations)
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile as tif

from microEye.Filters import BandpassFilter
from microEye.fitting.fit import CV_BlobDetector, localize_frame
from microEye.fitting.results import FittingMethod


# ---------- simulation parameters ----------
N_FRAMES        = 50          # number of frames to simulate
FRAME_SHAPE     = (256, 256)   # pixels (y, x)
SPOTS_PER_FRAME = (5, 15)      # random uniform range
SIGMA_PSF       = 1.5          # Gaussian sigma (px)
PHOTON_COUNT    = (500, 2000)  # random uniform range
BG_LEVEL        = 100          # background offset
NOISE_STD       = 10           # Gaussian read noise
REL_THRESHOLD   = 0.4
ROI_SIZE        = 13
# --------------------------------------------


def gaussian_spot(shape, y0, x0, sigma, amplitude):
    """Return image with a single 2D Gaussian."""
    y = np.arange(shape[0])[:, None]
    x = np.arange(shape[1])[None, :]
    im = amplitude * np.exp(-((y - y0) ** 2 + (x - x0) ** 2) / (2 * sigma ** 2))
    return im


def simulate_frame(shape, n_spots):
    """Generate one noisy STORM frame."""
    frame = np.full(shape, BG_LEVEL, dtype=np.float32)
    ys = np.random.uniform(0, shape[0], n_spots)
    xs = np.random.uniform(0, shape[1], n_spots)
    amps = np.random.uniform(*PHOTON_COUNT, n_spots)
    for y0, x0, amp in zip(ys, xs, amps):
        frame += gaussian_spot(shape, y0, x0, SIGMA_PSF, amp)
    frame += np.random.normal(0, NOISE_STD, shape)               # read noise
    frame = np.random.poisson(np.clip(frame, 0, None)).astype(np.float32)  # shot noise
    return frame


def recon_frame(idx, frame, pre_filter, detector):
    """Localise one frame, return rendered binary image + parameters."""
    filt = pre_filter
    filtered = frame.copy()
    params_image, params, *_ = localize_frame(
        idx, frame, filtered, None,
        filt, detector, REL_THRESHOLD,
        np.array([SIGMA_PSF]), ROI_SIZE,
        FittingMethod._2D_Phasor_CPU,
    )
    render = np.zeros_like(frame, dtype=np.uint16)
    if params.size:
        y, x = params[:, 1].astype(np.int32), params[:, 0].astype(np.int32)
        render[y, x] = 1
    return render, params


def main():
    pre_filter = BandpassFilter()
    detector   = CV_BlobDetector()

    recon_sum = np.zeros(FRAME_SHAPE, dtype=np.uint32)
    all_params = []

    t0 = time.time()
    for i in range(N_FRAMES):
        n_spots = np.random.randint(*SPOTS_PER_FRAME)
        frame   = simulate_frame(FRAME_SHAPE, n_spots)
        render, params = recon_frame(i + 1, frame, pre_filter, detector)
        recon_sum += render
        if params.size:
            all_params.append(params)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{N_FRAMES} frames processed")

    dt = time.time() - t0
    print(f"Done in {dt:.2f} s – {N_FRAMES / dt:.1f} fps")

    # save reconstruction and parameters
    outdir = Path("storm_sim_output")
    outdir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tif.imwrite(outdir / f"reconstruction_{timestamp}.tif", recon_sum.astype(np.uint16))

    if all_params:
        np.save(outdir / f"localizations_{timestamp}.npy", np.vstack(all_params))

    print(f"Results saved to → {outdir}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
