"""
denoise.py -- Stage 2: removing scanner noise from a reconstructed image.

WHAT WE ARE UNDOING
-------------------
`noise.py` adds complex white Gaussian noise to the measured k-space samples.
After the inverse FFT and magnitude, that noise is spread evenly over the whole
image as fine grain (Rician in the magnitude, see noise.py). This module tries
to take it back out, working on the finished image alone.

THE IDEA: NOISE IS SMALL AND EVERYWHERE, STRUCTURE IS LARGE AND SPARSE
----------------------------------------------------------------------
Take a wavelet transform of the image. The anatomy -- edges, boundaries,
smooth regions -- is described by a *few large* wavelet coefficients. White
noise has no structure, so it lands as *many small* coefficients spread evenly
across every band. That gives a simple rule:

    shrink every detail coefficient toward zero by a threshold T,
    setting to zero anything smaller than T  (soft thresholding)

Large coefficients survive (minus T); the sea of small ones vanishes. This is
the same soft-thresholding step that `cs.py` runs every iteration -- here it is
run once, on the finished image, with a threshold chosen for the noise rather
than for missing k-space.

We use BayesShrink, which picks a separate threshold for each wavelet band:
T = sigma^2 / sigma_signal, where sigma_signal is how much real signal that
band carries. Bands full of structure get a gentle threshold; bands that are
mostly noise get a harsh one.

THE DENOISER DOES NOT GET TOLD THE NOISE LEVEL
----------------------------------------------
We know exactly how much noise we added, but a real scanner never does. So the
noise level is *estimated from the noisy image itself*, from the finest
diagonal wavelet band -- which is almost pure noise in a natural image -- using
the robust median-absolute-deviation estimator: sigma = median(|d|) / 0.6745.

THE TRADE-OFF
-------------
Fine anatomical texture also lives in small coefficients. Any threshold large
enough to remove the noise also removes some real detail, so a denoised image
is always a little smoother than the truth. The `strength` knob exposes that
trade-off, and the residual (noisy - denoised) shows what was removed: pure
grain means only noise went; visible anatomy means detail went too.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma

from mri_sim.cs import DEFAULT_LEVEL, DEFAULT_WAVELET


def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    Estimate the image-domain noise standard deviation from the image alone.

    Median-absolute-deviation of the finest diagonal wavelet band (Donoho &
    Johnstone), as implemented by scikit-image.

    Caveat: the magnitude image's noise is Rician, not Gaussian. In bright
    regions the two are almost identical; in the near-black background the
    Rician noise is smaller and biased positive. The estimate is therefore a
    little low on images with a lot of background -- good enough to set a
    threshold, not a precision measurement.
    """
    return float(estimate_sigma(image, channel_axis=None))


def wavelet_denoise(
    image: np.ndarray,
    strength: float = 1.0,
    sigma: float | None = None,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> np.ndarray:
    """
    Wavelet soft-thresholding denoiser (BayesShrink).

    Parameters
    ----------
    image : 2-D float array
        The noisy magnitude reconstruction.
    strength : float
        Multiplies the estimated noise level before it sets the thresholds.
        1.0 is the textbook setting; below 1 keeps more detail and more
        noise, above 1 removes more noise and more detail. 0 returns the
        image unchanged.
    sigma : float or None
        Noise level to assume. None (the default, and the honest choice)
        estimates it from `image` with :func:`estimate_noise_sigma`.
    wavelet, level :
        Same defaults as the compressed-sensing reconstruction, so the two
        can be compared like for like.

    Returns
    -------
    2-D float array, same shape, non-negative.
    """
    if strength <= 0.0:
        return image.copy()

    if sigma is None:
        sigma = estimate_noise_sigma(image)
    if sigma <= 0.0:
        return image.copy()

    denoised = denoise_wavelet(
        image,
        sigma=sigma * strength,
        wavelet=wavelet,
        wavelet_levels=level,
        mode="soft",
        method="BayesShrink",
        channel_axis=None,
    )
    # A magnitude image cannot be negative; thresholding can undershoot
    # slightly near sharp edges.
    return np.clip(denoised, 0.0, None)


def residual(noisy: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    """
    What the denoiser removed: noisy - denoised.

    The diagnostic to look at. If it is featureless grain, only noise was
    removed. If you can see anatomy in it, real detail was removed too.
    """
    return noisy - denoised
