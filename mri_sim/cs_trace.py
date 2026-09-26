"""
cs_trace.py -- an instrumented, step-by-step version of the compressed-sensing
reconstruction in `cs.py`, written for teaching rather than for production.

WHY THIS MODULE EXISTS
----------------------
`cs.ista_reconstruct` runs FISTA and hands back a finished image. That is the
right shape for a pipeline and the wrong shape for explaining the algorithm:
everything interesting happens *inside* the loop and is then thrown away.

This module runs the same iteration as a **generator**. It yields one record
per iteration, so a caller can render the reconstruction while it is still
converging, scrub back through the history afterwards, or open a single
iteration up and look at every one of its sub-steps.

Nothing here changes the algorithm. The private helpers are imported from
`cs.py` rather than reimplemented, precisely so the demo cannot drift away
from the code it is meant to explain -- see `verify_matches_reference` at the
bottom, which asserts the two produce the same image to floating-point
precision.

WHAT ONE ITERATION ACTUALLY DOES
--------------------------------
The loop alternates two operators. Spelled out as the demo presents them:

    1. PREDICT          K_pred = F x            "what would this guess have
                                                 measured?"
    2. RESIDUAL         r = M (y - K_pred)      "how wrong is it, on the
                                                 points we actually measured?"
    3. DATA CONSISTENCY x <- F^-1 (K_pred + r)  "overwrite those points with
                                                 the truth and go back to
                                                 image space"
    4. ANALYSE          c = W x                 "describe the image as
                                                 wavelet coefficients"
    5. SHRINK           c <- soft(c, t)         "delete everything small --
                                                 this is the sparsity prior"
    6. SYNTHESISE       x <- W^-1 c             "rebuild the image from what
                                                 survived"
    7. MOMENTUM         (FISTA only) overshoot along the direction of travel

Steps 1-3 are one gradient step on the data term; 4-6 are the proximal
operator of the L1 term. Step 7 is Nesterov acceleration.

USAGE
-----
    for step in cs_trace.iterate(acquired, mask, reference=truth):
        show(step.image, step.psnr)          # light: one image plus scalars

    # One iteration opened up, all seven sub-steps:
    step = cs_trace.single_step(acquired, mask, at=12, reference=truth)
    show(step.detail.residual_kspace, step.detail.killed, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

# Imported, not reimplemented: the demo must show the real algorithm.
from .cs import (
    DEFAULT_LEVEL,
    DEFAULT_WAVELET,
    _detail_scale,
    _ifft_complex,
    _threshold_details,
    _wavedec,
    _waverec,
    soft_threshold,
)
from .kspace import to_kspace

# Sub-step names, in the order the algorithm performs them. The page drives its
# walkthrough off this list, so the ordering lives here next to the code that
# implements it and the two cannot fall out of sync.
SUBSTEPS = [
    "predict",
    "residual",
    "data_consistency",
    "analyse",
    "shrink",
    "synthesise",
    "momentum",
]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class StepDetail:
    """
    Every intermediate array produced inside one iteration.

    Heavy (a dozen 256x256 arrays), so this is attached to at most one
    iteration per run -- see `iterate(detail_at=...)`. The arrays are
    magnitudes, already real and float32: the caller is going to display them,
    and the phase is not what any of these panels is showing.
    """

    input_image: np.ndarray        # |x| entering the iteration (after momentum)
    predicted_kspace: np.ndarray   # |F x|, what this guess predicts
    measured_kspace: np.ndarray    # |y|, what the scanner actually got
    residual_kspace: np.ndarray    # |M (y - F x)|, the disagreement
    consistent_image: np.ndarray   # |x| after the measured points are restored
    correction_image: np.ndarray   # |consistent - input|, what step 3 changed
    coeffs_before: np.ndarray      # |W x|, the coefficient pyramid
    coeffs_after: np.ndarray       # |soft(W x, t)|
    killed: np.ndarray             # bool: coefficients this step set to zero
    protected: np.ndarray          # bool: the exempt approximation band
    band_edges: list               # (row, col) pyramid division lines
    denoised_image: np.ndarray     # |W^-1 soft(...)|, the new estimate
    removed_image: np.ndarray      # |denoised - consistent|, what shrinking cost
    momentum_image: np.ndarray     # |x| handed to the next iteration
    momentum_weight: float         # the FISTA extrapolation coefficient
    residual_energy: float         # ||M (y - F x)||^2
    n_killed: int                  # coefficients zeroed this step
    n_detail: int                  # detail coefficients in total
    n_protected: int               # approximation coefficients (never shrunk)


@dataclass
class Step:
    """
    One iteration, as the caller sees it.

    Light by design: a single display image plus scalars, so a 150-iteration
    history costs ~40 MB and can live in a browser session. `detail` is
    populated only for the one iteration the caller asked about.
    """

    iteration: int                 # 1-based
    image: np.ndarray              # the new estimate, magnitude, float32
    threshold: float               # the absolute soft-threshold in use
    objective: float               # ||M F x - y||^2 + t ||W x||_1
    data_error: float              # ||M F x - y||^2 alone
    l1: float                      # ||W x||_1 alone, the sparsity term
    sparsity: float                # fraction of detail coefficients at exactly 0
    change: float                  # ||x_k - x_{k-1}|| / ||x_{k-1}||
    psnr: float | None = None
    ssim: float | None = None
    is_final_projection: bool = False   # the closing data-consistency step
    detail: StepDetail | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _mag(array: np.ndarray) -> np.ndarray:
    """Magnitude as float32 -- these arrays exist only to be looked at."""
    return np.abs(array).astype(np.float32)


def _band_edges(slices) -> list[tuple[int, int]]:
    """
    The pyramid division lines of a `coeffs_to_array` layout.

    `wavedec2` packs the coefficients into one array shaped like the image:
    the coarse approximation sits in the top-left corner, and each level's
    three detail bands tile the L-shaped region around it, doubling in size as
    the scale gets finer. Overlaying these lines is the difference between "a
    grey texture" and "a pyramid you can read".

    Taken from the `slices` bookkeeping rather than assumed to be powers of
    two, so this stays correct for image sizes that are not 2^n.
    """
    edges = []
    for level_slices in slices[1:]:
        rows, cols = level_slices["dd"]
        edges.append((int(rows.start), int(cols.start)))
    return edges


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    PSNR against the ground truth, computed the way `metrics.compute_psnr`
    does (data range taken from the reference) but inline and on magnitudes,
    because this runs once per iteration inside the loop.
    """
    data_range = float(reference.max() - reference.min())
    if data_range == 0:
        return float("inf")
    mse = float(np.mean((reference - estimate) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _sparsity(coeff_array: np.ndarray, slices) -> float:
    """
    Fraction of *detail* coefficients sitting at exactly zero.

    The headline number of the whole method: "this image is described by N% of
    its coefficients". The approximation band is excluded because it is never
    thresholded, so counting it would dilute the figure with coefficients that
    were never eligible to be zeroed.
    """
    approx_zeros = int(np.count_nonzero(coeff_array[slices[0]] == 0))
    n_detail = coeff_array.size - coeff_array[slices[0]].size
    if n_detail == 0:
        return 0.0
    detail_zeros = int(np.count_nonzero(coeff_array == 0)) - approx_zeros
    return detail_zeros / n_detail


# ---------------------------------------------------------------------------
# The instrumented iteration
# ---------------------------------------------------------------------------


def iterate(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    lambda_: float = 0.01,
    n_iter: int = 60,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    use_fista: bool = True,
    final_data_consistency: bool = True,
    reference: np.ndarray | None = None,
    detail_at: int | None = None,
    track_ssim: bool = False,
) -> Iterator[Step]:
    """
    Run FISTA, yielding one :class:`Step` per iteration as it completes.

    Identical in arithmetic to :func:`cs.ista_reconstruct`; the only additions
    are the records yielded along the way.

    Parameters
    ----------
    kspace_measured, mask, lambda_, n_iter, wavelet, level, use_fista,
    final_data_consistency, reference
        As in :func:`cs.ista_reconstruct`.
    detail_at : int or None
        1-based iteration number to attach a full :class:`StepDetail` to. One
        iteration only -- the detail is a dozen extra arrays, and a whole
        run's worth would not fit in a browser session.
    track_ssim : bool
        Also record SSIM per iteration. Costs ~4 ms an iteration (roughly a
        quarter of the iteration itself), so it is off by default and turned
        on only for the cached run that feeds the charts, not for live
        animation.

    Yields
    ------
    Step
        One per iteration, plus a closing one with `is_final_projection` set
        if `final_data_consistency` is true.
    """
    mask = mask.astype(bool)

    # The zero-filled reconstruction: the Stage 1 baseline, and the image this
    # algorithm has to beat. Kept complex -- see `cs._ifft_complex`.
    current = _ifft_complex(kspace_measured * mask)

    start_coeffs, slices = _wavedec(current, wavelet, level)
    threshold = lambda_ * _detail_scale(start_coeffs, slices)

    # FISTA bookkeeping. `momentum_image` is what actually enters the next
    # iteration; `current` is the honest iterate the metrics describe.
    momentum_image = current.copy()
    t_previous = 1.0

    measured_display = _mag(np.where(mask, kspace_measured, 0.0))

    for iteration in range(1, n_iter + 1):
        want_detail = detail_at is not None and iteration == detail_at
        previous = current
        # Copied only when a detail record is wanted: the loop overwrites
        # `momentum_image` below, and paying for a copy on every iteration to
        # serve the one iteration being inspected would be waste.
        entering = momentum_image.copy() if want_detail else None

        # --- 1. predict: what would the current guess have measured? --------
        predicted = to_kspace(momentum_image)

        # --- 2. residual: where it disagrees with the scanner ---------------
        # Zeroed off the mask: we have no opinion about unmeasured points, and
        # inventing one is exactly the mistake zero-filling makes.
        residual = np.where(mask, kspace_measured - predicted, 0.0)

        # --- 3. data consistency --------------------------------------------
        # `predicted + residual` equals `where(mask, measured, predicted)`: the
        # measured points are overwritten with the truth, the rest is left as
        # the guess had them. Because F is unitary this is a gradient step of
        # size 1 on ||M F x - y||^2, which is why no step size appears.
        consistent = _ifft_complex(predicted + residual)

        # --- 4. analyse: into the sparsifying basis -------------------------
        coeffs, slices = _wavedec(consistent, wavelet, level)

        # --- 5. shrink: the sparsity prior, on the detail bands only ---------
        shrunk = _threshold_details(coeffs, slices, threshold)

        # --- 6. synthesise: back to an image --------------------------------
        updated = _waverec(shrunk, slices, wavelet, level)

        # --- 7. momentum -----------------------------------------------------
        if use_fista:
            t_current = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_previous ** 2))
            weight = (t_previous - 1.0) / t_current
            momentum_image = updated + weight * (updated - current)
            t_previous = t_current
        else:
            weight = 0.0
            momentum_image = updated

        current = updated

        # --- diagnostics -----------------------------------------------------
        # `shrunk` *is* the coefficient vector of `current` (W is orthogonal
        # under periodization, and `updated = W^-1 shrunk`), so the sparsity
        # and L1 figures are read straight off it instead of paying for
        # another forward transform every iteration.
        data_error = float(
            np.sum(np.abs(np.where(mask, to_kspace(current) - kspace_measured, 0.0)) ** 2)
        )
        magnitude = _mag(current)
        l1 = float(np.sum(np.abs(shrunk)))

        step = Step(
            iteration=iteration,
            image=magnitude,
            threshold=threshold,
            objective=data_error + threshold * l1,
            data_error=data_error,
            l1=l1,
            sparsity=_sparsity(shrunk, slices),
            change=_relative_change(current, previous),
        )
        if reference is not None:
            step.psnr = _psnr(reference, magnitude)
            if track_ssim:
                from .metrics import compute_ssim

                step.ssim = compute_ssim(reference, magnitude)

        if want_detail:
            step.detail = _build_detail(
                entering=entering,
                predicted=predicted,
                measured_display=measured_display,
                residual=residual,
                consistent=consistent,
                coeffs=coeffs,
                shrunk=shrunk,
                slices=slices,
                updated=updated,
                momentum_out=momentum_image,
                weight=weight,
            )

        yield step

    if final_data_consistency:
        # The loop ends on a shrink, which perturbs the samples we actually
        # measured. Put them back: those values are the one part of the
        # reconstruction known to be true, and restoring them is nearly free.
        predicted = to_kspace(current)
        current = _ifft_complex(np.where(mask, kspace_measured, predicted))
        magnitude = _mag(current)

        final = Step(
            iteration=n_iter + 1,
            image=magnitude,
            threshold=threshold,
            # This step optimises neither term -- it is a projection onto the
            # measured data -- so the objective and sparsity of the *loop* do
            # not describe it. NaN keeps it out of those charts instead of
            # planting a misleading point at the end of them.
            objective=float("nan"),
            data_error=0.0,          # measured points now agree exactly
            l1=float("nan"),
            sparsity=float("nan"),
            change=0.0,
            is_final_projection=True,
        )
        if reference is not None:
            final.psnr = _psnr(reference, magnitude)
            if track_ssim:
                from .metrics import compute_ssim

                final.ssim = compute_ssim(reference, magnitude)
        yield final


def _relative_change(current: np.ndarray, previous: np.ndarray) -> float:
    """How far the estimate moved this iteration, as a fraction of its size."""
    norm = float(np.linalg.norm(previous))
    if norm == 0:
        return 0.0
    return float(np.linalg.norm(current - previous) / norm)


def _build_detail(
    entering: np.ndarray,
    predicted: np.ndarray,
    measured_display: np.ndarray,
    residual: np.ndarray,
    consistent: np.ndarray,
    coeffs: np.ndarray,
    shrunk: np.ndarray,
    slices,
    updated: np.ndarray,
    momentum_out: np.ndarray,
    weight: float,
) -> StepDetail:
    """Package one iteration's intermediates for display."""
    # A coefficient counts as killed if it was nonzero going in and is exactly
    # zero coming out. Comparing against zero is safe rather than sloppy here:
    # soft thresholding sets those coefficients to a literal 0.0.
    killed = (coeffs != 0) & (shrunk == 0)
    protected = np.zeros(coeffs.shape, dtype=bool)
    protected[slices[0]] = True

    n_protected = int(protected.sum())
    return StepDetail(
        input_image=_mag(entering),
        predicted_kspace=_mag(predicted),
        measured_kspace=measured_display,
        residual_kspace=_mag(residual),
        consistent_image=_mag(consistent),
        correction_image=_mag(consistent - entering),
        coeffs_before=_mag(coeffs),
        coeffs_after=_mag(shrunk),
        killed=killed,
        protected=protected,
        band_edges=_band_edges(slices),
        denoised_image=_mag(updated),
        removed_image=_mag(updated - consistent),
        momentum_image=_mag(momentum_out),
        momentum_weight=float(weight),
        residual_energy=float(np.sum(np.abs(residual) ** 2)),
        n_killed=int(killed.sum()),
        n_detail=int(coeffs.size - n_protected),
        n_protected=n_protected,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def run(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    **kwargs,
) -> list[Step]:
    """The whole trace as a list, for callers that want to scrub, not stream."""
    return list(iterate(kspace_measured, mask, **kwargs))


def single_step(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    at: int,
    **kwargs,
) -> Step:
    """
    Just iteration `at`, with its full :class:`StepDetail`.

    The iterations before it still have to run -- FISTA is sequential, there is
    no shortcut to iterate 30 -- but only the one record is kept, so the memory
    cost is one iteration's worth however deep `at` is.
    """
    kwargs.pop("detail_at", None)
    kwargs["n_iter"] = at
    kwargs["final_data_consistency"] = False
    for step in iterate(kspace_measured, mask, detail_at=at, **kwargs):
        if step.iteration == at:
            return step
    raise ValueError(f"iteration {at} was never reached")


def compress_by_keeping(
    image: np.ndarray,
    fraction: float,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> tuple[np.ndarray, np.ndarray, list, float]:
    """
    Keep only the largest `fraction` of the image's wavelet detail
    coefficients, throw the rest away, and rebuild the image.

    This demonstrates the *premise* the whole method rests on, using no
    undersampling and no iteration at all. Compressed sensing is only worth
    attempting because medical images are compressible: discard 95% of the
    wavelet coefficients of a fully known brain image and it still looks like
    the same brain. That claim should be shown before it is relied on --
    otherwise the sparsity prior looks like an arbitrary trick rather than a
    property of the data.

    (Note the difference from the reconstruction: here we can pick the largest
    coefficients because the true image is in hand. During a real scan it is
    not, which is why the algorithm has to *search* for a sparse image
    consistent with the measurements.)

    Returns
    -------
    (reconstructed, kept_mask, band_edges, threshold)
        `kept_mask` marks the coefficients that survived, ready for
        `threshold_map`; `threshold` is the magnitude cut that `fraction`
        worked out to.
    """
    coeffs, slices = _wavedec(np.asarray(image, dtype=np.float64), wavelet, level)

    # The approximation band is kept in full, exactly as the reconstruction
    # exempts it, so this demo and the algorithm are measuring the same thing.
    is_detail = np.ones(coeffs.shape, dtype=bool)
    is_detail[slices[0]] = False

    detail_values = np.abs(coeffs[is_detail])
    n_keep = int(round(fraction * detail_values.size))

    if n_keep <= 0:
        cut = float(detail_values.max()) + 1.0    # keep nothing
    elif n_keep >= detail_values.size:
        cut = 0.0                                  # keep everything
    else:
        # The n_keep-th largest magnitude: everything at least this big stays.
        cut = float(np.partition(detail_values, -n_keep)[-n_keep])

    kept = (np.abs(coeffs) >= cut) & is_detail
    trimmed = np.where(kept | ~is_detail, coeffs, 0.0)
    return _waverec(trimmed, slices, wavelet, level), kept, _band_edges(slices), cut


def soft_threshold_curve(threshold: float, extent: float = 1.0, n: int = 241):
    """
    The shrinkage operator as a plottable curve, for the input/output graph
    that makes soft thresholding click.

    Returns (input values, soft-thresholded values) so a chart can show the
    dead zone of width 2t around the origin and the unit-slope lines outside
    it, each offset toward zero by exactly `threshold`.
    """
    values = np.linspace(-extent, extent, n)
    return values, soft_threshold(values, threshold)


def verify_matches_reference(
    kspace_measured: np.ndarray,
    mask: np.ndarray,
    **kwargs,
) -> float:
    """
    Largest absolute disagreement between this trace's final image and
    :func:`cs.ista_reconstruct` on the same inputs.

    A demo that quietly diverges from the algorithm it illustrates is worse
    than no demo, so this is checked rather than assumed.

    The comparison is made in float32, and returning exactly 0.0 is the
    expected result. `Step.image` is stored as float32 to keep a long history
    affordable, so comparing in float64 would bottom out at ~1e-7 -- the
    rounding of that store, not a difference in the arithmetic. Casting the
    reference the same way removes that floor, which turns a vague "close
    enough" into the real claim: the two run identical arithmetic and produce
    bit-identical images.
    """
    from .cs import ista_reconstruct

    expected, _ = ista_reconstruct(kspace_measured, mask, **kwargs)
    steps = run(kspace_measured, mask, **kwargs)
    return float(np.max(np.abs(expected.astype(np.float32) - steps[-1].image)))
