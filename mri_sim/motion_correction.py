"""
motion_correction.py -- undoing patient motion, with and without being told it.

MOTION DESTROYS NOTHING; IT ONLY SCRAMBLES PHASE
-------------------------------------------------
`motion.apply_motion` multiplies each k-space row by a unit-magnitude phase
ramp, exp(-j*2*pi*(ky*dy/ny + kx*dx/nx)). Multiplying by the conjugate ramp,
exp(+j*2*pi*(...)), puts every sample back exactly. No information is lost,
no noise is amplified (|ramp| = 1), and masking does not matter: an unmeasured
sample was zero before correction and is zero after it.

So correction is trivial *if you know the displacements*. The whole problem is
knowing them. This module offers two answers:

1. ORACLE CORRECTION -- `undo_motion` / `undo_motion_radial`.
   Uses the true displacements. On a real scanner these come from navigator
   echoes (short extra acquisitions that track position) or an optical
   camera. Here they come from the simulator, which is exactly the "cheat" a
   navigator gives you. Restores the image to machine precision.

2. AUTOFOCUS -- `autofocus`.
   Uses nothing but the corrupted data. Try candidate motions, undo each one,
   and keep the one that gives the *sharpest* image. Sharpness is scored by
   gradient entropy (below). The search assumes the motion *type* is known
   (jerk, drift or periodic) and estimates only its few parameters -- a
   reasonable assumption for breathing, which is always periodic. A fully
   blind search (one free shift per block of rows) was tried and gets stuck in
   local minima; see MOTION.md.

WHY GRADIENT ENTROPY MEASURES SHARPNESS
---------------------------------------
A sharp image has a few strong edges and flat regions elsewhere: its gradient
magnitudes are concentrated in a few pixels, so their normalised distribution
has LOW entropy. Ghosting duplicates every edge and blur smears it, spreading
the gradient energy over many more pixels: HIGHER entropy. Minimising it pulls
the ghosts back onto the object. It is scale-invariant (the gradients are
normalised to sum to 1), so image brightness does not matter.

ONE AMBIGUITY THAT CANNOT BE RESOLVED
-------------------------------------
A shift applied equally to the whole scan is harmless and invisible (the
shift theorem again), so only motion *relative* to a reference moment can be
recovered. Each model below is anchored at displacement 0 at its start, so a
corrected image may still sit a few pixels away from the ground truth as a
whole. That is not an error in the correction; it is physics.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import motion
from .kspace import from_kspace

Displacements = Sequence[tuple[float, float]]


# ---------------------------------------------------------------------------
# 1. Exact inverses of the two corruption functions in motion.py
# ---------------------------------------------------------------------------


def undo_motion(kspace_centered: np.ndarray, displacements: Displacements) -> np.ndarray:
    """
    Exact inverse of `motion.apply_motion`: multiply row i by the conjugate
    of the phase ramp that displacement i stamped on it.

    Vectorised (one ramp matrix, no Python loop over rows), because
    `autofocus` calls this hundreds of times.
    """
    ny, nx = kspace_centered.shape
    if len(displacements) != ny:
        raise ValueError(
            f"need one displacement per row: expected {ny}, got {len(displacements)}"
        )
    shifts = np.asarray(displacements, dtype=np.float64)
    ky = (np.arange(ny) - ny // 2)[:, None]
    dy = shifts[:, 0:1]
    if not np.any(shifts[:, 1]):
        # y-only motion (every model in motion.py): one complex scalar per
        # row, broadcast across it -- ~7x faster than the full ramp matrix.
        return kspace_centered * np.exp(2j * np.pi * ky * dy / ny)
    kx = (np.arange(nx) - nx // 2)[None, :]
    dx = shifts[:, 1:2]
    ramp = np.exp(2j * np.pi * (ky * dy / ny + kx * dx / nx))
    return kspace_centered * ramp


def undo_motion_radial(
    kspace_centered: np.ndarray,
    spoke_index: np.ndarray,
    spoke_displacements: Displacements,
) -> np.ndarray:
    """
    Exact inverse of `motion.apply_motion_radial`: each point gets the
    conjugate ramp of the spoke that measured it, using its own (ky, kx).
    Points on no spoke (`spoke_index == 0`) are left untouched, as in the
    forward function.
    """
    ny, nx = kspace_centered.shape
    n_spokes = int(spoke_index.max())
    if len(spoke_displacements) != n_spokes:
        raise ValueError(
            f"need one displacement per spoke: expected {n_spokes}, "
            f"got {len(spoke_displacements)}"
        )
    # Row 0 of the lookup table is the "no spoke" entry: zero displacement.
    table = np.vstack([[0.0, 0.0], np.asarray(spoke_displacements, dtype=np.float64)])
    per_point = table[spoke_index]
    ky = (np.arange(ny) - ny // 2)[:, None]
    kx = (np.arange(nx) - nx // 2)[None, :]
    ramp = np.exp(2j * np.pi * (ky * per_point[..., 0] / ny + kx * per_point[..., 1] / nx))
    return kspace_centered * ramp


# ---------------------------------------------------------------------------
# 2. The sharpness score
# ---------------------------------------------------------------------------


def gradient_entropy(image: np.ndarray) -> float:
    """
    Entropy of the normalised gradient magnitude. Lower means sharper.

    Uses the gradient along y (axis 0) and x (axis 1). Motion here is along
    y, so the y term does most of the work, but including x keeps the score
    honest about blur in general.
    """
    # Single precision: this is a ranking score, and float32 logs are ~2x
    # faster in the autofocus inner loop.
    image = image.astype(np.float32, copy=False)
    gy = np.abs(np.diff(image, axis=0))
    gx = np.abs(np.diff(image, axis=1))
    g = np.concatenate([gy.ravel(), gx.ravel()])
    total = float(g.sum(dtype=np.float64))
    if total <= 0.0:
        return 0.0
    g = g[g > 0]
    # H = -sum(p log p) with p = g/S, rearranged to avoid a second array.
    # The sum itself in float64: the finest autofocus steps compare scores
    # that differ in the 5th-6th significant digit.
    weighted = np.dot(g.astype(np.float64), np.log(g).astype(np.float64))
    return float(np.log(total) - weighted / total)


# ---------------------------------------------------------------------------
# 3. Autofocus: estimate the motion by maximising sharpness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Param:
    """One searched parameter: its range, first grid step and finest step."""
    name: str
    low: float
    high: float
    step: float
    min_step: float
    integer: bool = False


@dataclass
class AutofocusResult:
    """What `autofocus` found."""
    model: str
    params: dict                        # e.g. {"amp": 8.0, "at": 128}
    displacements: list[tuple[float, float]]
    score: float                        # gradient entropy of the corrected image
    initial_score: float                # gradient entropy before correction
    n_evaluations: int                  # how many candidate images were built


def _displacements_for(model: str, n: int, params: dict) -> list[tuple[float, float]]:
    if model == "sudden_jerk":
        return motion.sudden_jerk(n, params["amp"], int(params["at"]))
    if model == "slow_drift":
        return motion.slow_drift(n, params["amp"])
    if model == "periodic":
        return motion.periodic(n, params["amp"], params["cycles"])
    raise ValueError(f"unknown motion model: {model}")


def _search_space(model: str, n: int, max_amp: float) -> list[_Param]:
    amp = _Param("amp", -max_amp, max_amp, step=2.0, min_step=0.125)
    if model == "sudden_jerk":
        return [amp, _Param("at", 1, n - 1, step=max(n // 16, 1), min_step=1, integer=True)]
    if model == "slow_drift":
        return [amp]
    if model == "periodic":
        return [amp, _Param("cycles", 0.5, 20.0, step=1.0, min_step=1 / 64)]
    raise ValueError(f"unknown motion model: {model}")


def _grid(param: _Param, centre: float | None, half_width: float | None, step: float):
    if centre is None:
        low, high = param.low, param.high
    else:
        low = max(param.low, centre - half_width)
        high = min(param.high, centre + half_width)
    values = np.arange(low, high + step / 2, step)
    if param.integer:
        values = np.unique(np.clip(np.round(values), param.low, param.high)).astype(int)
    return values.tolist()


class _Scorer:
    """Scores candidate parameters and remembers the best one seen anywhere."""

    def __init__(self, model, n_events, acquired, undo):
        self.model, self.n_events = model, n_events
        self.acquired, self.undo = acquired, undo
        self.best_params: dict | None = None
        self.best_score = np.inf
        self.evaluations = 0

    def __call__(self, params: dict) -> float:
        displacements = _displacements_for(self.model, self.n_events, params)
        s = gradient_entropy(from_kspace(self.undo(self.acquired, displacements)))
        self.evaluations += 1
        if s < self.best_score:
            self.best_score, self.best_params = s, dict(params)
        return s


def _coarse_to_fine(
    space: list[_Param],
    scorer: _Scorer,
    start: dict | None = None,
    start_half: dict | None = None,
    start_step: dict | None = None,
) -> None:
    """
    Grid search that zooms in: each round grids +-half_width around the best
    point so far, then the next round uses half_width = this step and
    step = this step / 4, until every parameter has been searched at its
    finest step. Without `start`, the first round covers each full range.
    """
    step = dict(start_step) if start_step else {p.name: float(p.step) for p in space}
    half = dict(start_half) if start_half else None
    centre = dict(start) if start else None

    while True:
        axes = [
            _grid(p, None if centre is None else centre[p.name],
                  None if half is None else half[p.name], step[p.name])
            for p in space
        ]
        for values in itertools.product(*axes):
            scorer(dict(zip((p.name for p in space), values)))

        if all(step[p.name] <= p.min_step for p in space):
            return
        centre = scorer.best_params
        half = {p.name: step[p.name] for p in space}
        step = {p.name: max(step[p.name] / 4, p.min_step) for p in space}


def _periodic_seed(scorer: _Scorer, max_amp: float) -> dict:
    """
    Find a starting point for periodic motion that a coarse grid cannot.

    The score has a NARROW dip in `cycles` -- about +-0.25 cycles wide -- but
    a wide, smooth one in `amp`. A coarse grid over both steps straight over
    the cycles dip. So:

    1. Scan `cycles` finely (1/8 cycle) with a few PROBE amplitudes. Even a
       wrong amplitude of the right sign partly undoes the motion, so the dip
       shows at the true cycle count without knowing the true amplitude.
       Two sizes per sign: a small probe for gentle motion (overshooting by
       2x undoes nothing), a larger one so big motion still shows a dip
       above the noise.
    2. At the best cycle count, scan amplitude over its whole range.

    Known limit: the dip narrows as amplitude grows (roughly 1/(2*pi*amp)
    cycles wide), so very large, fast motion -- ~25 px at ~20 cycles -- can
    fall between the 1/8-cycle grid points. Tested reliable up to 20 px.
    """
    probes = [a for a in (-8.0, -2.0, 2.0, 8.0) if abs(a) <= max_amp]
    for cycles in np.arange(0.5, 20.0 + 1e-9, 0.125):
        for amp in probes:
            scorer({"amp": amp, "cycles": float(cycles)})
    cycles = scorer.best_params["cycles"]
    for amp in np.arange(-max_amp, max_amp + 1e-9, 1.0):
        scorer({"amp": float(amp), "cycles": cycles})
    return dict(scorer.best_params)


def autofocus(
    acquired_kspace: np.ndarray,
    model: str,
    n_events: int,
    undo: Callable[[np.ndarray, Displacements], np.ndarray],
    max_amp: float = 25.0,
) -> AutofocusResult:
    """
    Estimate the motion parameters that make the corrected image sharpest.

    Parameters
    ----------
    acquired_kspace : complex array
        What the scanner recorded: motion-corrupted, and possibly masked and
        noisy. Nothing else is used -- not the true motion, not the clean
        image.
    model : "sudden_jerk", "slow_drift" or "periodic"
        The motion type to assume. Only its parameters are estimated.
    n_events : int
        Number of acquisition events: `ny` rows for Cartesian, the spoke
        count for radial.
    undo : callable(kspace, displacements) -> kspace
        The matching exact inverse: `undo_motion`, or `undo_motion_radial`
        with the spoke index bound (e.g. via functools.partial).
    max_amp : float
        Largest displacement, in pixels, the search will consider.

    Search strategy: coarse-to-fine grid search (see `_coarse_to_fine`),
    ~10x cheaper than one fine grid. Jerk and drift start from a coarse grid
    over the full range. Periodic motion first gets a dedicated seeding stage
    (`_periodic_seed`), because its score has a dip in `cycles` too narrow for
    any coarse grid to hit.
    """
    space = _search_space(model, n_events, max_amp)
    scorer = _Scorer(model, n_events, acquired_kspace, undo)

    if model == "periodic":
        seed = _periodic_seed(scorer, max_amp)
        _coarse_to_fine(
            space, scorer, start=seed,
            start_half={"amp": 1.0, "cycles": 0.125},
            start_step={"amp": 0.25, "cycles": 1 / 32},
        )
    else:
        _coarse_to_fine(space, scorer)

    # Tidy the reported numbers (floating-point grid values like 7.999999).
    best_params = {
        k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 4))
        for k, v in scorer.best_params.items()
    }
    return AutofocusResult(
        model=model,
        params=best_params,
        displacements=_displacements_for(model, n_events, best_params),
        score=scorer.best_score,
        initial_score=gradient_entropy(from_kspace(acquired_kspace)),
        n_evaluations=scorer.evaluations,
    )
