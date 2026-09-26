"""
cs_lab_ui.py -- visualisation helpers used only by the compressed-sensing lab
page (`app/pages/2_Compressed_sensing_lab.py`).

These live apart from `journal_ui.py` on purpose. `journal_ui` is the shared
presentation layer for the whole app and every page depends on it; everything
here is specific to explaining one algorithm, and the lab is meant to be
removable without leaving a trace in the rest of the project. So the generic
pieces (`panel`, `figure`, `line_chart`, the stylesheet) are imported from
`journal_ui`, and only the algorithm-specific renderings are defined here:

    heatmap           -- a false-colour map, for residuals and differences
    kspace_heat       -- the same, on log-magnitude k-space
    wavelet_pyramid   -- the coefficient array with its subband grid drawn on
    threshold_map     -- which coefficients the shrink step deleted
    flow_strip        -- the seven sub-steps, with the current one lit up
    stage_header      -- a numbered sub-step heading with its formula
    formula           -- a display-maths line

WHY FALSE COLOUR HERE AND NOT ELSEWHERE
---------------------------------------
The rest of the app shows greyscale, matching what a scanner console shows.
These panels are not images of anatomy -- they are error maps and coefficient
arrays whose interesting values sit near zero and span several orders of
magnitude. A perceptually uniform colormap makes "almost zero" and "exactly
zero" distinguishable, which is the entire point of the panels that show what
soft thresholding removed.
"""

from __future__ import annotations

import html

import matplotlib
import numpy as np
import streamlit as st

import journal_ui as ui

# Perceptually uniform, dark at zero: error maps read as "bright where wrong".
ERROR_CMAP = "magma"
# Distinct from the error maps so the two kinds of panel are never confused.
KSPACE_CMAP = "viridis"

# Categorical colours for the threshold map. Chosen to survive a projector and
# to differ in lightness as well as hue, so the map still reads if the beamer
# mangles the colour.
COLOR_KILLED = (0.80, 0.16, 0.20)      # deleted by this shrink step
COLOR_SURVIVED = (1.00, 1.00, 1.00)    # still nonzero
COLOR_ALREADY_ZERO = (0.13, 0.15, 0.19)  # was already zero
COLOR_PROTECTED = (0.16, 0.44, 0.72)   # approximation band, never thresholded


# ---------------------------------------------------------------------------
# False-colour maps
# ---------------------------------------------------------------------------


def heatmap(
    array: np.ndarray,
    cmap: str = ERROR_CMAP,
    vmax: float | None = None,
) -> np.ndarray:
    """
    Map a non-negative array to RGB in [0, 1].

    `vmax` pins the top of the scale. Passing it is what lets a sequence of
    frames share one scale: without it every frame is normalised to its own
    peak, and a residual that is genuinely collapsing looks identically bright
    at iteration 1 and iteration 50. With it, the fade is the finding.
    """
    data = np.asarray(array, dtype=np.float64)
    top = float(data.max()) if vmax is None else float(vmax)
    scaled = data / top if top > 0 else np.zeros_like(data)
    rgba = matplotlib.colormaps[cmap](np.clip(scaled, 0.0, 1.0))
    return rgba[..., :3]


def kspace_heat(magnitude: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """
    log(1 + |K|) in false colour.

    k-space spans several orders of magnitude, so the log is not decoration --
    on a linear scale everything except the few central pixels is black. `vmax`
    is interpreted in the same log units, so callers sharing a scale across
    frames should pass a log-scale value (see :func:`log_scale_peak`).
    """
    return heatmap(np.log1p(np.abs(magnitude)), cmap=KSPACE_CMAP, vmax=vmax)


def log_scale_peak(magnitude: np.ndarray) -> float:
    """
    The `vmax` that :func:`kspace_heat` needs to pin its scale to this array.

    Exists so a caller can compute the scale once from the fully measured
    k-space and hold every later frame to it, rather than having to know that
    `kspace_heat` works in log units.
    """
    return float(np.log1p(np.abs(magnitude)).max())


# ---------------------------------------------------------------------------
# Wavelet coefficient displays
# ---------------------------------------------------------------------------


def _draw_grid(rgb: np.ndarray, band_edges, color=(1.0, 0.85, 0.2)) -> np.ndarray:
    """
    Overlay the subband division lines of a `coeffs_to_array` layout.

    `wavedec2` packs its output into one image-shaped array: the coarse
    approximation in the top-left corner, then each level's three detail bands
    (horizontal, vertical, diagonal) filling the L-shape around it and doubling
    in size at every finer scale. Undrawn, that is an unreadable grey texture;
    with the lines on, the pyramid structure is obvious and a viewer can see
    that the surviving coefficients cluster along anatomical edges.

    Each line is clipped to the L-shape it actually divides, rather than run
    across the whole array, so the drawing matches the real band layout.
    """
    out = rgb.copy()
    for row, col in band_edges:
        # The bands this line separates extend to twice the corner offset.
        out[row, : min(2 * col, out.shape[1])] = color
        out[: min(2 * row, out.shape[0]), col] = color
    return out


def wavelet_pyramid(
    coeff_magnitude: np.ndarray,
    band_edges,
    vmax: float | None = None,
    grid: bool = True,
) -> np.ndarray:
    """
    The wavelet coefficient array as a readable greyscale pyramid.

    Displayed as log(1 + |c|), and scaled against the largest *detail*
    coefficient rather than the largest coefficient overall. Both choices are
    needed to get a panel anyone can read:

    * On a linear scale the detail bands are invisible -- the coefficients that
      matter here are small by construction, which is the whole point.
    * Scaling against the global maximum is barely better, because that maximum
      always comes from the approximation band, whose coefficients are an order
      of magnitude larger than any detail coefficient. Normalising by it pushes
      every detail band to near-black and hides exactly the structure the panel
      exists to show.

    So the approximation band is allowed to saturate to white (it is a shrunken
    copy of the image, sitting in its own labelled corner -- nobody reads
    quantitative values off it) and the dynamic range is spent on the bands
    that carry the argument.
    """
    data = np.log1p(np.abs(np.asarray(coeff_magnitude, dtype=np.float64)))

    if vmax is None:
        vmax = detail_peak(coeff_magnitude, band_edges)
    top = float(vmax)

    scaled = np.clip(data / top, 0.0, 1.0) if top > 0 else np.zeros_like(data)
    rgb = np.repeat(scaled[..., None], 3, axis=2)
    return _draw_grid(rgb, band_edges) if grid else rgb


def detail_peak(coeff_magnitude: np.ndarray, band_edges) -> float:
    """
    The `vmax` that :func:`wavelet_pyramid` needs: the largest detail
    coefficient, in log units, with the approximation band excluded.

    Exposed separately so a caller can compute it once from the coefficients
    *before* shrinking and hold the after panel to the same scale -- otherwise
    each panel renormalises to its own peak and the two look identical, which
    would hide the very change they are placed side by side to show.
    """
    data = np.log1p(np.abs(np.asarray(coeff_magnitude, dtype=np.float64)))
    if not band_edges:
        return float(data.max())
    # band_edges[0] is the corner of the approximation block.
    rows, cols = band_edges[0]
    detail = data.copy()
    detail[:rows, :cols] = 0.0
    peak = float(detail.max())
    # A fully shrunk array can have no detail left at all; fall back to the
    # global peak so the panel renders mid-grey rather than dividing by zero.
    return peak if peak > 0 else float(data.max())


def threshold_map(
    killed: np.ndarray,
    protected: np.ndarray,
    survived: np.ndarray,
    band_edges,
) -> np.ndarray:
    """
    A four-colour map of what the shrink step did to every coefficient.

        red   -- was nonzero, is now exactly zero: deleted by this step
        white -- survived, still carries information
        dark  -- was already zero before this step
        blue  -- the approximation band, exempt from thresholding

    This is the single most useful panel on the page. The sparsity prior is an
    abstraction until you see that most of the array is red and the white
    survivors trace the anatomy's edges.
    """
    rgb = np.zeros(killed.shape + (3,), dtype=np.float64)
    rgb[...] = COLOR_ALREADY_ZERO
    rgb[survived] = COLOR_SURVIVED
    rgb[killed] = COLOR_KILLED
    rgb[protected] = COLOR_PROTECTED
    return _draw_grid(rgb, band_edges)


def threshold_legend() -> None:
    """The colour key for :func:`threshold_map`, as a compact inline row."""
    entries = [
        (COLOR_KILLED, "deleted by this step"),
        (COLOR_SURVIVED, "survived, still nonzero"),
        (COLOR_ALREADY_ZERO, "already zero"),
        (COLOR_PROTECTED, "approximation band (exempt)"),
    ]
    chips = "".join(
        f'<span class="cs-key"><i style="background:{_css_rgb(color)}"></i>'
        f"{html.escape(label)}</span>"
        for color, label in entries
    )
    _html(f'<div class="cs-legend">{chips}</div>')


def _html(markup: str) -> None:
    """Write raw HTML. Local rather than borrowed from `journal_ui` so this
    module depends only on that module's public surface."""
    st.markdown(markup, unsafe_allow_html=True)


def _css_rgb(color) -> str:
    r, g, b = (int(round(channel * 255)) for channel in color)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# The sub-step flow indicator
# ---------------------------------------------------------------------------

# Short label and one-line formula for each sub-step, keyed by the names in
# `cs_trace.SUBSTEPS`. Kept here rather than in the page body so the flow strip
# and the stage headings cannot disagree about what a step is called.
STEP_LABELS = {
    "predict": ("Predict", "K̂ = F x"),
    "residual": ("Residual", "r = M (y − K̂)"),
    "data_consistency": ("Data consistency", "x ← F⁻¹(K̂ + r)"),
    "analyse": ("Analyse", "c = W x"),
    "shrink": ("Shrink", "c ← soft(c, t)"),
    "synthesise": ("Synthesise", "x ← W⁻¹ c"),
    "momentum": ("Momentum", "z = x + β (x − x_prev)"),
}


def flow_strip(steps, active: str | None = None, done: set | None = None) -> None:
    """
    The sub-steps as a horizontal chain, with the current one highlighted.

    Gives the page a persistent "you are here": as the walkthrough advances,
    completed steps stay marked and the active one lights up, so a viewer can
    see both where in the iteration they are and how much is left.
    """
    done = done or set()
    chips = []
    for index, name in enumerate(steps, start=1):
        label = STEP_LABELS[name][0]
        state = "active" if name == active else ("done" if name in done else "todo")
        chips.append(
            f'<span class="cs-chip {state}"><b>{index}</b>{html.escape(label)}</span>'
        )
    _html('<div class="cs-flow">' + '<span class="cs-arrow">→</span>'.join(chips) + "</div>")


def stage_header(number: int, name: str) -> None:
    """A numbered sub-step heading with its formula set beside the title."""
    label, math = STEP_LABELS[name]
    _html(
        f'<div class="cs-stage"><span class="cs-stage-n">{number}</span>'
        f'<span class="cs-stage-t">{html.escape(label)}</span>'
        f'<code class="cs-stage-m">{html.escape(math)}</code></div>'
    )


def formula(text: str) -> None:
    """A centred display-maths line, for the objective and the like."""
    _html(f'<div class="cs-formula">{html.escape(text)}</div>')


def readout(items) -> None:
    """
    A row of label/value pairs, for the per-step numbers.

    Deliberately not `st.metric`: that renders a large coloured card that
    fights the journal styling used everywhere else in the app.
    """
    cells = "".join(
        f'<span class="cs-read"><span class="k">{html.escape(str(key))}</span>'
        f'<span class="v">{html.escape(str(value))}</span></span>'
        for key, value in items
    )
    _html(f'<div class="cs-readout">{cells}</div>')


# ---------------------------------------------------------------------------
# Page stylesheet
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
.cs-flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:0.3rem;
  margin:0.5rem 0 1.1rem; }}
.cs-chip {{ font-family:{ui.SANS}; font-size:0.84rem; font-weight:600;
  padding:0.28rem 0.6rem; border:1px solid {ui.HAIRLINE}; border-radius:2px;
  color:{ui.INK_3}; background:{ui.PAPER}; white-space:nowrap; }}
.cs-chip b {{ display:inline-block; margin-right:0.4rem; font-family:{ui.SERIF};
  color:{ui.INK_3}; }}
.cs-chip.done {{ color:{ui.INK}; border-color:{ui.INK_3}; }}
.cs-chip.done b {{ color:{ui.GREEN}; }}
.cs-chip.active {{ color:{ui.PAPER}; background:{ui.BLUE}; border-color:{ui.BLUE}; }}
.cs-chip.active b {{ color:{ui.PAPER}; }}
.cs-arrow {{ color:{ui.HAIRLINE}; font-size:0.9rem; }}

.cs-stage {{ display:flex; align-items:baseline; gap:0.6rem; margin:1.4rem 0 0.4rem;
  border-bottom:1px solid {ui.HAIRLINE}; padding-bottom:0.3rem; }}
.cs-stage-n {{ font-family:{ui.SERIF}; font-weight:700; font-size:1.05rem;
  color:{ui.PAPER}; background:{ui.INK}; width:1.55rem; height:1.55rem;
  display:inline-flex; align-items:center; justify-content:center; flex:none; }}
.cs-stage-t {{ font-family:{ui.SERIF}; font-weight:600; font-size:1.18rem; color:{ui.INK}; }}
.cs-stage-m {{ font-size:0.9rem !important; color:{ui.INK_2} !important;
  background:{ui.PANEL} !important; padding:0.1rem 0.4rem; }}

.cs-formula {{ font-family:{ui.SERIF}; font-size:1.1rem; text-align:center;
  color:{ui.INK}; background:{ui.PANEL}; padding:0.7rem 1rem; margin:0.7rem 0 1rem;
  border-left:3px solid {ui.INK_3}; }}

.cs-readout {{ display:flex; flex-wrap:wrap; gap:0 1.8rem; margin:0.5rem 0 0.9rem;
  border-top:1px solid {ui.HAIRLINE}; border-bottom:1px solid {ui.HAIRLINE};
  padding:0.5rem 0; }}
.cs-read {{ display:flex; flex-direction:column; }}
.cs-read .k {{ font-family:{ui.SANS}; font-size:0.7rem; text-transform:uppercase;
  letter-spacing:0.07em; font-weight:700; color:{ui.INK_3}; }}
.cs-read .v {{ font-family:{ui.SERIF}; font-size:1.15rem; color:{ui.INK};
  font-variant-numeric:tabular-nums; }}

.cs-legend {{ display:flex; flex-wrap:wrap; gap:0 1.1rem; margin:0.45rem 0 0.9rem; }}
.cs-key {{ font-family:{ui.SANS}; font-size:0.82rem; color:{ui.INK_2};
  display:inline-flex; align-items:center; gap:0.35rem; }}
.cs-key i {{ width:0.8rem; height:0.8rem; display:inline-block;
  box-shadow:0 0 0 1px {ui.INK_3}; }}
</style>
"""


def apply_style() -> None:
    """
    Inject the shared journal stylesheet plus this page's additions.

    Streamlit runs each page as its own script, so the CSS the main app injects
    is not present here and both sheets have to go in on every run.
    """
    ui.apply_style()
    _html(_CSS)
