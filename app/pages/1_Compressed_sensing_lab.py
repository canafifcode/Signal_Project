"""
1_Compressed_sensing_lab.py -- an interactive, step-by-step walkthrough of the
compressed-sensing reconstruction.

Run the app exactly as before:

    streamlit run app/streamlit_app.py

Streamlit picks up anything in `app/pages/` automatically, so this appears as a
second entry in the sidebar navigation. Nothing in the original app was
modified to make that happen.

WHAT THIS PAGE IS FOR
---------------------
Tab 4 of the main app answers "is compressed sensing better than zero-filling?"
with two images and a metrics table. This page answers the question a viewer
asks immediately afterwards -- *how?* -- by rendering the algorithm as it runs
instead of only its result.

Five sections, in the order the argument has to be made:

    1 Premise      -- medical images really are compressible. Shown on the
                      fully known image, before any undersampling, because the
                      whole method is worthless if this is not true.
    2 One step      -- a single iteration opened up: all seven sub-steps, every
                      intermediate array, with the numbers each one produced.
    3 Converge      -- the loop run live, frame by frame, with the metrics
                      drawn as they arrive.
    4 Incoherence   -- why this needs the random mask. CS run on all three
                      sampling strategies at one ratio.
    5 Lambda        -- the one knob, swept from under- to over-regularised.

REMOVABILITY
------------
This page depends on `mri_sim/cs_trace.py` (the instrumented algorithm) and
`app/cs_lab_ui.py` (its visualisations), and on nothing else that did not
already exist. Deleting those two files and `app/pages/` restores the original
single-page app exactly; no existing module imports any of them.

IMPLEMENTATION NOTES
--------------------
* Every widget key is prefixed `lab_`. Streamlit's session state is shared
  across pages, so an unprefixed key could collide with a widget of the same
  label in the main app and make the two pages fight over one value.
* The heavy runs are cached on `sample_id` plus parameters rather than on the
  arrays themselves, matching what `streamlit_app.py` does.
* Section 3 deliberately does *not* use the cache: its entire purpose is to be
  watched happening. The finished trace is kept in session state so the
  timeline underneath it works without recomputing.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

# This file sits two directories below the project root (app/pages/), and needs
# both the root (for `mri_sim`, `kspace_store`) and `app/` (for the shared
# presentation layer) importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.join(PROJECT_ROOT, "app")
for path in (PROJECT_ROOT, APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from kspace_store.store import KSpaceStore              # noqa: E402
from mri_sim import cs_trace, kspace as ks, metrics, noise  # noqa: E402

import cs_lab_ui as lab                                 # noqa: E402
import journal_ui as ui                                 # noqa: E402

STORE_PATH = os.path.join(PROJECT_ROOT, "data", "kspace_store")

SHORT_LABELS = {
    "cartesian": "Cartesian",
    "radial": "Radial",
    "variable_density": "Variable-density",
}

# Wavelets offered in the sidebar. db4 is the package default; Haar is included
# because its blocky basis makes the "wrong sparsifying transform" failure mode
# visible, which is worth being able to show.
WAVELETS = ["db4", "db2", "db8", "sym4", "haar"]

# Lambda values for section 5. Spread over a decade and a half so the sweep
# spans clearly under-regularised to clearly over-regularised.
LAMBDA_SWEEP = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------


@st.cache_resource
def get_store() -> KSpaceStore:
    return KSpaceStore(STORE_PATH)


@st.cache_data(show_spinner=False)
def load_sample(sample_id: str):
    """
    Arrays for one sample, as plain numpy so Streamlit's cache can hash them.

    k-space is promoted to complex128: it is stored as complex64 to halve the
    file size, but this page runs hundreds of FFTs and is better off in double
    precision.
    """
    sample = get_store().load(sample_id)
    return sample.image.astype(np.float64), sample.kspace.astype(np.complex128)


@st.cache_data(show_spinner=False)
def build_mask(kind: str, shape: tuple[int, int], ratio: float, seed: int) -> np.ndarray:
    extra = {"seed": seed} if kind == "variable_density" else {}
    return ks.build_mask(kind, shape, ratio, **extra)


@st.cache_data(show_spinner=False)
def acquire(sample_id: str, kind: str, ratio: float, snr_db: float | None, seed: int):
    """
    Simulate the scan: build the mask, sample k-space, optionally add noise.

    Returns (mask, acquired k-space, zero-filled reconstruction). The zero-fill
    comes back because it is both the algorithm's starting point and the
    baseline every section compares against.
    """
    _, full_kspace = load_sample(sample_id)
    mask = build_mask(kind, full_kspace.shape, ratio, seed)

    if snr_db is None:
        acquired = ks.apply_mask(full_kspace, mask)
    else:
        # Mask first, then noise: only the points actually measured carry
        # measurement noise.
        acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=snr_db, seed=seed)

    return mask, acquired, ks.from_kspace(acquired)


@st.cache_data(show_spinner=False)
def premise(sample_id: str, fraction: float, wavelet: str, level: int):
    """The "images are compressible" demo of section 1."""
    image, _ = load_sample(sample_id)
    reconstructed, kept, band_edges, cut = cs_trace.compress_by_keeping(
        image, fraction, wavelet=wavelet, level=level
    )
    approximation = np.abs(reconstructed)
    return (
        approximation,
        kept,
        band_edges,
        cut,
        metrics.compute_metrics(image, approximation),
    )


@st.cache_data(show_spinner=False)
def one_step(sample_id: str, acq: tuple, recon: tuple, at: int):
    """
    Iteration `at` with its full sub-step detail.

    Cached because the slider that picks `at` is dragged a lot, and reaching
    iteration 40 means actually running 40 iterations -- FISTA is sequential.
    """
    image, _ = load_sample(sample_id)
    strategy, ratio, snr_db, seed = acq
    lambda_, wavelet, level, use_fista = recon
    mask, acquired, _ = acquire(sample_id, strategy, ratio, snr_db, seed)
    return cs_trace.single_step(
        acquired, mask, at=at,
        lambda_=lambda_, wavelet=wavelet, level=level, use_fista=use_fista,
        reference=image,
    )


@st.cache_data(show_spinner=False)
def compare_strategies(
    sample_id: str, ratio: float, snr_db: float | None, seed: int,
    recon: tuple, n_iter: int,
):
    """
    CS versus zero-fill on all three sampling strategies -- section 4.

    Takes `ratio`/`snr_db`/`seed` rather than the page's acquisition tuple
    precisely because it must *not* depend on the selected strategy: it runs all
    three itself, and keying the cache on the sidebar's choice would throw three
    seconds of work away every time that dropdown changed without altering a
    single number in the result.
    """
    image, _ = load_sample(sample_id)
    lambda_, wavelet, level, use_fista = recon

    results = {}
    for kind in ks.ACQUISITION_MASKS:
        mask, acquired, zero_filled = acquire(sample_id, kind, ratio, snr_db, seed)
        steps = cs_trace.run(
            acquired, mask, lambda_=lambda_, n_iter=n_iter, wavelet=wavelet,
            level=level, use_fista=use_fista, reference=image,
        )
        cs_image = steps[-1].image.astype(np.float64)
        results[kind] = {
            "mask": mask,
            "zero_fill": zero_filled,
            "cs": cs_image,
            "zero_fill_metrics": metrics.compute_metrics(image, zero_filled),
            "cs_metrics": metrics.compute_metrics(image, cs_image),
            "sampled": ks.sampling_ratio(mask),
        }
    return results


@st.cache_data(show_spinner=False)
def lambda_sweep(
    sample_id: str, acq: tuple, wavelet: str, level: int, use_fista: bool,
    n_iter: int, lambdas: tuple,
):
    """
    Final reconstruction and scores at each lambda -- section 5.

    `lambda_` is deliberately absent from the arguments: this function sweeps
    its own values, so keying the cache on the sidebar's λ would recompute six
    reconstructions every time that slider moved and return exactly the same
    table.
    """
    image, _ = load_sample(sample_id)
    strategy, ratio, snr_db, seed = acq
    mask, acquired, _ = acquire(sample_id, strategy, ratio, snr_db, seed)

    rows, images = [], {}
    for value in lambdas:
        steps = cs_trace.run(
            acquired, mask, lambda_=value, n_iter=n_iter, wavelet=wavelet,
            level=level, use_fista=use_fista, reference=image,
        )
        final = steps[-1]
        # The closing projection has no objective of its own, so the sparsity
        # and L1 figures are read from the last true iteration.
        last_loop = steps[-2] if len(steps) > 1 else final
        scores = metrics.compute_metrics(image, final.image.astype(np.float64))
        rows.append({
            "lambda": value,
            "PSNR (dB)": scores["psnr"],
            "SSIM": scores["ssim"],
            "zeros (%)": last_loop.sparsity * 100.0,
            "data error": last_loop.data_error,
            "L1": last_loop.l1,
        })
        images[value] = final.image
    return pd.DataFrame(rows), images


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Compressed sensing, step by step", layout="wide")
lab.apply_style()

ui.title(
    "Compressed sensing, step by step",
    "The reconstruction rendered as it runs: every sub-step, every "
    "intermediate array, and the numbers each one produced",
)

try:
    store = get_store()
except FileNotFoundError:
    st.error(
        "No k-space store found. Build it first:\n\n"
        "```\npython -m kspace_store.build\n```"
    )
    st.stop()

# --- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("Scan setup")

    collection = st.selectbox(
        "Collection",
        store.collections(),
        format_func=lambda name: {
            "BrainTumorDataPublic": "Brain tumours (with masks)",
            "NINS_Dataset": "Brain pathologies",
            "MRI_Dataset": "Spine (real DICOM, T1/T2)",
            "3D_volumetric_imaging": "Other anatomy & CT",
        }.get(name, name),
        key="lab_collection",
    )
    records = store.records(collection=collection)
    titles = {record["id"]: record["title"] for record in records}
    sample_id = st.selectbox(
        "Subject", list(titles), format_func=lambda key: titles[key],
        key="lab_sample",
    )

    st.divider()

    strategy = st.selectbox(
        "Sampling strategy",
        ks.ACQUISITION_MASKS,
        index=ks.ACQUISITION_MASKS.index("variable_density"),
        format_func=lambda kind: ks.MASK_LABELS[kind],
        key="lab_strategy",
        help="Compressed sensing needs incoherent artifacts. Section 4 shows "
             "what happens with the other two.",
    )
    ratio = st.slider(
        "k-space sampled", min_value=0.03, max_value=0.60, value=0.125, step=0.005,
        format="%.3f", key="lab_ratio",
        help="Fraction of k-space acquired. 0.125 is an 8x faster scan.",
    )
    ui.note(f"R = {1 / ratio:.1f}, so the scan takes {ratio * 100:.1f}% of the time.")

    st.divider()
    st.subheader("Reconstruction")

    lambda_ = st.slider(
        "λ, sparsity strength", min_value=0.002, max_value=0.10, value=0.02,
        step=0.002, format="%.3f", key="lab_lambda",
        help="Larger = sparser = smoother. Too large and real anatomy is "
             "thresholded away.",
    )
    n_iter = st.slider(
        "Iterations", min_value=5, max_value=150, value=60, step=5, key="lab_iters",
    )
    use_fista = st.checkbox(
        "FISTA momentum", value=True, key="lab_fista",
        help="Off = plain ISTA. Same cost per iteration, much slower convergence.",
    )
    wavelet = st.selectbox(
        "Sparsifying wavelet", WAVELETS, key="lab_wavelet",
        help="db4 is the usual CS-MRI default. haar is blocky enough to see "
             "the basis in the result.",
    )
    level = st.slider(
        "Decomposition levels", min_value=1, max_value=4, value=3, step=1,
        key="lab_level",
    )

    st.divider()

    add_noise = st.checkbox("Simulate scanner noise", value=False, key="lab_noise")
    snr_db = st.slider(
        "k-space SNR (dB)", min_value=0.0, max_value=50.0, value=25.0, step=1.0,
        disabled=not add_noise, key="lab_snr",
    ) if add_noise else None
    seed = int(st.number_input(
        "Random seed", min_value=0, max_value=9999, value=0, step=1, key="lab_seed",
    ))

# --- Shared state -----------------------------------------------------------

image, full_kspace = load_sample(sample_id)
mask, acquired, zero_filled = acquire(sample_id, strategy, ratio, snr_db, seed)
zero_fill_scores = metrics.compute_metrics(image, zero_filled)
sampled_fraction = ks.sampling_ratio(mask)

# The configuration is carried into the cached functions as two tuples rather
# than one, so that each cache is keyed on only what it actually depends on.
# Bundling everything together would mean, for instance, that nudging λ threw
# away section 4's three reconstructions even though λ is one of the things it
# holds fixed.
acq_params = (strategy, ratio, snr_db, seed)
recon_params = (lambda_, wavelet, level, use_fista)

# Sections 4 and 5 run three and six full reconstructions respectively, so
# their iteration count is capped independently of the sidebar. Past ~60
# iterations these reconstructions have converged and the extra ones only cost
# the viewer time.
SURVEY_ITERS = min(n_iter, 60)

# Error maps are held to a common scale: the peak error of the zero-filled
# baseline. Auto-scaling each panel to its own maximum would make a genuinely
# improving reconstruction look equally wrong at every iteration.
error_vmax = float(np.abs(image - zero_filled).max())

if strategy != "variable_density":
    ui.remark(
        f"The sampling strategy is set to **{SHORT_LABELS[strategy]}**. Compressed "
        "sensing assumes the undersampling artifacts look like noise in the wavelet "
        "basis, which is only true for incoherent sampling — so expect it to "
        "underperform here. That is the point of section 4, not a bug.",
        kind="caution",
    )

def gate(flag_key: str, button_key: str, label: str, note: str) -> bool:
    """
    A button that unlocks an expensive section and then stays unlocked.

    Streamlit runs the body of *every* tab on every rerun, whichever one is
    actually open, so a section costing several seconds would be paid for on the
    first page load and again on every later interaction. Deferring it behind
    this gate means it is only computed when a viewer asks for it; the flag
    lives in session state so they are asked once, and the results underneath
    are cached regardless.

    Returns True when the section should render.
    """
    if st.session_state.get(flag_key):
        return True
    if st.button(label, key=button_key, type="primary"):
        st.session_state[flag_key] = True
        return True
    ui.note(note)
    return False


tabs = st.tabs([
    "1 The premise",
    "2 One iteration, step by step",
    "3 Watch it converge",
    "4 Why random sampling",
    "5 The λ dial",
])


# ---------------------------------------------------------------------------
# Section 1: the premise
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Before anything else: are these images actually compressible?")
    ui.lede(
        "Compressed sensing recovers an image from too few measurements by "
        "preferring the *sparsest* answer. That only works if the true image is "
        "sparse to begin with. So before undersampling anything, here is the "
        "claim tested directly: take the **fully known** image, throw away all "
        "but the largest few percent of its wavelet coefficients, and rebuild it."
    )

    keep = st.select_slider(
        "Detail coefficients kept",
        options=[0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005],
        value=0.05,
        format_func=lambda value: f"{value * 100:g}%",
        key="lab_keep",
    )

    approximation, kept, premise_edges, cut, premise_scores = premise(
        sample_id, keep, wavelet, level
    )

    # Every coefficient that is not kept and not in the exempt approximation
    # band was discarded, which is what the red in the map means here.
    protected = np.zeros(kept.shape, dtype=bool)
    protected[: premise_edges[0][0], : premise_edges[0][1]] = True
    discarded = ~kept & ~protected

    ui.figure(
        [
            ui.panel(image, "Original, fully known"),
            ui.panel(
                lab.threshold_map(discarded, protected, kept, premise_edges),
                f"Coefficients kept ({int(kept.sum()):,} of {int((~protected).sum()):,})",
            ),
            ui.panel(approximation, f"Rebuilt from {keep * 100:g}% of them"),
            ui.panel(
                lab.heatmap(np.abs(image - approximation)),
                f"Difference (peak {np.abs(image - approximation).max():.3f})",
            ),
        ],
        caption=(
            f"Wavelet compression of the true image with no undersampling at all. "
            f"Keeping the largest {keep * 100:g}% of the {wavelet} detail coefficients "
            f"(magnitude above {cut:.4f}) still reproduces the anatomy at "
            f"**{premise_scores['psnr']:.2f} dB / SSIM {premise_scores['ssim']:.4f}**. "
            "The approximation band, in blue, is kept in full — as the "
            "reconstruction also exempts it."
        ),
        number="1",
    )
    lab.threshold_legend()

    ui.remark(
        f"This is the entire justification for the method. The image has "
        f"{int((~protected).sum()):,} detail coefficients and needs about "
        f"{int(kept.sum()):,} of them to look right — so it carries far less "
        "information than its pixel count suggests, and there is room to recover "
        "it from far fewer than one measurement per pixel.",
        kind="result",
    )
    ui.remark(
        "One honest caveat. Here the *largest* coefficients could be picked "
        "because the true image was in hand. During a real scan it is not — which "
        "is exactly why the reconstruction has to **search** for a sparse image "
        "that agrees with the measurements, instead of just selecting one. That "
        "search is what the next section walks through.",
        kind="caution",
    )


# ---------------------------------------------------------------------------
# Section 2: one iteration, step by step
# ---------------------------------------------------------------------------


def render_stage(number: int, name: str, step, detail, coeff_vmax: float) -> None:
    """
    Draw one sub-step: its heading, its prose, its panels and its numbers.

    Shared by the static view and the animated walkthrough so the two cannot
    drift apart -- the only difference between them is *when* this gets called.
    """
    lab.stage_header(number, name)

    if name == "predict":
        ui.lede(
            "Start from the current estimate and ask the forward question: if "
            "this image were the patient, what would the scanner have recorded? "
            "That is just its Fourier transform."
        )
        ui.figure(
            [
                ui.panel(detail.input_image, "Estimate entering this iteration"),
                ui.panel(lab.kspace_heat(detail.predicted_kspace), "Its k-space, |F x|"),
                ui.panel(lab.kspace_heat(detail.measured_kspace), "What we measured, |y|"),
            ],
            columns=3,
        )
        ui.note(
            "Panel (c) is black wherever the scanner never looked. Panel (b) is "
            "filled everywhere — the estimate has an opinion about every "
            "frequency, including the ones nobody measured. Comparing the two "
            "only makes sense where (c) has data."
        )

    elif name == "residual":
        ui.lede(
            "Subtract the prediction from the measurements, and keep the "
            "difference **only where the scanner actually sampled**. Everywhere "
            "else we have no measurement to disagree with, so the residual is "
            "set to zero — we have no opinion there, and inventing one is "
            "precisely the mistake zero-filling makes."
        )
        ui.figure(
            [
                ui.panel(lab.kspace_heat(detail.residual_kspace), "Residual, |M (y − F x)|"),
                ui.panel(mask, f"The mask M, {sampled_fraction * 100:.1f}% sampled"),
            ],
            columns=2,
        )
        measured_energy = float(np.sum(detail.measured_kspace.astype(np.float64) ** 2))
        lab.readout([
            ("residual energy ‖r‖²", f"{detail.residual_energy:.3e}"),
            ("as % of ‖y‖²", f"{100 * detail.residual_energy / measured_energy:.3f}%"),
        ])
        ui.note(
            "This number is the algorithm's only contact with reality, and "
            "driving it toward zero is the data-consistency half of the job."
        )

    elif name == "data_consistency":
        ui.lede(
            "Fold the residual back in and return to image space. Because the "
            "measured points are simply overwritten with the truth and the rest "
            "left as the estimate had them, this is a gradient step on "
            "‖M F x − y‖² — and since the FFT is unitary, the correct step size "
            "is exactly 1, which is why no learning rate appears anywhere."
        )
        ui.figure(
            [
                ui.panel(detail.input_image, "Before"),
                ui.panel(detail.consistent_image, "After data consistency"),
                ui.panel(
                    lab.heatmap(detail.correction_image),
                    f"What changed (peak {detail.correction_image.max():.4f})",
                ),
            ],
            columns=3,
        )
        ui.note(
            "Panel (c) is the information the measurements just injected. It is "
            "the only step in the loop that adds anything true; every other step "
            "reshapes what is already there."
        )

    elif name == "analyse":
        ui.lede(
            "Now change the description of the image. The wavelet transform "
            "repacks it as a coarse approximation plus edge detail at each "
            "scale: the small square top-left is a shrunken copy of the image, "
            "and the L-shaped bands around it hold horizontal, vertical and "
            "diagonal edges, doubling in size as the scale gets finer."
        )
        ui.figure(
            [
                ui.panel(detail.consistent_image, "The image, in pixels"),
                ui.panel(
                    lab.wavelet_pyramid(detail.coeffs_before, detail.band_edges, coeff_vmax),
                    "The same image, in wavelet coefficients",
                ),
            ],
            columns=2,
        )
        lab.readout([
            ("detail coefficients", f"{detail.n_detail:,}"),
            ("approximation band", f"{detail.n_protected:,}"),
            ("levels", str(level)),
            ("wavelet", wavelet),
        ])
        ui.note(
            "Notice how much of the detail region is nearly black: smooth tissue "
            "produces almost no detail coefficients. That near-emptiness is the "
            "sparsity the next step is about to exploit."
        )

    elif name == "shrink":
        ui.lede(
            "The sparsity prior, and the only nonlinear step in the algorithm. "
            "Every detail coefficient is pulled toward zero by exactly t; "
            "anything that was already smaller than t lands on zero and is gone. "
            "The approximation band is left alone — it is a dense, high-energy "
            "summary of the anatomy, and shrinking it every iteration would "
            "steadily drain contrast out of the image."
        )
        survived = (detail.coeffs_after > 0) & ~detail.protected
        ui.figure(
            [
                ui.panel(
                    lab.wavelet_pyramid(detail.coeffs_before, detail.band_edges, coeff_vmax),
                    "Coefficients before",
                ),
                ui.panel(
                    lab.wavelet_pyramid(detail.coeffs_after, detail.band_edges, coeff_vmax),
                    "Coefficients after shrinking",
                ),
                ui.panel(
                    lab.threshold_map(detail.killed, detail.protected, survived, detail.band_edges),
                    "What the threshold did",
                ),
            ],
            columns=3,
        )
        lab.threshold_legend()
        lab.readout([
            ("threshold t", f"{step.threshold:.4f}"),
            ("deleted this step", f"{detail.n_killed:,}"),
            ("still nonzero", f"{int(survived.sum()):,}"),
            ("detail zeros", f"{step.sparsity * 100:.1f}%"),
        ])

        curve_x, curve_y = cs_trace.soft_threshold_curve(step.threshold, extent=4 * step.threshold + 1e-9)
        shrink_frame = pd.DataFrame({
            "input coefficient": np.concatenate([curve_x, curve_x]),
            "output": np.concatenate([curve_y, curve_x]),
            "curve": ["soft(c, t)"] * len(curve_x) + ["unchanged (c)"] * len(curve_x),
        })
        chart_left, chart_right = st.columns([2, 3], gap="large")
        with chart_left:
            ui.line_chart(
                shrink_frame, x="input coefficient", y="output", series="curve",
                series_order=["soft(c, t)", "unchanged (c)"],
                x_title="coefficient in", y_title="coefficient out",
                points=False, height=280,
            )
            ui.caption(
                f"The shrinkage operator at t = {step.threshold:.4f}. Inside the "
                f"dead zone of width 2t the output is exactly zero; outside it the "
                f"slope is 1 but offset toward zero by t — so *every* surviving "
                "coefficient is also reduced, not just the deleted ones.",
                number="2",
            )
        with chart_right:
            ui.remark(
                f"**{detail.n_killed:,}** of {detail.n_detail:,} detail coefficients "
                f"were set to zero by this one step, leaving "
                f"{step.sparsity * 100:.1f}% of them at exactly zero. In the map "
                "above, the white survivors trace the anatomy's edges — that is "
                "sparsity in the wavelet basis, made visible.",
                kind="result",
            )
            ui.note(
                "Soft thresholding is used rather than simply deleting small "
                "coefficients because it is the exact proximal operator of the L1 "
                "norm, which is what makes the alternation provably convergent. "
                "Hard thresholding would also create zeros, but the iteration "
                "would no longer be solving a well-posed problem."
            )

    elif name == "synthesise":
        ui.lede(
            "Invert the wavelet transform to get back an image. What the "
            "shrinking removed is worth looking at closely: it should be "
            "low-level incoherent grain spread over the whole field, not "
            "recognisable anatomy."
        )
        ui.figure(
            [
                ui.panel(detail.consistent_image, "Before shrinking"),
                ui.panel(detail.denoised_image, "After, the new estimate"),
                ui.panel(
                    lab.heatmap(detail.removed_image),
                    f"What shrinking removed (peak {detail.removed_image.max():.4f})",
                ),
            ],
            columns=3,
        )
        ui.remark(
            "Panel (c) is the diagnostic for λ. Grain and undersampling artifacts "
            "there mean the prior is doing its job. If you can make out edges, "
            "organs or the skull in it, λ is too large and the reconstruction is "
            "discarding real anatomy — try section 5.",
            kind="note",
        )

    elif name == "momentum":
        if detail.momentum_weight == 0.0:
            ui.lede(
                "FISTA momentum is switched off, so this step does nothing: the "
                "estimate is handed to the next iteration unchanged. That is "
                "plain ISTA — it converges to the same answer, just considerably "
                "more slowly. Turn momentum on in the sidebar to see the "
                "difference in section 3."
            )
            ui.figure(
                [ui.panel(detail.denoised_image, "Passed to the next iteration unchanged")],
                columns=1,
            )
        else:
            ui.lede(
                "Nesterov acceleration. Rather than handing the new estimate "
                "straight to the next iteration, overshoot a little further along "
                "the direction the last two iterates were already moving. It costs "
                "nothing extra per iteration and converges roughly quadratically "
                "faster."
            )
            overshoot = np.abs(detail.momentum_image - detail.denoised_image)
            ui.figure(
                [
                    ui.panel(detail.denoised_image, "The new estimate, x"),
                    ui.panel(detail.momentum_image, "Extrapolated, z — this enters the next step"),
                    ui.panel(lab.heatmap(overshoot), f"The overshoot (peak {overshoot.max():.4f})"),
                ],
                columns=3,
            )
            lab.readout([
                ("momentum weight β", f"{detail.momentum_weight:.4f}"),
                ("step change ‖Δx‖/‖x‖", f"{step.change:.3e}"),
            ])
            ui.note(
                "β climbs toward 1 as the iteration count grows, so early "
                "iterations barely extrapolate and later ones lean hard on the "
                "momentum."
            )


with tabs[1]:
    st.subheader("One iteration, opened up")
    ui.lede(
        "The loop alternates two ideas: *match what was measured*, then *simplify "
        "what was not*. Steps 1–3 below are one gradient step on the data term, "
        "steps 4–6 are the sparsity prior, and step 7 is the acceleration. Pick an "
        "iteration and walk through it."
    )
    lab.formula("minimise  ‖ M F x − y ‖²  +  λ ‖ W x ‖₁")
    ui.note(
        "F is the 2-D FFT, M the sampling mask, y the measured samples, W the "
        "wavelet transform. The first term says *agree with the scanner*; the "
        "second says *be simple*. λ sets the exchange rate between them."
    )

    picker = st.columns([2, 1, 1], gap="large")
    at = picker[0].slider(
        "Iteration to inspect", min_value=1, max_value=n_iter, value=min(3, n_iter),
        step=1, key="lab_at",
        help="Early iterations change a lot; later ones are fine adjustments.",
    )
    pause = picker[1].select_slider(
        "Walkthrough pace", options=[0.0, 0.4, 0.9, 1.6], value=0.9,
        format_func=lambda value: {0.0: "instant", 0.4: "brisk", 0.9: "steady", 1.6: "slow"}[value],
        key="lab_pace",
    )
    picker[2].markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    walk = picker[2].button("▶ Reveal step by step", key="lab_walk", use_container_width=True)

    with st.spinner(f"Running {at} iteration(s)..."):
        step = one_step(sample_id, acq_params, recon_params, at)
    detail = step.detail
    # Both pyramid panels are held to the peak of the coefficients *before*
    # shrinking, so the before/after pair is a real comparison.
    coeff_vmax = lab.detail_peak(detail.coeffs_before, detail.band_edges)

    ui.rule()
    lab.readout([
        ("iteration", f"{step.iteration} of {n_iter}"),
        ("PSNR now", ui.format_psnr(step.psnr)),
        ("zero-fill baseline", ui.format_psnr(zero_fill_scores["psnr"])),
        ("threshold t", f"{step.threshold:.4f}"),
        ("detail zeros", f"{step.sparsity * 100:.1f}%"),
        ("‖Δx‖/‖x‖", f"{step.change:.2e}"),
    ])

    flow_slot = st.empty()
    # One placeholder per sub-step. Creating them all up front fixes the page
    # layout, so the animated reveal fills them in order instead of making the
    # page jump around as elements are appended.
    stage_slots = [st.empty() for _ in cs_trace.SUBSTEPS]

    if walk:
        # The reveal. Streamlit flushes each element to the browser as it is
        # written, so filling the slots one at a time with a pause between them
        # animates the iteration in the order the algorithm performs it.
        done: set[str] = set()
        for index, name in enumerate(cs_trace.SUBSTEPS, start=1):
            with flow_slot.container():
                lab.flow_strip(cs_trace.SUBSTEPS, active=name, done=done)
            with stage_slots[index - 1].container():
                render_stage(index, name, step, detail, coeff_vmax)
            done.add(name)
            if pause:
                time.sleep(pause)
        with flow_slot.container():
            lab.flow_strip(cs_trace.SUBSTEPS, active=None, done=done)
    else:
        with flow_slot.container():
            lab.flow_strip(cs_trace.SUBSTEPS, active=None, done=set(cs_trace.SUBSTEPS))
        for index, name in enumerate(cs_trace.SUBSTEPS, start=1):
            with stage_slots[index - 1].container():
                render_stage(index, name, step, detail, coeff_vmax)

    ui.rule()
    st.markdown("#### Where that leaves us")
    ui.figure(
        [
            ui.panel(image, "Ground truth"),
            ui.panel(zero_filled, f"Zero-filled, {ui.format_psnr(zero_fill_scores['psnr'])}"),
            ui.panel(step.image, f"After iteration {step.iteration}, {ui.format_psnr(step.psnr)}"),
            ui.panel(
                lab.heatmap(np.abs(image - step.image.astype(np.float64)), vmax=error_vmax),
                "Remaining error",
            ),
        ],
        caption=(
            f"State of the reconstruction after {step.iteration} of {n_iter} "
            f"iterations. Panels (b) and (c) use exactly the same measurements; the "
            "difference is entirely in what each one assumed about the frequencies "
            "nobody measured. The error map shares its scale with (b)'s error, so "
            "the fading is real."
        ),
        number="3",
    )


# ---------------------------------------------------------------------------
# Section 3: watch it converge
# ---------------------------------------------------------------------------


def render_frame(slots: dict, step, previous_image: np.ndarray, history: list) -> None:
    """
    Draw one animation frame into a fixed set of placeholders.

    Used by both the live run and the timeline below it, so what you scrub back
    to is exactly what went past during the run.
    """
    estimate = step.image.astype(np.float64)
    error = np.abs(image - estimate)
    movement = np.abs(estimate - previous_image.astype(np.float64))

    label = (
        "closing data-consistency step"
        if step.is_final_projection
        else f"iteration {step.iteration} of {n_iter}"
    )

    with slots["panels"].container():
        ui.figure(
            [
                ui.panel(image, "Ground truth"),
                ui.panel(estimate, f"Estimate, {label}"),
                ui.panel(lab.heatmap(error, vmax=error_vmax), "Error vs truth"),
                ui.panel(lab.heatmap(movement), f"Changed this step (peak {movement.max():.4f})"),
            ],
            columns=4,
        )

    with slots["readout"].container():
        lab.readout([
            ("iteration", label),
            ("PSNR", ui.format_psnr(step.psnr)),
            ("gain over zero-fill", f"{step.psnr - zero_fill_scores['psnr']:+.2f} dB"),
            ("detail zeros", "—" if np.isnan(step.sparsity) else f"{step.sparsity * 100:.1f}%"),
            ("‖Δx‖/‖x‖", f"{step.change:.2e}"),
        ])

    frame = pd.DataFrame([
        {"iteration": entry.iteration, "PSNR (dB)": entry.psnr}
        for entry in history
        if entry.psnr is not None and np.isfinite(entry.psnr)
    ])
    with slots["chart"].container():
        if len(frame) > 1:
            ui.line_chart(
                frame, x="iteration", y="PSNR (dB)",
                x_title="iteration", y_title="PSNR (dB)",
                reference=(
                    zero_fill_scores["psnr"],
                    f"zero-fill baseline, {zero_fill_scores['psnr']:.2f} dB",
                ),
                points=False, height=260,
            )


with tabs[2]:
    st.subheader("The loop, running")
    ui.lede(
        "Press run and watch. The estimate starts as the zero-filled "
        "reconstruction — the Stage 1 baseline, artifacts and all — and each "
        "iteration alternates the two steps from section 2. The error map and the "
        "PSNR curve update as the frames arrive."
    )

    controls = st.columns([1, 1, 2], gap="large")
    speed = controls[0].select_slider(
        "Playback", options=["fast", "normal", "slow"], value="normal", key="lab_speed",
    )
    stride = controls[1].slider(
        "Draw every Nth iteration", min_value=1, max_value=10, value=1, step=1,
        key="lab_stride",
        help="Rendering is slower than the arithmetic. Raise this for a long run.",
    )
    controls[2].markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    run_live = controls[2].button(
        f"▶ Run {n_iter} iterations", key="lab_run", type="primary",
        use_container_width=True,
    )
    frame_pause = {"fast": 0.0, "normal": 0.06, "slow": 0.3}[speed]

    slots = {
        "status": st.empty(),
        "panels": st.empty(),
        "readout": st.empty(),
        "chart": st.empty(),
    }
    timeline_slot = st.container()

    # The trace is kept in session state so the timeline survives the reruns
    # that scrubbing it causes. It is tagged with the configuration that
    # produced it, so changing a slider invalidates it rather than silently
    # showing a trace from different settings.
    trace_key = (sample_id,) + acq_params + recon_params + (n_iter,)

    if run_live:
        # Drop the timeline's position before the slider is created this run.
        # A keyed widget takes its value from session state in preference to its
        # `value=` default, so without this a new run would leave the slider
        # pointing at whatever frame was last scrubbed to -- and if the new run
        # is shorter, at a frame beyond its end.
        st.session_state.pop("lab_frame", None)

        progress = slots["status"].progress(0.0, text="Running FISTA...")
        history: list = []
        previous = zero_filled

        for live_step in cs_trace.iterate(
            acquired, mask, lambda_=lambda_, n_iter=n_iter, wavelet=wavelet,
            level=level, use_fista=use_fista, reference=image,
        ):
            history.append(live_step)
            is_last = live_step.is_final_projection
            if live_step.iteration % stride == 0 or is_last:
                render_frame(slots, live_step, previous, history)
                progress.progress(
                    min(live_step.iteration / (n_iter + 1), 1.0),
                    text=(
                        "Closing data-consistency step"
                        if is_last
                        else f"Iteration {live_step.iteration} of {n_iter} — "
                             f"PSNR {live_step.psnr:.2f} dB"
                    ),
                )
                if frame_pause:
                    time.sleep(frame_pause)
            previous = live_step.image

        slots["status"].empty()
        st.session_state["lab_trace"] = history
        st.session_state["lab_trace_key"] = trace_key

    have_trace = st.session_state.get("lab_trace_key") == trace_key
    if not have_trace and "lab_trace" in st.session_state:
        # A run for different settings is not just useless, it is tens of
        # megabytes of float32 images. Drop it rather than holding it for a
        # configuration the viewer has moved on from.
        st.session_state.pop("lab_trace", None)
        st.session_state.pop("lab_trace_key", None)
        st.session_state.pop("lab_frame", None)
    trace = st.session_state.get("lab_trace") if have_trace else None

    if trace:
        with timeline_slot:
            ui.rule()
            st.markdown("#### Timeline")
            ui.note(
                "The run is kept in memory, so you can scrub back through it "
                "without recomputing."
            )
            position = st.slider(
                "Frame", min_value=1, max_value=len(trace), value=len(trace), step=1,
                key="lab_frame",
                format="%d",
            )
            if not run_live:
                chosen = trace[position - 1]
                previous_image = trace[position - 2].image if position > 1 else zero_filled
                render_frame(slots, chosen, previous_image, trace[:position])

            final = trace[-1]
            final_scores = metrics.compute_metrics(image, final.image.astype(np.float64))
            ui.table(
                ["Metric", "Zero-filled", "Compressed sensing", "Gain"],
                [
                    (
                        "PSNR (dB)",
                        f"{zero_fill_scores['psnr']:.2f}",
                        f"{final_scores['psnr']:.2f}",
                        f"{final_scores['psnr'] - zero_fill_scores['psnr']:+.2f}",
                    ),
                    (
                        "SSIM",
                        f"{zero_fill_scores['ssim']:.4f}",
                        f"{final_scores['ssim']:.4f}",
                        f"{final_scores['ssim'] - zero_fill_scores['ssim']:+.4f}",
                    ),
                ],
                caption=(
                    f"Final reconstruction after {n_iter} iterations at "
                    f"{sampled_fraction * 100:.1f}% sampling, λ = {lambda_:.3f}"
                ),
                number="1",
            )

            st.markdown("#### What the two halves of the objective were doing")
            charts = st.columns(3, gap="large")
            loop_only = [entry for entry in trace if not entry.is_final_projection]
            with charts[0]:
                ui.line_chart(
                    pd.DataFrame({
                        "iteration": [e.iteration for e in loop_only],
                        "log₁₀ ‖M F x − y‖²": [np.log10(max(e.data_error, 1e-30)) for e in loop_only],
                    }),
                    x="iteration", y="log₁₀ ‖M F x − y‖²",
                    x_title="iteration", y_title="log₁₀ data error",
                    points=False, height=250,
                )
                ui.caption(
                    "Data term. Falls as the estimate comes to agree with the "
                    "measurements that were actually taken.",
                    number="4",
                )
            with charts[1]:
                ui.line_chart(
                    pd.DataFrame({
                        "iteration": [e.iteration for e in loop_only],
                        "detail zeros (%)": [e.sparsity * 100 for e in loop_only],
                    }),
                    x="iteration", y="detail zeros (%)",
                    x_title="iteration", y_title="detail coefficients at zero (%)",
                    points=False, height=250,
                )
                ui.caption(
                    "Sparsity term. The fraction of detail coefficients at exactly "
                    "zero — the image getting simpler.",
                    number="5",
                )
            with charts[2]:
                ui.line_chart(
                    pd.DataFrame({
                        "iteration": [e.iteration for e in loop_only],
                        "log₁₀ ‖Δx‖/‖x‖": [np.log10(max(e.change, 1e-30)) for e in loop_only],
                    }),
                    x="iteration", y="log₁₀ ‖Δx‖/‖x‖",
                    x_title="iteration", y_title="log₁₀ relative change",
                    points=False, height=250,
                )
                ui.caption(
                    "Step size. Flattening toward zero is what convergence looks "
                    "like from the inside.",
                    number="6",
                )

            ui.remark(
                "The two terms pull against each other: data consistency pushes "
                "the estimate toward the measurements, shrinkage pushes it toward "
                "simplicity. The iteration is converging when neither can improve "
                "the total any further — which is where the step-size curve "
                "flattens out.",
                kind="remark",
            )
    else:
        with timeline_slot:
            ui.note(
                "No run in memory for these settings yet — press the button above. "
                "Changing any sidebar setting clears it, since the trace would no "
                "longer describe what the sidebar says."
            )


# ---------------------------------------------------------------------------
# Section 4: why random sampling
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Why this needs incoherent sampling")
    ui.lede(
        "Compressed sensing does not work on any undersampling pattern. It needs "
        "the artifacts to look like *noise* in the wavelet basis. Regular "
        "Cartesian skipping fails that test badly: it produces crisp, shifted "
        "copies of the anatomy, and a ghost of a brain is exactly as sparse as a "
        "brain — so no sparsity prior can tell which is which. Here is the same "
        "algorithm, same ratio, same λ, on all three strategies."
    )

    if gate(
        "lab_show_compare", "lab_cmp_btn", "Reconstruct all three strategies",
        f"Three reconstructions at {SURVEY_ITERS} iterations each — a few "
        "seconds. Press the button to run them.",
    ):

        with st.spinner("Reconstructing three ways..."):
            comparison = compare_strategies(
                sample_id, ratio, snr_db, seed, recon_params, SURVEY_ITERS
            )

        for kind in ks.ACQUISITION_MASKS:
            entry = comparison[kind]
            gain = entry["cs_metrics"]["psnr"] - entry["zero_fill_metrics"]["psnr"]
            st.markdown(f"#### {SHORT_LABELS[kind]} — {entry['sampled'] * 100:.1f}% sampled")
            ui.figure(
                [
                    ui.panel(entry["mask"], "Mask"),
                    ui.panel(
                        entry["zero_fill"],
                        f"Zero-filled, {ui.format_psnr(entry['zero_fill_metrics']['psnr'])}",
                    ),
                    ui.panel(
                        entry["cs"],
                        f"Compressed sensing, {ui.format_psnr(entry['cs_metrics']['psnr'])}",
                    ),
                    ui.panel(
                        lab.heatmap(np.abs(image - entry["zero_fill"])),
                        "Zero-fill error — the artifact structure",
                    ),
                    ui.panel(
                        lab.heatmap(np.abs(image - entry["cs"])),
                        "CS error — what survived the prior",
                    ),
                ],
                columns=5,
            )
            ui.note(f"CS changes PSNR by **{gain:+.2f} dB** on this strategy.")

        ui.caption(
            "The same reconstruction, at one sampling ratio and one λ, on all three "
            "strategies. The two error panels are the argument: where the zero-fill "
            "error is structured (coherent ghosts, streaks), the sparsity prior cannot "
            "distinguish it from anatomy and little is recovered; where it is "
            "fine-grained and incoherent, the prior removes it.",
            number="7",
        )

        table = pd.DataFrame([
            {
                "strategy": SHORT_LABELS[kind],
                "sampled %": comparison[kind]["sampled"] * 100,
                "zero-fill PSNR": comparison[kind]["zero_fill_metrics"]["psnr"],
                "CS PSNR": comparison[kind]["cs_metrics"]["psnr"],
                "PSNR gain": (
                    comparison[kind]["cs_metrics"]["psnr"]
                    - comparison[kind]["zero_fill_metrics"]["psnr"]
                ),
                "zero-fill SSIM": comparison[kind]["zero_fill_metrics"]["ssim"],
                "CS SSIM": comparison[kind]["cs_metrics"]["ssim"],
            }
            for kind in ks.ACQUISITION_MASKS
        ])
        ui.dataframe_table(
            table,
            {
                "sampled %": "{:.1f}",
                "zero-fill PSNR": "{:.2f}",
                "CS PSNR": "{:.2f}",
                "PSNR gain": "{:+.2f}",
                "zero-fill SSIM": "{:.4f}",
                "CS SSIM": "{:.4f}",
            },
            caption=f"What compressed sensing buys on each strategy, λ = {lambda_:.3f}",
            number="2",
        )

        best = table.loc[table["PSNR gain"].idxmax(), "strategy"]
        ui.remark(
            f"The largest gain is on **{best}** sampling. This is the reason the "
            "variable-density mask exists in this project at all: it is not a better "
            "mask in isolation — it is the mask whose artifacts a sparsity prior can "
            "actually remove.",
            kind="result",
        )


# ---------------------------------------------------------------------------
# Section 5: the lambda dial
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("λ: how much simplicity to insist on")
    ui.lede(
        "λ sets the exchange rate between the two terms. Too small and the "
        "reconstruction barely differs from zero-filling — the prior is not "
        "asserting anything. Too large and the sparsity term wins outright: real "
        "anatomy gets thresholded away and the image goes smooth and dim. The "
        "useful range is narrow, and this is how to find it."
    )

    if gate(
        "lab_show_lambda", "lab_lam_btn", "Run the λ sweep",
        f"{len(LAMBDA_SWEEP)} reconstructions at {SURVEY_ITERS} iterations each — "
        "under a minute. Press the button to run them.",
    ):

        with st.spinner(f"Reconstructing at {len(LAMBDA_SWEEP)} values of λ..."):
            sweep_table, sweep_images = lambda_sweep(
                sample_id, acq_params, wavelet, level, use_fista,
                SURVEY_ITERS, tuple(LAMBDA_SWEEP),
            )

        ui.figure(
            [ui.panel(zero_filled, f"λ = 0 (zero-fill), {ui.format_psnr(zero_fill_scores['psnr'])}")]
            + [
                ui.panel(
                    sweep_images[value],
                    f"λ = {value:.3f}, "
                    f"{sweep_table.loc[sweep_table['lambda'] == value, 'PSNR (dB)'].iloc[0]:.2f} dB",
                )
                for value in LAMBDA_SWEEP
            ],
            caption=(
                f"The same measurements reconstructed at increasing λ, "
                f"{sampled_fraction * 100:.1f}% sampling, {SURVEY_ITERS} iterations. Read left "
                "to right: the artifacts clear, then the anatomy starts going with them."
            ),
            number="8",
            columns=len(LAMBDA_SWEEP) + 1,
        )

        left, right = st.columns(2, gap="large")
        with left:
            ui.line_chart(
                sweep_table, x="lambda", y="PSNR (dB)",
                x_title="λ", y_title="PSNR (dB)",
                reference=(
                    zero_fill_scores["psnr"],
                    f"zero-fill, {zero_fill_scores['psnr']:.2f} dB",
                ),
                height=300,
            )
            ui.caption(
                "Quality against λ. The peak is the best trade-off for this ratio; "
                "both directions away from it are worse for different reasons.",
                number="9",
            )
        with right:
            ui.line_chart(
                sweep_table, x="L1", y="data error",
                x_title="‖W x‖₁  (sparsity term — smaller is simpler)",
                y_title="‖M F x − y‖²  (data term)",
                height=300,
            )
            ui.caption(
                "The trade-off curve, each point one λ. Moving left buys simplicity "
                "and pays in data fidelity. The corner of the L is the classic "
                "heuristic for a well-chosen λ.",
                number="10",
            )

        ui.dataframe_table(
            sweep_table,
            {
                "lambda": "{:.3f}",
                "PSNR (dB)": "{:.2f}",
                "SSIM": "{:.4f}",
                "zeros (%)": "{:.1f}",
                "data error": "{:.3e}",
                "L1": "{:.1f}",
            },
            caption=f"λ sweep at {sampled_fraction * 100:.1f}% sampling",
            number="3",
        )

        best_row = sweep_table.loc[sweep_table["PSNR (dB)"].idxmax()]
        ui.remark(
            f"Best PSNR here is **{best_row['PSNR (dB)']:.2f} dB** at "
            f"**λ = {best_row['lambda']:.3f}**, which drives "
            f"{best_row['zeros (%)']:.1f}% of the detail coefficients to zero — against "
            f"{zero_fill_scores['psnr']:.2f} dB for zero-filling the same data. Note "
            "that the best λ depends on the undersampling ratio: the more aggressive "
            "the acceleration, the more the prior has to do, and the larger λ wants "
            "to be.",
            kind="result",
        )
