"""
streamlit_app.py -- the interactive MRI k-space simulator.

Run it from the project root:

    streamlit run app/streamlit_app.py

then open http://localhost:8501 in a browser.

WHAT THIS IS
------------
A teaching front end for everything in `mri_sim`, driven by the pre-built
k-space sample store so it starts instantly and never touches the 12 GB raw
dataset. Five tabs, each answering one question:

    1. Acquire      -- what does undersampling do to the image?
    2. Center vs edges -- which frequencies carry what?
    3. Noise        -- what does scanner noise do, and how does it interact
                       with undersampling?
    4. Compressed sensing -- can we do better than assuming the missing data
                       was zero?
    5. Sweep        -- how do PSNR and SSIM fall off as we accelerate?

DESIGN NOTES
------------
* The heavy work is cached with `@st.cache_data`, so dragging a slider only
  recomputes what actually changed. A masked reconstruction is ~1 ms, so the
  UI keeps up with the slider.
* Every array that crosses a function boundary is centered k-space, matching
  the convention used everywhere else in the project.
* Reconstructions are displayed with a fixed [0, 1] range, never auto-scaled,
  so the brightness you see is the brightness the reconstruction has.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the project root: the app
# lives one directory down, so the project root has to be importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kspace_store.store import KSpaceStore          # noqa: E402
from mri_sim import cs, denoise, kspace as ks, metrics, motion, noise, roi  # noqa: E402
from mri_sim import motion_correction as mcorr                             # noqa: E402

# The presentation layer lives next to this file.  puts the
# script's directory on the path, but being explicit keeps imports working
# however the app is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import journal_ui as ui                              # noqa: E402

# Anchored to the project root, not the working directory: `streamlit run`
# can be invoked from anywhere, and a relative path would make the app
# claim the store is missing when it is merely elsewhere.
STORE_PATH = os.path.join(PROJECT_ROOT, "data", "kspace_store")

# Ratios used by the sweep tab. 1.0 is included as the "no undersampling"
# reference point.
SWEEP_RATIOS = [1.0, 0.5, 0.25, 0.125, 0.0625]

# Short strategy names for captions, remarks and chart legends, where the
# long `ks.MASK_LABELS` descriptions would wrap.
SHORT_LABELS = {
    "cartesian": "Cartesian",
    "radial": "Radial",
    "variable_density": "Variable-density",
}


# ---------------------------------------------------------------------------
# Data access, cached
# ---------------------------------------------------------------------------


@st.cache_resource
def get_store() -> KSpaceStore:
    """The store object itself: opened once per server process."""
    return KSpaceStore(STORE_PATH)


@st.cache_data(show_spinner=False)
def load_sample(sample_id: str):
    """
    Arrays for one sample.

    Returned as plain numpy arrays rather than the Sample dataclass because
    Streamlit's cache hashes what it stores, and arrays hash cleanly.

    The k-space is promoted to complex128 here. It is stored as complex64 to
    halve the file size, but the iterative reconstruction runs hundreds of
    FFTs and is better off in double precision.
    """
    sample = get_store().load(sample_id)
    return (
        sample.image.astype(np.float64),
        sample.kspace.astype(np.complex128),
        None if sample.tumor_mask is None else sample.tumor_mask.astype(bool),
        sample.meta,
    )


@st.cache_data(show_spinner=False)
def build_mask(kind: str, shape: tuple[int, int], ratio: float, seed: int) -> np.ndarray:
    """Cached mask construction (the radial mask's binary search is the slow one)."""
    extra = {"seed": seed} if kind == "variable_density" else {}
    return ks.build_mask(kind, shape, ratio, **extra)


@st.cache_data(show_spinner=False)
def acquire(
    sample_id: str,
    kind: str,
    ratio: float,
    snr_db: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate one scan: build the mask, sample k-space, add noise, reconstruct.

    Returns (acquired k-space, reconstruction, metrics).
    """
    image, full_kspace, _, _ = load_sample(sample_id)
    mask = build_mask(kind, image.shape, ratio, seed)

    if snr_db is None:
        acquired = ks.apply_mask(full_kspace, mask)
    else:
        # Mask first, then noise: only measured points carry measurement noise.
        acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=snr_db, seed=seed)

    reconstruction = ks.from_kspace(acquired)
    return acquired, reconstruction, metrics.compute_metrics(image, reconstruction)


@st.cache_data(show_spinner=False)
def acquire_with_motion(
    sample_id: str,
    kind: str,
    ratio: float,
    model: str,
    amp: float,
    at_frac: float,
    cycles: float,
    snr_db: float | None,
    seed: int,
):
    """
    Like `acquire`, but with patient motion timed correctly for `kind`.

    The branch below is the whole point of this function. A Cartesian scan
    fills k-space one ROW per repetition, so row index is acquisition time
    and `apply_motion` is right. A radial scan fills one SPOKE per
    repetition, and every spoke crosses the k-space centre -- timing it by
    row would stamp a single row's phase error onto that centre and throw
    away radial's motion robustness, which is exactly the effect this tab
    exists to show. So radial goes through `draw_spokes_indexed` /
    `apply_motion_radial` instead, with the mask rebuilt from the same
    `n_spokes` so mask and index map agree point for point.

    Motion is applied BEFORE masking, so noise (added by
    `simulate_acquisition`) still lands only on the samples the scanner
    actually measured.

    Returns (mask, acquired k-space, reconstruction, metrics, displacements).
    """
    image, full_kspace, _, _ = load_sample(sample_id)
    ny, _ = image.shape

    if kind == "radial":
        n_spokes = motion.n_spokes_for_ratio(image.shape, ratio)
        index_map = motion.draw_spokes_indexed(image.shape, n_spokes)
        mask = ks.radial_mask(image.shape, n_spokes=n_spokes)
        # index_map.max(), not n_spokes: a spoke can lose every one of its
        # pixels to the angle tie-break, and apply_motion_radial validates
        # the displacement count against what is actually in the map.
        displacements = build_displacements(
            model, int(index_map.max()), amp, at_frac, cycles
        )
        corrupted = motion.apply_motion_radial(full_kspace, index_map, displacements)
    else:
        mask = build_mask(kind, image.shape, ratio, seed)
        displacements = build_displacements(model, ny, amp, at_frac, cycles)
        corrupted = motion.apply_motion(full_kspace, displacements)

    if snr_db is None:
        acquired = ks.apply_mask(corrupted, mask)
    else:
        acquired = noise.simulate_acquisition(corrupted, mask, snr_db=snr_db, seed=seed)

    reconstruction = ks.from_kspace(acquired)
    return (
        mask,
        acquired,
        reconstruction,
        metrics.compute_metrics(image, reconstruction),
        displacements,
    )


@st.cache_data(show_spinner=False)
def spoke_index_map(shape: tuple[int, int], ratio: float) -> np.ndarray:
    """Cached: the spoke-count binary search is the slow part of radial."""
    return motion.draw_spokes_indexed(shape, motion.n_spokes_for_ratio(shape, ratio))


def motion_undo(kind: str, shape: tuple[int, int], ratio: float):
    """
    The exact inverse matching `acquire_with_motion`'s corruption for `kind`:
    per-row for Cartesian and the others, per-spoke for radial. The spoke
    index map is deterministic, so rebuilding it here gives exactly the one
    the corruption used.
    """
    if kind != "radial":
        return mcorr.undo_motion
    index_map = spoke_index_map(tuple(shape), ratio)
    return lambda kspace, displacements: mcorr.undo_motion_radial(
        kspace, index_map, displacements
    )


@st.cache_data(show_spinner=False)
def run_autofocus(
    sample_id: str,
    kind: str,
    ratio: float,
    model: str,
    amp: float,
    at_frac: float,
    cycles: float,
    snr_db: float | None,
    seed: int,
):
    """
    Estimate the motion from the corrupted scan alone and undo it.

    The true displacements returned by `acquire_with_motion` are used only
    for their COUNT (rows or spokes); autofocus never sees their values.

    Returns (AutofocusResult, corrected reconstruction, metrics).
    """
    image, _, _, _ = load_sample(sample_id)
    _, acquired, _, _, displacements = acquire_with_motion(
        sample_id, kind, ratio, model, amp, at_frac, cycles, snr_db, seed
    )
    undo = motion_undo(kind, image.shape, ratio)
    result = mcorr.autofocus(acquired, model, len(displacements), undo)
    corrected = ks.from_kspace(undo(acquired, result.displacements))
    return result, corrected, metrics.compute_metrics(image, corrected)


def build_displacements(
    model: str, n: int, amp: float, at_frac: float, cycles: float
) -> list[tuple[float, float]]:
    """
    One (dy, dx) displacement per acquisition event, ready for `apply_motion`.

    `n` is the number of events, which is NOT the same thing for every
    trajectory: a Cartesian scan acquires one row per repetition, so n = ny,
    while a radial scan acquires one whole spoke per repetition, so
    n = n_spokes. Keeping that count a parameter is what lets the same three
    motion models drive both pipelines.
    """
    if model == "none" or amp == 0.0:
        return [(0.0, 0.0)] * n
    if model == "sudden_jerk":
        return motion.sudden_jerk(n, amp, int(round(at_frac * n)))
    if model == "slow_drift":
        return motion.slow_drift(n, amp)
    return motion.periodic(n, amp, cycles)


@st.cache_data(show_spinner=False)
def run_cs(
    sample_id: str,
    ratio: float,
    lambda_: float,
    n_iter: int,
    snr_db: float | None,
    seed: int,
) -> dict:
    """Cached compressed-sensing run (the only genuinely slow operation, ~1 s)."""
    image, full_kspace, _, _ = load_sample(sample_id)
    mask = build_mask("variable_density", image.shape, ratio, seed)

    if snr_db is None:
        acquired = ks.apply_mask(full_kspace, mask)
    else:
        acquired = noise.simulate_acquisition(full_kspace, mask, snr_db=snr_db, seed=seed)

    result = cs.compare_with_zero_fill(
        acquired, mask, image, lambda_=lambda_, n_iter=n_iter
    )
    result["mask"] = mask
    return result


@st.cache_data(show_spinner=False)
def sweep(sample_id: str, snr_db: float | None, seed: int) -> pd.DataFrame:
    """PSNR/SSIM for every strategy at every ratio -- the summary chart."""
    rows = []
    for kind in ks.ACQUISITION_MASKS:
        for ratio in SWEEP_RATIOS:
            _, _, scores = acquire(sample_id, kind, ratio, snr_db, seed)
            mask = build_mask(kind, load_sample(sample_id)[0].shape, ratio, seed)
            rows.append({
                "strategy": SHORT_LABELS[kind],
                "sampling %": ks.sampling_ratio(mask) * 100.0,
                "acceleration": ks.acceleration_factor(mask),
                "PSNR (dB)": scores["psnr"],
                "SSIM": scores["ssim"],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def roi_locality(sample_id: str, quadrant: str) -> dict:
    """
    Cached "k-space is not spatially local" demo -- deletes one quadrant of
    k-space and reports where in the *image* the damage landed.
    """
    image, _, _, _ = load_sample(sample_id)
    return roi.kspace_locality_demo(image, quadrant=quadrant)


@st.cache_data(show_spinner=False)
def roi_compare(sample_id: str, center: tuple[int, int], R: int, seed: int) -> dict:
    """
    Cached four-way reduced-FOV comparison.

    One call covers both the pipeline panel and the comparison table: the
    `reduced_fov` variant it returns already carries the excited object, the
    coarse grid, its k-space and the reconstruction, so the tab never has to
    run the acquisition twice.
    """
    image, _, _, _ = load_sample(sample_id)
    return roi.compare_roi_strategies(image, center, R=R, seed=seed)


def box_overlay(base: np.ndarray, box: roi.ROIBox) -> np.ndarray:
    """
    The image with the excitation box outlined in red.

    Drawn as an RGB overlay rather than by dimming the outside, so panel 1
    shows *where the box is on the whole head* while panel 2 shows what is
    left after the pulse -- the two panels are making different points.
    """
    rgb = np.stack([np.clip(base, 0.0, 1.0)] * 3, axis=-1)
    y1, x1 = box.y0 + box.size - 1, box.x0 + box.size - 1
    edge = [1.0, 0.25, 0.25]
    rgb[box.y0, box.x0:x1 + 1] = edge
    rgb[y1, box.x0:x1 + 1] = edge
    rgb[box.y0:y1 + 1, box.x0] = edge
    rgb[box.y0:y1 + 1, x1] = edge
    return rgb


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def error_panel(original: np.ndarray, reconstruction: np.ndarray, label: str) -> dict:
    """Absolute difference map, auto-scaled (the label reports the true peak)."""
    error = np.abs(original - reconstruction)
    return ui.panel(error, f"{label} (peak {error.max():.3f})", stretch=True)


def kspace_panel(kspace: np.ndarray, label: str) -> dict:
    return ui.panel(ui.kspace_display(kspace), label)


def sweep_long(table: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    One metric from the sweep table, in long format for plotting.

    The fully sampled point is dropped: its PSNR is infinite and its SSIM is
    exactly 1, so it only stretches the axis without saying anything.
    """
    frame = table[table["sampling %"] < 99.5][["strategy", "sampling %", value]]
    return frame[np.isfinite(frame[value])]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MRI k-Space Reconstruction Simulator",
    layout="wide",
)
ui.apply_style()

ui.title(
    "MRI k-Space Reconstruction Simulator",
    "Fourier-domain acquisition, and what happens to the image when a scan "
    "is accelerated by measuring less of it",
)

try:
    store = get_store()
except FileNotFoundError:
    st.error(
        "No k-space store found. Build it first:\n\n"
        "```\npython -m kspace_store.build\n```"
    )
    st.stop()

# --- Sidebar: the scan setup ------------------------------------------------

with st.sidebar:
    st.header("Scan setup")

    # Group the sample picker by collection so the list is navigable.
    collection = st.selectbox(
        "Collection",
        store.collections(),
        format_func=lambda name: {
            "BrainTumorDataPublic": "Brain tumours (with masks)",
            "NINS_Dataset": "Brain pathologies",
            "MRI_Dataset": "Spine (real DICOM, T1/T2)",
            "3D_volumetric_imaging": "Other anatomy & CT",
        }.get(name, name),
    )
    records = store.records(collection=collection)
    titles = {record["id"]: record["title"] for record in records}
    sample_id = st.selectbox(
        "Subject", list(titles), format_func=lambda key: titles[key]
    )

    st.divider()

    strategy = st.selectbox(
        "Sampling strategy",
        ks.ACQUISITION_MASKS,
        format_func=lambda kind: ks.MASK_LABELS[kind],
    )
    ratio = st.slider(
        "k-space sampled", min_value=0.02, max_value=1.0, value=0.25, step=0.01,
        format="%.2f",
        help="Fraction of k-space the scanner acquires. 0.25 means a 4x faster scan.",
    )
    ui.note(
        f"R = {1 / ratio:.1f}, so the scan takes {ratio * 100:.0f}% of the full "
        "acquisition time."
    )

    st.divider()

    add_noise = st.checkbox(
        "Simulate scanner noise", value=False,
        help="Complex Gaussian noise added to the k-space samples that were "
             "actually measured.",
    )
    snr_db = st.slider(
        "k-space SNR (dB)", min_value=0.0, max_value=50.0, value=25.0, step=1.0,
        disabled=not add_noise,
        help="40 dB is a clean clinical scan, 20 dB is visibly grainy, 10 dB is bad.",
    ) if add_noise else None

    seed = st.number_input(
        "Random seed", min_value=0, max_value=9999, value=0, step=1,
        help="Controls the random mask draw and the noise realisation.",
    )

    st.divider()
    ui.note(
        f"Store: {len(store)} samples at "
        f"{store.manifest['resolution']}×{store.manifest['resolution']}."
    )

# --- Load the chosen subject ------------------------------------------------

image, full_kspace, tumor_mask, meta = load_sample(sample_id)
mask = build_mask(strategy, image.shape, ratio, int(seed))
acquired, reconstruction, scores = acquire(
    sample_id, strategy, ratio, snr_db, int(seed)
)
strategy_label = SHORT_LABELS[strategy]

tabs = st.tabs([
    "1 Acquire",
    "2 Centre vs edges",
    "3 Noise",
    "4 Compressed sensing",
    "5 Sweep",
    "6 Motion",
    "7 Reduced FOV",
    "8 Sample",
])

# ---------------------------------------------------------------------------
# Tab 1: the main pipeline
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("The pipeline, end to end")
    ui.lede(
        "The scanner measures **k-space**, not an image. Skipping samples makes "
        "the scan faster; the inverse FFT then has to assume the missing samples "
        "were zero, and that wrong assumption is what you see as artifacts."
    )

    sampled = ks.sampling_ratio(mask) * 100
    ui.figure(
        [
            ui.panel(image, "Ground truth"),
            kspace_panel(full_kspace, "Full k-space, log(1 + |K|)"),
            ui.panel(mask, f"Mask, {sampled:.1f}% acquired"),
            kspace_panel(acquired, "What the scanner got"),
            ui.panel(reconstruction, "Reconstruction (zero-filled)"),
        ],
        caption=(
            f"{strategy_label} undersampling at R = {ks.acceleration_factor(mask):.1f} "
            f"({sampled:.1f}% of k-space). (a) Ground truth. (b) Fully sampled "
            "k-space on a log scale. (c) Sampling mask. (d) Acquired k-space. "
            "(e) Zero-filled reconstruction."
        ),
        number="1",
    )

    left, middle, right = st.columns([3, 2, 5], gap="large")
    with left:
        ui.metrics_table(
            scores,
            caption="Reconstruction quality",
            number="1",
            extra=[
                ("k-space acquired", f"{sampled:.1f}%"),
                ("Acceleration", f"{ks.acceleration_factor(mask):.1f}×"),
            ],
        )
    with middle:
        ui.figure(
            [error_panel(image, reconstruction, "Absolute error")],
            first_letter="f",
        )
    with right:
        ui.remark(
            {
                "cartesian": "Skipping whole lines folds the image onto itself. "
                             "The ghosts are crisp copies of the anatomy, shifted "
                             "by FOV/acceleration. Coherent artifacts like these "
                             "are the hardest kind to remove, because they look "
                             "like real structure.",
                "radial": "Spokes oversample the centre and leave gaps that "
                          "widen outward, so the error appears as streaks "
                          "radiating from bright edges. Radial is also known for "
                          "being robust to motion, since every spoke re-measures "
                          "the centre.",
                "variable_density": "The error is spread out as incoherent, "
                                    "noise-like grain instead of structured "
                                    "ghosts. That is exactly the property "
                                    "compressed sensing needs (see tab 4).",
            }[strategy],
            lead=f"{strategy_label}.",
        )

# ---------------------------------------------------------------------------
# Tab 2: centre vs edges
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Which part of k-space carries what?")
    ui.lede(
        "Both reconstructions below use the **same number of samples**. The only "
        "difference is *where* in k-space those samples were taken."
    )

    demo_ratio = st.slider(
        "Fraction of k-space kept (in both cases)",
        min_value=0.01, max_value=0.5, value=0.10, step=0.01, format="%.2f",
    )

    center_mask = ks.build_mask("center_only", image.shape, demo_ratio)
    edges_mask = ks.build_mask("edges_only", image.shape, demo_ratio)
    center_recon = ks.from_kspace(ks.apply_mask(full_kspace, center_mask))
    edges_recon = ks.from_kspace(ks.apply_mask(full_kspace, edges_mask))

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Centre only: a low-pass filter")
        ui.figure(
            [
                ui.panel(center_mask, "Mask"),
                ui.panel(center_recon, "As reconstructed"),
                ui.panel(center_recon, "Contrast stretched", stretch=True),
            ],
        )
        ui.metrics_table(metrics.compute_metrics(image, center_recon))
        ui.remark(
            f"Mean brightness **{center_recon.mean():.3f}** vs {image.mean():.3f} "
            "for the original. Contrast and shape are intact; only fine detail is "
            "lost. The faint rings around sharp edges are **Gibbs ringing**, from "
            "truncating the Fourier series at the rim of the disc.",
            kind="result",
        )

    with right:
        st.markdown("#### Edges only: a high-pass filter")
        ui.figure(
            [
                ui.panel(edges_mask, "Mask"),
                ui.panel(edges_recon, "As reconstructed"),
                ui.panel(edges_recon, "Contrast stretched", stretch=True),
            ],
            first_letter="d",
        )
        ui.metrics_table(metrics.compute_metrics(image, edges_recon))
        ui.remark(
            f"Mean brightness **{edges_recon.mean():.4f}**, essentially black. "
            "Throwing away the centre throws away the DC term, i.e. the average "
            "brightness of the whole image, along with every slowly-varying "
            "structure. Stretched, it is an edge map.",
            kind="caution",
        )

    ui.caption(
        f"Two masks keeping the same {demo_ratio * 100:.0f}% of k-space. "
        "(a–c) Central disc only. (d–f) Everything except the central disc.",
        number="2",
    )

    energy = meta["stats"]["energy_within_r0.1"]
    ui.remark(
        f"For this sample, **{energy * 100:.1f}%** of all k-space energy sits inside "
        "the central 10% radius, which is about 1% of the samples. That is why "
        "every realistic mask in tab 1 protects the centre."
    )

# ---------------------------------------------------------------------------
# Tab 3: noise
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Scanner noise lives in k-space")
    ui.lede(
        "Real noise is added to the **measured samples**, not to the finished "
        "image. It is complex (the receiver has an I and a Q channel) and white, "
        "with the same power at every frequency. Since the outer samples are tiny "
        "and the centre is huge, the same noise destroys fine detail long before "
        "it touches overall contrast."
    )

    demo_snr = st.slider(
        "k-space SNR (dB)", min_value=0.0, max_value=50.0, value=20.0, step=1.0,
        key="noise_tab_snr",
    )

    noisy_full = noise.add_kspace_noise(full_kspace, snr_db=demo_snr, seed=int(seed))
    noisy_recon = ks.from_kspace(noisy_full)
    noisy_acquired = noise.simulate_acquisition(
        full_kspace, mask, snr_db=demo_snr, seed=int(seed)
    )
    noisy_under = ks.from_kspace(noisy_acquired)

    ui.figure(
        [
            ui.panel(image, "Ground truth"),
            ui.panel(noisy_recon, "Noisy, fully sampled"),
            ui.panel(reconstruction, f"Undersampled {ratio * 100:.0f}%, noiseless"),
            ui.panel(noisy_under, f"Undersampled {ratio * 100:.0f}% + noise"),
        ],
        caption=(
            f"Noise at {demo_snr:.0f} dB k-space SNR, with and without "
            f"{strategy_label.lower()} undersampling. (a) Noiseless, fully "
            "sampled reference."
        ),
        number="3",
    )

    columns = st.columns(4, gap="medium")
    for column, recon in zip(
        columns[1:], (noisy_recon, reconstruction, noisy_under)
    ):
        with column:
            ui.metrics_table(metrics.compute_metrics(image, recon))

    ui.note(
        f"Verification: requested {demo_snr:.0f} dB, measured "
        f"{noise.measured_snr_db(full_kspace, noisy_full):.2f} dB on the full k-space."
    )

    with st.expander("Why does undersampling let in less total noise?"):
        st.markdown(
            "Noise enters once per measurement, so a mask that keeps 25% of "
            "k-space also admits about 25% of the noise energy, the familiar "
            "`SNR ∝ √N` of MRI. That does **not** make fast scans cleaner: you "
            "lose signal and gain artifacts at the same time. A faster scan is "
            "noisier *per unit of signal*.\n\n"
            "Notice also that the magnitude operation turns zero-mean complex "
            "noise into strictly positive **Rician** noise, which is why the "
            "background of a noisy MRI image is a faint grey haze rather than "
            "true black."
        )

    # --- Denoising ----------------------------------------------------------
    # Applied to the noisy *fully sampled* image, so what is removed is noise
    # alone, not undersampling artifacts. See mri_sim/denoise.py.
    ui.rule()
    st.subheader("Taking the noise back out")
    ui.lede(
        "Anatomy is carried by a **few large** wavelet coefficients; white noise "
        "spreads into **many small** ones. Soft thresholding shrinks every "
        "coefficient toward zero and drops the small ones, removing grain while "
        "keeping edges. The denoiser is **not told** the noise level: it "
        "estimates it from the noisy image, as a real scanner would have to."
    )

    denoise_on = st.checkbox("Denoise the noisy image", value=True, key="denoise_on")
    strength = st.slider(
        "Denoising strength", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
        key="denoise_strength", disabled=not denoise_on,
        help="Scales the estimated noise level that sets the thresholds. 1.0 is "
             "the textbook setting; higher removes more noise and more detail.",
    )

    if denoise_on:
        estimated_sigma = denoise.estimate_noise_sigma(noisy_recon)
        denoised = denoise.wavelet_denoise(noisy_recon, strength=strength)
        removed = denoise.residual(noisy_recon, denoised)

        # The true image-domain noise level, which only a simulator can know.
        # numpy's ifft2 divides by N, so white noise of per-channel sigma in
        # k-space becomes per-channel sigma / sqrt(N) in the image.
        true_sigma = noise.noise_sigma_for_snr(full_kspace, demo_snr) / np.sqrt(
            full_kspace.size
        )

        ui.figure(
            [
                ui.panel(noisy_recon, "Noisy, fully sampled"),
                ui.panel(denoised, f"Wavelet denoised (strength {strength:.1f})"),
                ui.panel(removed, "Removed: noisy − denoised", stretch=True),
            ],
            caption=(
                "Wavelet soft-thresholding (BayesShrink) applied to "
                "(a). Panel (c) is contrast-stretched with zero at mid-grey. "
                "Featureless grain means only noise was removed; visible anatomy "
                "means real detail was removed too."
            ),
            number="3b",
        )

        noisy_scores = metrics.compute_metrics(image, noisy_recon)
        denoised_scores = metrics.compute_metrics(image, denoised)
        left, right = st.columns(2, gap="medium")
        with left:
            ui.metrics_table(noisy_scores, caption="Noisy")
        with right:
            ui.metrics_table(
                denoised_scores, baseline=noisy_scores,
                caption="Denoised (change vs noisy)",
            )

        ui.note(
            f"Noise level: estimated σ = {estimated_sigma:.4f} from the image alone; "
            f"true σ = {true_sigma:.4f} (known only because this is a simulation)."
        )

        if denoised_scores["psnr"] < noisy_scores["psnr"]:
            ui.remark(
                "Denoising made this image *worse*. At high SNR there is little "
                "noise to remove, and fine texture in the finest wavelet band can "
                "be mistaken for noise, so the estimate runs high and real detail "
                "is smoothed away. Look for anatomy in panel (c), or lower the "
                "strength.",
                kind="caution",
            )
        else:
            ui.remark(
                "Denoising helps most when noise is heavy. Push the strength past "
                "~1.5 and PSNR usually falls again: the thresholds start eating "
                "real edges and texture. There is no free lunch, only a trade-off "
                "between grain and blur.",
                kind="result",
            )

# ---------------------------------------------------------------------------
# Tab 4: compressed sensing
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Compressed sensing: a better guess at the missing data")
    ui.lede(
        "Zero-filling assumes every unmeasured point was zero. Compressed "
        "sensing instead asks: *of all the images consistent with what we "
        "measured, which one is the sparsest in a wavelet basis?* It needs the "
        "**random variable-density** mask, because its artifacts are incoherent. "
        "Cartesian ghosts are just as sparse as real anatomy, so no sparsity "
        "prior can tell them apart."
    )

    controls = st.columns(3, gap="large")
    cs_ratio = controls[0].slider(
        "k-space sampled", min_value=0.03, max_value=0.5, value=0.125, step=0.005,
        format="%.3f", key="cs_ratio",
    )
    cs_lambda = controls[1].slider(
        "λ, sparsity strength", min_value=0.002, max_value=0.10, value=0.01,
        step=0.002, format="%.3f",
        help="Larger = sparser = smoother. Too large and real anatomy is "
             "thresholded away.",
    )
    cs_iters = controls[2].slider(
        "Iterations", min_value=10, max_value=150, value=80, step=10,
    )

    with st.spinner("Running FISTA..."):
        result = run_cs(sample_id, cs_ratio, cs_lambda, cs_iters, snr_db, int(seed))

    cs_sampled = ks.sampling_ratio(result["mask"])
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.figure(
            [
                ui.panel(result["mask"], f"Mask, {cs_sampled * 100:.1f}%"),
                ui.panel(result["zero_fill_image"], "Zero-filled (linear, instant)"),
                ui.panel(result["cs_image"], "Compressed sensing (FISTA)"),
            ],
            caption=(
                f"Variable-density sampling at R = {1 / cs_sampled:.1f}, "
                f"λ = {cs_lambda:.3f}, {cs_iters} iterations. (b) and (c) use "
                "exactly the same measurements."
            ),
            number="4",
        )
    with right:
        z, c = result["zero_fill_metrics"], result["cs_metrics"]
        rows = [
            ("PSNR", ui.format_psnr(z["psnr"]), ui.format_psnr(c["psnr"])),
            ("SSIM", f"{z['ssim']:.4f}", f"{c['ssim']:.4f}"),
        ]
        ui.table(
            ["Metric", "Zero-filled", "Compressed sensing"], rows,
            caption="Zero-filling vs compressed sensing", number="4",
        )

        delta_psnr = c["psnr"] - z["psnr"]
        delta_ssim = c["ssim"] - z["ssim"]
        if delta_psnr > 0:
            ui.remark(
                f"CS wins by **{delta_psnr:+.2f} dB** PSNR and **{delta_ssim:+.4f}** "
                "SSIM from exactly the same measurements.",
                kind="result",
            )
        else:
            ui.remark(
                f"At this ratio CS trades **{delta_psnr:.2f} dB** of PSNR for "
                f"**{delta_ssim:+.4f}** SSIM. That is the expected behaviour when "
                "undersampling is mild: zero-filling is already close to perfect, so "
                "the sparsity prior costs more than it gains. Push the ratio below "
                "~0.15 and CS pulls ahead on both.",
                kind="caution",
            )

    history = pd.DataFrame(result["history"])
    if "psnr" in history:
        st.markdown("#### Convergence")
        ui.line_chart(
            history, x="iteration", y="psnr",
            x_title="FISTA iteration", y_title="PSNR (dB)",
            reference=(z["psnr"], f"zero-fill baseline, {z['psnr']:.2f} dB"),
            points=False, height=300,
        )
        ui.caption(
            "PSNR of the compressed-sensing estimate at each iteration. The "
            "dashed line is the zero-filled reconstruction.",
            number="5",
        )

# ---------------------------------------------------------------------------
# Tab 5: the sweep
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Quality versus acceleration")
    ui.lede(
        "Every strategy, every ratio, scored against the ground truth. This is "
        "the quantitative version of tab 1."
    )

    with st.spinner("Sweeping..."):
        table = sweep(sample_id, snr_db, int(seed))

    order = [SHORT_LABELS[kind] for kind in ks.ACQUISITION_MASKS]
    left, right = st.columns(2, gap="large")
    with left:
        ui.line_chart(
            sweep_long(table, "PSNR (dB)"), x="sampling %", y="PSNR (dB)",
            series="strategy", series_order=order,
            x_title="k-space sampled (%)", y_title="PSNR (dB)", log2_x=True,
        )
    with right:
        ui.line_chart(
            sweep_long(table, "SSIM"), x="sampling %", y="SSIM",
            series="strategy", series_order=order,
            x_title="k-space sampled (%)", y_title="SSIM", log2_x=True,
        )
    ui.caption(
        "Reconstruction quality against the fraction of k-space acquired, on a "
        "log₂ axis so each step is a doubling of acceleration. Left: PSNR. "
        "Right: SSIM. The fully sampled point is omitted (PSNR is infinite).",
        number="6",
    )

    ui.dataframe_table(
        table,
        {
            "sampling %": "{:.1f}",
            "acceleration": "{:.1f}×",
            "PSNR (dB)": "{:.2f}",
            "SSIM": "{:.4f}",
        },
        caption="All strategies at every sampling ratio",
        number="5",
    )
    st.download_button(
        "Download as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_id}_metrics.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 6: motion
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Patient motion during the scan")
    ui.lede(
        "Moving the patient does **not** change the magnitude of k-space. It "
        "stamps a linear **phase ramp** on it. A *uniform* shift is therefore "
        "harmless: the image simply moves. The artifact comes from the patient "
        "being in a **different place for different parts of the scan**, so the "
        "measured k-space is not the transform of any one consistent object."
    )

    controls = st.columns(4, gap="medium")
    motion_model = controls[0].selectbox(
        "Motion model",
        ["none", "sudden_jerk", "slow_drift", "periodic"],
        index=1,
        format_func={
            "none": "None",
            "sudden_jerk": "Sudden jerk (ghost)",
            "slow_drift": "Slow drift (blur)",
            "periodic": "Periodic breathing (ghost train)",
        }.get,
    )
    motion_amp = controls[1].slider(
        "Amplitude (pixels)", min_value=0.0, max_value=25.0, value=8.0, step=0.5,
        help="How far the patient moves, in image pixels.",
    )
    jerk_at = controls[2].slider(
        "Jerk at (fraction of scan)", min_value=0.0, max_value=1.0, value=0.5,
        step=0.05, disabled=motion_model != "sudden_jerk",
    )
    motion_cycles = controls[3].slider(
        "Cycles over the scan", min_value=0.5, max_value=20.0, value=6.0, step=0.5,
        disabled=motion_model != "periodic",
        help="How many breathing cycles fit in one scan; this sets the ghost spacing.",
    )

    mask_m, acquired_m, recon_m, scores_m, displacements = acquire_with_motion(
        sample_id, strategy, ratio, motion_model, motion_amp, jerk_at,
        motion_cycles, snr_db, int(seed),
    )

    ui.figure(
        [
            ui.panel(image, "Ground truth"),
            ui.panel(reconstruction, "No motion"),
            ui.panel(recon_m, "With motion"),
            error_panel(image, recon_m, "Motion error"),
        ],
        caption=(
            f"{strategy_label} acquisition at {ratio * 100:.0f}% of k-space, "
            f"without and with {motion_amp:.1f} px of patient motion. (d) is the "
            "absolute error of (c)."
        ),
        number="7",
    )

    columns = st.columns(4, gap="medium")
    with columns[1]:
        ui.metrics_table(scores)
    with columns[2]:
        ui.metrics_table(scores_m, baseline=scores)

    left, right = st.columns([3, 2], gap="large")
    with left:
        axis = "Spoke index" if strategy == "radial" else "k-space row"
        ui.line_chart(
            pd.DataFrame({
                "event": np.arange(len(displacements)),
                "dy": [dy for dy, _ in displacements],
            }),
            x="event", y="dy", x_title=f"{axis} (acquisition time)",
            y_title="dy (pixels)", points=False, height=260,
        )
        ui.caption(
            "Where the patient was over the course of the scan, earliest on the left.",
            number="8",
        )
    with right:
        ui.remark(
            {
                "none": "This is the same reconstruction as tab 1. Pick a model "
                        "above to corrupt it.",
                "sudden_jerk": "The rows before the jump and the rows after it "
                               "each describe a perfectly sharp object, just two "
                               "objects offset by the amplitude. The result is a "
                               "superposition of two sharp copies: a **discrete "
                               "ghost**, not a smear.",
                "slow_drift": "Every row disagrees slightly with its neighbours "
                              "instead of splitting into two consistent blocks, so "
                              "the inconsistency spreads continuously across "
                              "k-space and reads as **blur** rather than a second "
                              "copy.",
                "periodic": "A sinusoidal phase error is equivalent to convolving "
                            "the image with a pair of offset deltas, giving a "
                            "**regular train of ghosts** along the phase-encode "
                            "axis. Raise the cycle count to push the ghosts "
                            "further apart.",
            }[motion_model],
            lead={
                "none": "No motion.",
                "sudden_jerk": "Sudden jerk.",
                "slow_drift": "Slow drift.",
                "periodic": "Periodic motion.",
            }[motion_model],
        )
        if strategy != "radial":
            ui.remark(
                "Acquisition time is modelled as the k-space row index across "
                "all rows, but an undersampled Cartesian scan only acquires the rows "
                "the mask keeps. The artifact character is unaffected (unsampled rows "
                "are zeroed anyway), but the jerk position above is nominal, not exact.",
                kind="note",
            )

    # --- Motion correction ----------------------------------------------------
    # See mri_sim/motion_correction.py. Motion only changed PHASE, so undoing
    # it is exact once the displacements are known; the question is how to
    # know them.
    ui.rule()
    st.subheader("Correcting the motion")
    ui.lede(
        "Motion only rotated the **phase** of each row or spoke, so nothing was "
        "destroyed: multiplying by the opposite phase ramp puts every sample "
        "back. The hard part is knowing the motion. **(b)** uses the true "
        "displacements, as a scanner's navigator echoes or optical tracker would "
        "supply. **(c)** uses **autofocus**: it sees only the corrupted data, "
        "tries candidate motions of the chosen type, and keeps the one that "
        "makes the image sharpest (lowest gradient entropy)."
    )

    if motion_model == "none" or motion_amp == 0.0:
        ui.note("Pick a motion model and a non-zero amplitude above to correct it.")
    else:
        undo = motion_undo(strategy, image.shape, ratio)
        oracle_recon = ks.from_kspace(undo(acquired_m, displacements))
        oracle_scores = metrics.compute_metrics(image, oracle_recon)

        autofocus_key = (
            sample_id, strategy, ratio, motion_model, motion_amp, jerk_at,
            motion_cycles, snr_db, int(seed),
        )
        requested = st.session_state.setdefault("autofocus_requested", set())
        if st.button(
            "Run autofocus",
            help="Estimates the motion from the corrupted scan alone. Takes "
                 "0.2 s (drift) to ~5 s (periodic, radial); results are "
                 "cached, so returning to the same settings is instant.",
        ):
            requested.add(autofocus_key)

        autofocus = None
        if autofocus_key in requested:
            with st.spinner("Searching for the motion that makes the image sharpest..."):
                autofocus, auto_recon, auto_scores = run_autofocus(*autofocus_key)

        if autofocus is None:
            panels = [
                ui.panel(recon_m, "With motion"),
                ui.panel(oracle_recon, "Corrected, true motion (oracle)"),
            ]
            caption = (
                "Motion correction. (b) undoes the true displacements exactly. "
                "Press **Run autofocus** to estimate them from (a) alone."
            )
        else:
            panels = [
                ui.panel(recon_m, "With motion"),
                ui.panel(oracle_recon, "Corrected, true motion (oracle)"),
                ui.panel(auto_recon, "Corrected, estimated motion (autofocus)"),
                error_panel(image, auto_recon, "Autofocus error"),
            ]
            caption = (
                "Motion correction. (b) undoes the true displacements; (c) undoes "
                "the displacements autofocus estimated from (a) alone; (d) is the "
                "absolute error of (c). With undersampling or noise, the best any "
                "correction can reach is the no-motion reconstruction of Figure 7(b)."
            )
        ui.figure(panels, caption=caption, number="8b")

        columns = st.columns(4, gap="medium")
        with columns[0]:
            ui.metrics_table(scores_m, caption="With motion")
        with columns[1]:
            ui.metrics_table(oracle_scores, baseline=scores_m, caption="Oracle (vs motion)")
        if autofocus is not None:
            with columns[2]:
                ui.metrics_table(auto_scores, baseline=scores_m, caption="Autofocus (vs motion)")

        if autofocus is not None:
            n_events = len(displacements)
            axis = "Spoke index" if strategy == "radial" else "k-space row"
            left, right = st.columns([3, 2], gap="large")
            with left:
                ui.line_chart(
                    pd.DataFrame({
                        "event": np.tile(np.arange(n_events), 2),
                        "dy": [dy for dy, _ in displacements]
                              + [dy for dy, _ in autofocus.displacements],
                        "series": ["True motion"] * n_events
                                  + ["Autofocus estimate"] * n_events,
                    }),
                    x="event", y="dy", series="series",
                    x_title=f"{axis} (acquisition time)", y_title="dy (pixels)",
                    points=False, height=260,
                )
                ui.caption(
                    "True patient position against the position autofocus "
                    "inferred from the corrupted data.",
                    number="8c",
                )
            with right:
                truth = {"amp": f"{motion_amp:g} px"}
                if motion_model == "sudden_jerk":
                    truth["at"] = f"event {int(round(jerk_at * n_events))}"
                if motion_model == "periodic":
                    truth["cycles"] = f"{motion_cycles:g}"
                found = autofocus.params
                units = {"amp": " px", "cycles": ""}
                ui.table(
                    ["Parameter", "True", "Estimated"],
                    [
                        (name, truth[name],
                         f"event {found[name]}" if name == "at"
                         else f"{found[name]:g}{units[name]}")
                        for name in truth
                    ],
                    caption=(
                        f"{autofocus.n_evaluations} candidate images tried; "
                        f"gradient entropy {autofocus.initial_score:.3f} → "
                        f"{autofocus.score:.3f}"
                    ),
                )
                ui.remark(
                    "Autofocus is told the motion **type** but not its size or "
                    "timing. A shift of the whole scan is invisible (the shift "
                    "theorem again), so only motion *relative* to the start of "
                    "the scan can be recovered. Very large, fast breathing "
                    "(~25 px at ~20 cycles) can fall between the search's grid "
                    "points; the oracle still fixes it.",
                    kind="note",
                )

    with st.expander("Cartesian ghosts vs radial streaks", expanded=False):
        st.markdown(
            "The *same* motion, put through both trajectories at the same "
            "sampling ratio. Collapsed by default because the radial pipeline "
            "has to search for its spoke count the first time you open it.\n\n"
            "Look at the **character** of the corruption, not the score: "
            "Cartesian motion produces coherent ghosts, sharp anatomy-shaped "
            "copies that a radiologist can mistake for structure, while radial "
            "spreads the same error into incoherent streaks."
        )
        compared = {
            kind: acquire_with_motion(
                sample_id, kind, ratio, motion_model, motion_amp,
                jerk_at, motion_cycles, snr_db, int(seed),
            )
            for kind in ("cartesian", "radial")
        }
        left, right = st.columns(2, gap="large")
        for column, kind, letter in ((left, "cartesian", "a"), (right, "radial", "b")):
            _, _, compare_recon, compare_scores, _ = compared[kind]
            with column:
                ui.figure([ui.panel(compare_recon, ks.MASK_LABELS[kind])], first_letter=letter)
                ui.metrics_table(compare_scores)
        ui.remark(
            "**Radial will usually score *worse* here, and that is a limitation "
            "of this simulator rather than a fact about radial MRI.** Real "
            "radial acquisition is motion-robust because every spoke "
            "re-measures the k-space centre, and those redundant measurements "
            "*average*, so the motion errors partly cancel. This model rasterizes "
            "spokes onto the Cartesian grid and gives each grid point a single "
            "owning spoke, so no averaging happens: the centre becomes a "
            "patchwork of many different phase errors instead of one averaged "
            "value. That is worse than the Cartesian centre block, where a "
            "contiguous run of rows mostly shares one patient position. "
            "Reproducing the real advantage needs a gridding/NUFFT "
            "reconstruction that accumulates every spoke crossing a point.",
            kind="caution",
            lead="Limitation.",
        )


# ---------------------------------------------------------------------------
# Tab 7: reduced field of view
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Scan only the part that matters")
    ui.lede(
        "*“We only care about the pituitary, or this one lesion. Can we scan "
        "just that bit and finish in a fraction of the time?”* Yes, but not "
        "the way almost everyone first guesses, and the gap between the wrong "
        "guess and the right answer is the whole lesson."
    )

    st.markdown("### 1. The wrong answer: keep the part of k-space where the target is")
    ui.lede(
        "The lesion is in the top-left of the image, so keep the top-left of "
        "k-space? Delete one quadrant of k-space and watch **where** "
        "in the image the damage lands."
    )

    quadrant = st.radio(
        "Quadrant of k-space to delete",
        list(roi.QUADRANTS),
        horizontal=True,
        key="roi_quadrant",
    )
    locality = roi_locality(sample_id, quadrant)

    left, right = st.columns([3, 1], gap="large")
    with left:
        ui.figure(
            [
                ui.panel(image, "Ground truth"),
                kspace_panel(locality["kspace_damaged"], f"k-space, {quadrant} deleted"),
                ui.panel(locality["reconstruction"], "Reconstruction"),
                error_panel(image, locality["reconstruction"], "Where the error landed"),
            ],
            caption=(
                f"Deleting the {quadrant} quadrant of k-space. (d) is the absolute "
                "error of (c), spread across the whole image."
            ),
            number="9",
        )
    with right:
        errors = locality["quadrant_errors"]
        ui.dataframe_table(
            pd.DataFrame(
                errors,
                index=["Top half", "Bottom half"],
                columns=["Left half", "Right half"],
            ),
            {"Left half": "{:.4f}", "Right half": "{:.4f}"},
            caption="Mean absolute error per image quadrant",
            number="6",
            index=True,
        )

    spread = errors.max() / errors.min()
    ui.remark(
        f"The largest quadrant error is only **{spread:.1f}×** the smallest, "
        "nowhere near the total wipeout in one cell that the guess predicts. "
        f"Deleting the {quadrant} of k-space damaged the **entire image**, roughly "
        "evenly. k-space is **not spatially local**: every sample is an inner "
        "product of the *whole* slice with one global sinusoid, so every sample "
        "carries information about every pixel. Position lives in the **phase "
        "relationships between** samples, not in where the samples sit.",
        kind="caution",
    )

    ui.rule()
    st.markdown("### 2. The right answer: shrink the FOV, not the k-space region")

    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.lede(
            "Two *independent* Fourier relationships govern a Cartesian scan, "
            "and the entire method is about keeping them apart:"
        )
        ui.table(
            ["Knob", "What it controls"],
            [
                ("<code>dk</code>, spacing between samples", "<strong>FOV</strong> = 1 / dk"),
                ("<code>k_max</code>, how far out you sample",
                 "<strong>resolution</strong> = 1 / (2·k_max)"),
            ],
            numeric=[False, False],
        )
        ui.lede(
            "Tab 2's centre-only sampling shrinks `k_max`, the **wrong knob**: "
            "it buys speed with resolution. Reduced-FOV turns the other one: a "
            "spatially selective RF pulse excites *only a box* around the "
            "target, so the object itself is now R times smaller, so samples "
            "may sit R times further apart without it folding onto itself. "
            "`k_max` never changes, so **resolution never changes**."
        )
    with right:
        ui.remark(
            "In this simulator the RF pulse is emulated by multiplying the "
            "image by a box **before** the forward FFT. That is a model, not a "
            "cheat: spatially restricting the excitation is exactly what the "
            "physical pulse does, and everything after it (FFT, decimation, "
            "inverse FFT) is the real pipeline.",
            kind="note",
            lead="Modelling note.",
        )

    controls = st.columns([1, 2, 2], gap="large")
    legal_R = roi.reduction_factors(image.shape)
    R = controls[0].selectbox(
        "Reduction factor R",
        legal_R,
        index=legal_R.index(roi.DEFAULT_REDUCTION) if roi.DEFAULT_REDUCTION in legal_R else 0,
        help="Must divide the image size exactly, or the aliasing period is "
             "fractional and the method quietly degrades.",
    )

    use_tumor = False
    if tumor_mask is not None:
        use_tumor = controls[1].checkbox(
            "Centre the box on the expert tumour mask", value=True,
            help="Uses roi_center_from_mask() on the segmentation shipped with "
                 "this sample.",
        )

    if use_tumor:
        center = roi.roi_center_from_mask(tumor_mask)
        with controls[1]:
            ui.note(f"Tumour centroid: row {center[0]}, col {center[1]}.")
    else:
        center_y = controls[1].slider(
            "ROI centre, row", 0, image.shape[0] - 1, image.shape[0] // 2,
            key="roi_cy",
        )
        center_x = controls[2].slider(
            "ROI centre, column", 0, image.shape[1] - 1, image.shape[1] // 2,
            key="roi_cx",
        )
        center = (center_y, center_x)

    comparison = roi_compare(sample_id, center, int(R), int(seed))
    box = comparison["box"]
    method = comparison["variants"][0]
    compact = roi.compact_reconstruct(method["kspace"], int(R), box)

    ui.figure(
        [
            ui.panel(box_overlay(image, box), "Where the box goes"),
            ui.panel(method["object"], "After the RF pulse"),
            ui.panel(method["mask"], f"Every {R}th sample"),
            ui.panel(method["reconstruction"], "Reconstructed, full grid"),
            ui.panel(compact, f"What the scanner returns ({box.size}×{box.size})"),
        ],
        caption=(
            f"Reduced-FOV acquisition with a {box.size}×{box.size} px box at rows "
            f"{box.y0}–{box.y0 + box.size}, cols {box.x0}–{box.x0 + box.size}. "
            f"Sampling every {R}th point on both axes keeps {100.0 / (R * R):.2f}% "
            f"of k-space, a **{R * R}× faster** scan at full resolution inside "
            "the box."
        ),
        number="10",
    )

    left, right = st.columns([1, 2], gap="large")
    with left:
        ui.metrics_table(
            {"psnr": method["psnr"], "ssim": method["ssim"]},
            caption="Scored inside the box only",
        )
    with right:
        ui.remark(
            "Whole-image metrics are meaningless here, because nothing outside "
            "the box was excited, so there is no ground truth out there to get "
            "wrong. Panel (d) shows the periodic replicas of the ROI filling the "
            "rest of the FOV; that is expected and harmless. Panel (e) is the "
            "real output: a smaller field of view at **the same resolution**, "
            "which is all that was ever measured.",
            kind="result",
        )

    ui.rule()
    st.markdown("### 3. Four ways to spend the same number of samples")
    ui.lede(
        f"All four acquisitions below use about **{100.0 / (R * R):.2f}%** of "
        "k-space, the same scan time. They differ only in the idea behind "
        "how to spend it. Every one is scored inside the ROI."
    )

    ui.figure(
        [ui.panel(variant["reconstruction"], variant["label"])
         for variant in comparison["variants"]],
        caption=" ".join(
            f"({chr(ord('a') + i)}) "
            + variant["note"].replace(" -> ", " → ").replace(" -- ", ": ") + "."
            for i, variant in enumerate(comparison["variants"])
        ),
        number="11",
    )

    ui.table(
        ["Strategy", "k-space used", "Acceleration", "PSNR in ROI", "SSIM in ROI"],
        [
            (
                ui.inline(variant["label"]),
                f"{variant['ratio'] * 100.0:.2f}%",
                f"{variant['acceleration']:.1f}×",
                ui.format_psnr(variant["psnr"]),
                f"{variant['ssim']:.4f}",
            )
            for variant in comparison["variants"]
        ],
        caption="Four strategies at the same scan time",
        number="7",
    )

    by_key = {variant["key"]: variant for variant in comparison["variants"]}
    no_supp_psnr = by_key["no_suppression"]["psnr"]
    rfov_psnr = by_key["reduced_fov"]["psnr"]
    rfov_text = "exact" if not np.isfinite(rfov_psnr) else f"{rfov_psnr:.1f} dB"
    # The sign of the no-suppression score depends on R -- at R=2 only three
    # replicas fold in and it still scrapes a positive PSNR -- so the sentence
    # about it has to follow the number rather than assert one.
    verdict = (
        "A **negative** PSNR means the error is literally larger than the signal."
        if no_supp_psnr < 0 else
        f"Even at R={R}, where only {R * R - 1} replicas fold in, that is a "
        "useless image."
    )
    ui.remark(
        "`no_suppression` uses the *identical* samples as `reduced_fov`; the only "
        "difference is that the RF pulse was skipped, so the rest of the head is "
        f"still producing signal and folds {R * R - 1} other pieces of the anatomy "
        f"directly on top of the ROI. It scores **{no_supp_psnr:.1f} dB** against "
        f"**{rfov_text}**. {verdict} The excitation is not an optimisation on top "
        "of the method. **It is the method.**",
        kind="caution",
        lead="The last row is the headline.",
    )
    ui.remark(
        "The two middle rows are not catastrophic, just mediocre; they are the "
        "real competitors. Reduced-FOV beats them by hundreds of dB because it "
        "is not approximating anything: inside the box it is a complete, "
        "critically-sampled measurement. This is the one tab in the app where "
        "you do not trade quality for speed. The catch is that you only get "
        "the box, and you have to know where to aim it before the scan starts."
    )


# ---------------------------------------------------------------------------
# Tab 8: provenance
# ---------------------------------------------------------------------------

with tabs[7]:
    st.subheader(meta["title"])

    left, right = st.columns([1, 2], gap="large")
    with left:
        panels = [ui.panel(image, "Ground truth")]
        if tumor_mask is not None:
            # Red overlay on the expert tumour mask.
            overlay = np.stack([image] * 3, axis=-1)
            overlay[tumor_mask] = [1.0, 0.25, 0.25]
            panels.append(ui.panel(overlay, "Expert tumour mask"))
        ui.figure(panels, columns=1 if len(panels) == 1 else 2)

    with right:
        ui.table(
            ["Field", "Value"],
            [
                ("Collection", ui.inline(f"`{meta['collection']}`")),
                ("Source file", ui.inline(f"`{meta['source_file']}`")),
                ("Stored shape", f"{meta['shape'][0]}×{meta['shape'][1]}"),
                ("Tags", ui.inline(", ".join(meta["tags"]))),
            ],
            caption="Provenance",
            number="8",
            numeric=[False, False],
        )
        ui.note(meta["collection_note"])

        acquisition = {k: v for k, v in meta["acquisition"].items() if v not in (None, "")}
        ui.table(
            ["Parameter", "Value"],
            [(key.replace("_", " ").capitalize(), ui.inline(str(value)))
             for key, value in acquisition.items()],
            caption="Acquisition",
            number="9",
            numeric=[False, False],
        )

        stats = meta["stats"]
        ui.table(
            ["Statistic", "Value"],
            [
                ("Energy inside the central 10% radius",
                 f"{stats['energy_within_r0.1'] * 100:.1f}%"),
                ("Energy inside the central 25% radius",
                 f"{stats['energy_within_r0.25'] * 100:.1f}%"),
                ("k-space dynamic range (peak / median magnitude)",
                 f"{stats['kspace_dynamic_range_db']:.0f} dB"),
                ("Hermitian asymmetry (≈0 for a real-valued image)",
                 f"{stats['hermitian_asymmetry']:.2f}"),
            ],
            caption="Derived statistics",
            number="10",
        )
        ui.note(
            "The dynamic range is why k-space is always shown on a log scale. An "
            "asymmetry near 0 would mean half of k-space is redundant."
        )

    with st.expander("How this k-space was made, and what is simulated"):
        st.markdown(
            "This project starts from reconstructed images and runs a **forward "
            "FFT** to manufacture k-space. A real scanner measures k-space "
            "directly; everything downstream of that point behaves identically.\n\n"
            "The source files are *magnitude* images, so the original scanner "
            "phase no longer exists. A smooth synthetic phase map (a B₀-like "
            "quadratic bowl, a gradient ramp, and coil-like ripples) is applied "
            "before the FFT. Without it the k-space would be perfectly "
            "Hermitian-symmetric, half the data would be a free copy of the "
            "other half, and any partial-Fourier demonstration would be "
            "unrealistically perfect.\n\n"
            f"Stored as: `kspace = fftshift(fft2(image · exp(i·phase)))`, "
            f"complex64, {meta['shape'][0]}×{meta['shape'][1]}."
        )
