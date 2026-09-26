# Presentation Runbook

**For the CSE 220 project defence.**
This covers what to do and say. The physics explanation is in
[PROJECT_BRIEF.md](PROJECT_BRIEF.md).

---

## 0. Before the presentation: checklist

| Check            | Command                              | Expected                                                                                          |
| ---------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Free space on C: | -                                    | Streamlit and matplotlib both write to `%TEMP%` on C:, so the demo can fail if the drive is full. |
| App launches     | `streamlit run app/streamlit_app.py` | Opens `http://localhost:8501`                                                                     |
| Store is present | `python -m kspace_store.demo --list` | 40 samples listed                                                                                 |
| Figures exist    | `python main.py --center-edges --cs` | Writes 15 PNGs to `outputs/`                                                                      |

Run `python main.py --center-edges --cs` before the presentation. It takes a
couple of minutes and produces every figure needed as a fallback if the live app
fails. Keep the `outputs/` folder open in a second window.

---

## 1. Opening

> An MRI scanner doesn't measure an image. It measures the image's 2-D Fourier
> transform, one sample at a time, and scan time is proportional to how many
> samples it collects. This project shows what goes wrong when the scan is sped
> up by measuring less.

Then the consequence:

- Scan time is proportional to the number of k-space samples.
- The inverse FFT needs _all_ of k-space to be exact.
- So measuring less finishes sooner but gives a worse image. **How much worse, and can the skipped samples be chosen well?**

---

## 2. Demo flow (8-10 minutes)

Use the Streamlit app and go through the six tabs in order.

### Tab 1: Acquire

Pick a brain tumour sample. Start at 100% sampling, then drag the ratio slider down.

**Say:** "At 100% the reconstruction is exact. PSNR is about 320 dB, which is
just float64 round-off, not a real measurement. Now watch what happens as I drop
samples."

Switch strategies at a fixed 12.5% and compare the artifacts:

- **Cartesian**: sharp _ghost copies_ of the anatomy, evenly spaced
- **Radial**: _streaks_ radiating outward
- **Variable density**: _noise-like grain_, anatomy still readable

Same amount of data, three different kinds of failure. This is the main visual
result of the project.

### Tab 2: Centre vs edges

Shows why the masks are designed the way they are. Keep only the centre 10% of
k-space, then the same number of samples from the outer edge only.

**Say:** "Centre only: right shape, right contrast, blurry. Edges only: the
anatomy vanishes and only outlines remain. The centre acts as a low-pass filter
and the edges as a high-pass filter. That's why every mask I build protects the
centre."

Support it with the measured number: **92.6% of the k-space energy sits inside
the central 10% radius, which is about 1% of the samples.**

### Tab 3: Noise

Lower the SNR slider and show the grain appear.

**Say:** "The noise is added in k-space, at the moment of measurement, not to
the finished image. And only on the points we actually measured. An unsampled
point isn't a noisy zero; it was never measured at all."

### Tab 4: Compressed sensing

Same mask, same samples, two different reconstructions.

**Say:** "Zero-filling assumes everything we didn't measure was zero. That
assumption is wrong; we just didn't measure those points. Compressed sensing
instead asks: of all the images consistent with what we _did_ measure, which is
the simplest?"

**Result at 12.5% sampling (8x acceleration):**

|            | PSNR         | SSIM      |
| ---------- | ------------ | --------- |
| Zero-fill  | 25.81 dB     | 0.478     |
| CS (FISTA) | 27.26 dB     | **0.785** |
| **Gain**   | **+1.45 dB** | **+0.31** |

Point at the SSIM: "PSNR went up by 1.45 dB, but SSIM went up by 0.31, so the
structure came back. This is why both metrics are reported."

### Tab 5: Sweep

The summary chart: one line per strategy, PSNR and SSIM vs sampling ratio.

### Tab 6: About this sample

Shows that this is real clinical data, with the source file recorded for each
sample. This answers "where did this data come from?"

---

## 3. Numbers to know

Full sweep on the Shepp-Logan phantom at 256x256 (`python main.py`):

| Strategy             | 50%                  | 25%               | 12.5% (8x)        |
| -------------------- | -------------------- | ----------------- | ----------------- |
| Cartesian            | 26.81 dB / 0.683     | 20.54 / 0.549     | 17.43 / 0.535     |
| Radial               | 29.64 dB / 0.476     | 23.29 / 0.312     | 19.65 / 0.276     |
| **Variable density** | **31.58 dB / 0.823** | **28.86 / 0.722** | **25.81 / 0.478** |

**Variable density has the highest PSNR at every ratio**, and the highest SSIM
at 50% and 25%. At 12.5%, Cartesian has a slightly higher SSIM (0.535 vs 0.478).

### Where the two metrics disagree

At 50%, **radial has higher PSNR than Cartesian (29.64 vs 26.81) but much lower
SSIM (0.476 vs 0.683)**.

It is worth pointing this out before being asked. The explanation: PSNR is a
per-pixel error measure and ignores structure. Radial's streaks are low-energy
and spread thin, so they add little squared error, but they are _coherent
structure_ that wasn't in the original, and SSIM penalises that. This is why a
single metric isn't enough.

### Store facts

- 40 samples, 256x256, from 4 collections (brain tumours with expert masks, clinical pathologies, real Siemens spine DICOM, volumetric CT/MRI)
- k-space dynamic range: **62-85 dB** (mean 73), which is why k-space is shown on a log scale
- Energy inside central 10% radius: **92.6% mean** (min 74%, max 99%)

### Correctness checks you can run live

```bash
python -m mri_sim.motion      # Fourier shift theorem: max error 6.66e-16
python main.py                # FFT round-trip: max error ~5.6e-16
```

Both errors are at machine precision. If asked "how do you know it's right?",
run these.

---

## 4. Code tour (if asked to show the implementation)

The code is a package, not a single script:

| File                                         | What it does                                                               |
| -------------------------------------------- | -------------------------------------------------------------------------- |
| [mri_sim/kspace.py](mri_sim/kspace.py)       | **The core.** Forward/inverse FFT and all five masks. Open this one first. |
| [mri_sim/cs.py](mri_sim/cs.py)               | FISTA compressed sensing                                                   |
| [mri_sim/noise.py](mri_sim/noise.py)         | Complex Gaussian noise in k-space                                          |
| [mri_sim/motion.py](mri_sim/motion.py)       | Patient motion via the Fourier shift theorem                               |
| [mri_sim/metrics.py](mri_sim/metrics.py)     | PSNR / SSIM                                                                |
| [mri_sim/visualize.py](mri_sim/visualize.py) | All matplotlib output                                                      |
| [main.py](main.py)                           | Runs the whole sweep end to end                                            |

The three functions to have ready: `to_kspace`, `from_kspace`, `cartesian_mask`.

---

## 5. Likely questions

**"Why `fftshift`?"**
NumPy's `fft2` puts DC in the corner and wraps negative frequencies to the far
edges. `fftshift` moves DC to the middle, which is the MRI convention and makes
masks easier to write, since everything becomes "distance from the centre". It
only re-indexes the array; no information changes.

**"Why `ifftshift` and not `fftshift` on the way back?"**
They're identical for even-sized arrays but differ for odd sizes. Using the wrong
one gives a slightly shifted image.

**"Why take the magnitude?"**
The inverse FFT returns a complex image. With full data the imaginary part is
numerical noise, but once part of k-space is zeroed out the masked data is no
longer the transform of a real image, so there is a real imaginary component.
Real scanners have the same problem for physical reasons (B0 inhomogeneity, coil
phase) and solve it the same way: they display the magnitude image.

**"Why does random sampling beat regular skipping?"**
Regular skipping produces _coherent_ aliasing: sharp ghost copies that are just
as sparse as the real anatomy, so no sparsity-based method can tell them apart.
Random sampling spreads the error into _incoherent_ low-level grain, which a
sparsity-promoting denoiser can remove. Compressed sensing depends on this
difference.

**"How does this relate to Nyquist?"**
Skipping every Nth k-space line is undersampling in the Fourier domain, so the
aliasing appears in the _image_ domain and the image folds onto itself. That's
why Cartesian artifacts are evenly spaced copies: it is ordinary aliasing, with
the two domains swapped compared with the usual 1-D textbook case.

**"Is this real MRI data?"**
The images are real clinical scans (DICOM from a Siemens 1.5 T scanner, plus
public brain-tumour and pathology datasets). The **k-space is simulated**: I
start from reconstructed images and run a forward FFT, because raw scanner
k-space isn't available for these datasets. The store's README says this, and
the manifest records the exact source file for every sample.

**"Why is the phase synthetic?"**
Source files are magnitude images; the original scanner phase was discarded
before they were saved. A purely real image has exactly Hermitian k-space,
`K(-k) = conj(K(k))`, so half the data would be a copy of the other half and
any partial-Fourier demo would be unrealistically perfect. Adding a plausible
smooth phase map (B0 bowl + linear ramp + coil ripples) removes that symmetry.
Measured asymmetry is 1.23, so the data is not Hermitian.

**"What is CS doing, step by step?"**
Two alternating steps, 80 iterations:

1. **Data consistency**: put the measured samples back where they belong.
2. **Sparsify**: wavelet transform, soft-threshold small coefficients to zero, transform back.

Step 1 is a gradient step on the data-fidelity term (the FFT is unitary, so the
step size is exactly 1, with no tuning). Step 2 is the proximal operator of the
L1 norm. Alternating them is guaranteed to converge. FISTA adds Nesterov
momentum, which converges faster at the same cost per iteration.

**"Why does CS only help the random mask?"**
See the coherent/incoherent answer above. On a regular Cartesian mask the ghosts
are sparse too, so sparsity can't distinguish signal from artifact.

---

## 6. If the live demo breaks

1. **Don't debug during the presentation.** Switch to the pre-generated PNGs in `outputs/`.
2. `summary_metrics.png` and `mask_gallery.png` cover most of the results.
3. If Streamlit won't start, check free space on C: first.

---

## 7. Closing

> Undersampling isn't just "less data, worse image". _Which_ samples are skipped
> matters more than how many, and if they are skipped randomly, a non-linear
> reconstruction can recover most of the quality. That's compressed sensing, and
> it is one of the reasons modern MRI scans can be much shorter.
