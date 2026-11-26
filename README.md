<p align="center">
  <img src="My_app/ui/logo.svg" alt="PhaseMark logo" width="260">
</p>

# PhaseMark

**PhaseMark** is a Python-based GUI for evaluating the efficiency and validity of phase-reconstruction algorithms in **Digital Holographic Microscopy (DHM)**.  

It implements a **standardized protocol** for Quantitative Phase Imaging (QPI) evaluation, structured into four main modules (*cores*) that cover background stability, global phase distortions, ground-truth comparisons, and computational complexity.  

---

The protocol is organized into **four evaluation cores**:  

1. Background Metrics
2. Global Metrics  
3. Ground-Truth Comparisons  
4. Computational Complexity  

Additionally, a set of **test holograms** is included to benchmark performance across different types of samples.  

---

## Nucleo I – Residual Background Phase Variance  

Focuses on phase flatness in object-free regions of reconstructed phase images. Requires segmentation to isolate background regions.  

**Metrics:**  
- Standard deviation (STD)
- Mean Absolute Deviation (MAD)  
- Root Mean Square (RMS), reported with the background mean  
- Peak-to-Valley (P–V) Value in background  
- Full Width at Half Maximum (FWHM) of the phase histogram  
- Entropy of background phase map
- Phase jumps (percentage of pixel pairs still crossing a 2π wrap)  
- Background phase tilt/curvature residuals (Legendre coefficients)  

---

##  Nucleo II – Global Phase Distortion Metrics  

For samples without large empty regions. Evaluates distortions across the entire phase map.  

**Metrics:**  
- Peak-to-Valley (P–V)
- Standard deviation (STD) over the whole field
- PGM (Phase Gradient Metric, mean local gradient)  
- GPG (Global Phase Gradient, tilt of the least-squares plane)  
- TSM (Thresholding and Intensity Summation Metric)  
- Laplacian energy (LE)  
- Spatial frequency (SF)  
- Entropy of phase map  
- GSM (Generalized Sharpness Metric)  
- FWHM of the whole-field histogram  

---

##  Nucleo III – Ground-Truth Comparisons  

Provides benchmark holograms and reference-based evaluation.  

**Metrics:**  
- Mean Squared Error (MSE)
- SSIM (Structural Similarity Index)  
- pSNR (peak Signal-to-Noise Ratio)  

---

##  Nucleo IV – Computational Complexity  

Estimates computational cost of each algorithm:  
- Operation counts  
- Execution time and memory profiling  

---

## Samples  

Bundled samples are available from the **Samples** menu:

```
Samples
├── Holograms
│   ├── Telecentric
│   └── Non-telecentric
└── Reconstruction Samples
```

### Holograms

The repository includes a diverse set of test holograms to ensure robust evaluation, available from **Samples → Holograms**. They are **holograms, not phase maps**: they are meant to be reconstructed and compensated with the algorithm under test, and the resulting phase map is then evaluated in PhaseMark.  
Included sample holograms correspond to calibration patterns (USAF and Star Target) and biological samples. File name for each hologram indicates the parameters used to record it:

**Telecentric/T_Usaf_20x_632_3.75.bmp** — USAF pattern, 20x objective, 632 nm wavelength, 3.75 µm pixel size.<br>
**Telecentric/T_Star_10x_632_3.75.bmp** — Star Target pattern, 10x objective, 632 nm wavelength, 3.75 µm pixel size.<br>
**Telecentric/T_Probiotics_20x_632_3.75.bmp** — Biological sample (probiotics), 20x objective, 632 nm wavelength, 3.75 µm pixel size.<br>
**No Telecentric/N_Star_20x_532_5.86_-4cm.tiff** — Star Target pattern, 20x objective, 532 nm wavelength, 5.86 µm pixel size, -4 cm defocus.<br>
**No Telecentric/NredBlood_40x_632_4.65_30mm.tiff** — Biological sample (red blood cells), 40x objective, 632 nm wavelength, 4.65 µm pixel size, 30 mm defocus.<br>

Naming convention: [Prefix]_[Sample]_[Objective]_[Wavelength in nm]_[Pixel size in µm]_[Additional parameters].[extension]

Where:

**Prefix:** indicates the hologram type (T = Telecentric configuration, N = No telecentric configuration)<br>
**Sample:** type of object recorded (Usaf, Star, Probiotics, etc.)<br>
**Objective:** microscope objective magnification used (10x, 20x, 40x)<br>
**Wavelength**: in nanometers (532, 632)<br>
**Pixel size:** in micrometers<br>
Additional parameters (optional): such as defocus distance (-4cm).<br>

### Reconstruction Samples

**Samples → Reconstruction Samples** opens **compensated phase maps**, ready to be measured with every module. The file name gives the object and the compensation method, so two methods applied to the same object can be compared directly:

**Drosophila_PCA.bmp** / **Drosophila_VortexLeg.bmp** — Drosophila, compensated with PCA and with VortexLeg.<br>
**starTarget_PCA.bmp** / **starTarget_VortexLeg.bmp** — Star Target, compensated with PCA and with VortexLeg.<br>

The submenu lists every file in `My_app/Samples compensated phase/`, so a phase map added to that folder appears in it automatically.

---
## Repository structure
 
```
My_app/
├── analysis/
│   ├── module1/
│   │   ├── residual_background.py
│   │   ├── segmentation.py
│   │   └── utilitiesRBPV.py
│   ├── module2/
│   │   └── global_phase.py
│   ├── module3/
│   │   └── ground_truth_comparison.py
│   ├── cache.py
│   ├── catalog.py
│   └── __init__.py
│
├── complexity_algorithm/          # Module IV: computational complexity tool
│   ├── Main.py
│   ├── complexity_fit.py
│   ├── perf_protocol.py
│   ├── visualizer_v2.py
│   ├── setup.py / setup_windows.py
│   ├── test_complexity_fit.py
│   ├── Algorythm_results_windows/
│   ├── Gray_images/               # test images, 128x128 ... 4096x4096
│   ├── Hologram_stack/            # test holograms, 128x128 ... 4096x4096
│   ├── Test_algorithm/
│   ├── Test_files/
│   ├── User_manual/
│   │   └── User_Manual.docx
│   └── README.md
│
├── core/
│   ├── file_manager.py
│   └── imagej_bridge.py
│
├── menus/
│   ├── analysis_menu.py
│   ├── file_menu.py
│   ├── help_menu.py
│   └── samples_menu.py
│
├── Samples Hologram/
│   ├── Telecentric/
│   │   ├── T_Probiotics_20x_632_3.75.bmp
│   │   ├── T_Star_10x_632_3.75.bmp
│   │   └── T_Usaf_20x_632_3.75.bmp
│   └── No Telecentric/
│       ├── N_Star_20x_532_5.86_-4cm.tiff
│       └── NredBlood_40x_632_4.65_30mm.tiff
│
├── Samples compensated phase/     # listed under Samples → Reconstruction Samples
│   ├── Drosophila_PCA.bmp
│   ├── Drosophila_VortexLeg.bmp
│   ├── starTarget_PCA.bmp
│   └── starTarget_VortexLeg.bmp
│
├── ui/
│   ├── icons/                     # toolbar icons (SVG)
│   ├── hidpi.py
│   ├── icons.py
│   ├── image_selection_dialog.py
│   ├── legendre_viewer.py
│   ├── main_frame.py
│   ├── metric_selection_dialog.py
│   ├── results_dialog.py
│   ├── threshold_dialog.py
│   ├── tsm_dialog.py
│   └── zone_selector.py
│
├── User_manual/
│   └── User_Manual.docx
│
├── app.py              # Main entry point — run this file to launch PhaseMark
├── app.spec            # PyInstaller build file (produces PhaseMark.exe)
├── detached_notebook.py
└── requirements.txt
```
---

## Getting Started  

### Requirements  

| Package | Minimum version |
|---------|---------------|
| Python | ≥ 3.13.9 |
| wxPython | ≥ 4.2 |
| numPy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| matplotlib | ≥ 3.7 |
| pillow | ≥ 10.0 |
| scikit-image | ≥ 0.21.0 |
| scipy | ≥ 1.10.0 |
| openpyxl | ≥ 3.1.0 |
| scikit-learn | ≥ 1.3 | 
| opencv-python | ≥ 4.8 | 
| pyimagej | ≥ 1.7.0 (optional, for ImageJ integration) |
| JPype1 | ≥ 1.3.0 (optional, for ImageJ integration) |

The ImageJ integration also needs a Java runtime. Everything else works without it.

With Python 3.13.9 or higher already installed, install all dependencies using the provided requirements.txt file:
```bash
pip install -r My_app/requirements.txt
```
**Note**: As an alternative method (without using the file), dependencies can also be installed manually by running:

```bash
pip install wxPython numpy pandas matplotlib pillow scikit-image scipy openpyxl scikit-learn opencv-python pyimagej JPype1
```
 
---
 
## Quick start — GUI
 
```bash
python My_app/app.py
```
## How the Application Works

### 1. Launching the Application
After running `app.py`, the PhaseMark main window opens, giving access to the workspace, the toolbar and the menu bar.

The **toolbar** puts the most used actions one click away: Open, Save, ImageJ, the four modules (Background, Global, Ground truth, Complexity), Clear and Help. It can be hidden and shown again with **View → Toolbar**, the arrow at its right end, or **Ctrl+T**.

### 2. Loading an Image
1. Go to **File → Open** (Ctrl+O).
2. Select the phase map/image (.mat, .npy, .raw, .bin, .png, .jpg, .bmp, .tif) to analyze. A real `.mat` or `.npy` array, a float `.raw`/`.bin`, or a 32-bit float TIFF is used as the phase in radians; a complex `.mat` or `.npy` array is a field whose phase is kept; an 8-bit image is mapped from gray levels onto [−π, π]. A `.raw`/`.bin` has no header, so a dialog asks for its width, height, data type and byte order, prefilled from a name like `fase_640x480_float32.raw`.
3. The image is displayed in the main workspace.

PhaseMark expects a **compensated phase map**, not a hologram. While the active tab holds one of the bundled holograms, Modules I to III are disabled.

### 3. Analysis Menu
The **Analysis** menu contains four modules:

- **Module 1** – Residual Background Phase Variance
- **Module 2** – Global Phase
- **Module 3** – Ground Truth Comparison
- **Module 4** – Computational complexity

---

### Module 1 – Residual Background Phase Variance

When this module is selected, a metric-selection window appears, listing all available metrics (STD, MAD, RMS, PV, FWHM, Entropy, Phase jumps, Legendre — each for Unwrapped Background, Background, and Background Zones). The user can select one or several metrics and click **Apply**.

Each metric offers **3 calculation options**:

**Option 1 — Unwrapped Background**
- The phase is unwrapped before measuring. If this is the only metric selected, a warning is shown first, since unwrapping may jeopardize phase measurements.
- A **segmentation step** follows:
  - The user adjusts the threshold to separate sample from background.
  - Once satisfied, the user clicks **OK** and the metric is calculated.

**Option 2 — Background (wrapped phase)**
- Skips the unwrapping step and warning.
- Starts directly at the threshold/segmentation step; the rest of the process is identical to Option 1.

**Option 3 — Zone-based analysis**
1. The user is asked whether to unwrap the phase (once per run).
2. The user specifies the number of zones to analyze.
3. The image is displayed, and the user draws a bounding box for each zone of interest.
4. The metric is calculated per zone, along with the overall average across all zones.

---

### Module 2 – Global Phase

For each available metric, the user chooses between:
- Calculate **without** unwrapping the phase (Global), or
- Calculate **with** unwrapping the phase (Unwrapped).

Once an option is selected, the metric is calculated for the entire field — no segmentation or zone selection is required. The only exception is **TSM**, which opens a dialog where the user sets the binarization threshold and the sample side.

---

### Module 3 – Ground Truth Comparison

1. The module first requires a **reference image** of the same size. If none is loaded, the user is prompted to load one.
2. The user selects the desired metrics.
3. The selected metrics are calculated by comparing the loaded image against the reference. Neither image is unwrapped.

---

### Results Table and Export

As metrics are calculated (from any module), their values are appended to a results table shown at the bottom of the workspace, labeled with the metric name. When a run finishes, a summary window shows the same results with one row per metric.

Results can be exported in two ways:
- **Right-click** on the table → export to **.xlsx** or **.csv**, or
- **File → Save Results** (Ctrl+S), which exports every table to Excel.

A copy of the image in the current tab can be saved with **File → Save Image As** (Ctrl+Shift+S). A tab holding a phase can also export it, in radians, as a 32-bit TIFF or as a headerless `.raw`/`.bin` (float32, little-endian); the size is written into the file name, e.g. `fase_640x480_float32.raw`.

---

### ImageJ Integration

**ImageJ → Send current image to ImageJ** (or the ImageJ toolbar button) opens ImageJ (Fiji) with the **phase** of the active tab as a **32-bit float image in radians** (for a `.mat` file, its true phase; for a hologram, which holds no phase, its 8-bit picture).

Edits made in ImageJ come back into PhaseMark as a tab named `<image>_ImageJ`, refreshed after each change. That tab keeps two things:
- **What it shows:** the image as drawn in ImageJ, with LUT, contrast and overlays.
- **What it measures:** the 32-bit phase values, in radians. LUTs, Brightness/Contrast (without *Apply*) and overlays do not affect the metrics.

If the image is converted away from 32-bit in ImageJ, PhaseMark warns that it no longer holds the phase.

**File → Save Image As** on an ImageJ tab saves either the image as shown (PNG/TIFF/BMP) or the phase in radians as a 32-bit TIFF or `.raw`/`.bin`, which **File → Open** reads back as phase.

---

### Computational complexity Module

This module evaluates the **computational complexity** of phase-compensation algorithms, including execution time, number of operations, and related performance metrics. The profiling relies on the Linux `perf` utility, so measurements require Linux or WSL2.

---

### Help

**Help → View Help** (F1) explains every metric, how to read it, and the keyboard shortcuts.

## References

[1] Lloret T, Navarro-Fuster V, Ramírez MG, Morales-Vidal M, Beléndez A, Pascual I. Aberration-Based Quality Metrics in Holographic Lenses. *Polymers* 2020;12:993. https://doi.org/10.3390/POLYM12040993<br>
[2] Godden TM, Muñiz A, Claverley JD, Yacoot A, Humphry MJ, Seaberg MD, et al. Phase calibration target for quantitative phase imaging with ptychography. *Optics Express* 2016;24:7679–92. https://doi.org/10.1364/OE.24.007679<br>
[3] Yang J, Li F, Du J, et al. Automatic aberration compensation for digital holographic microscopy based on bicubic downsampling and improved minimization of global phase gradients. *Optics Express* 2023;31:36188–201. https://doi.org/10.1364/OE.496840<br>
[4] Castaneda R, Trujillo C, Doblas A. Video-rate quantitative phase imaging using a digital holographic microscope and a generative adversarial network. *Sensors* 2021;21:8021. https://doi.org/10.3390/S21238021<br>
[5] Bansal R, Raj G, Choudhury T. Blur image detection using Laplacian operator and Open-CV. *Proceedings of the 5th International Conference on System Modeling and Advancement in Research Trends, SMART* 2017:63–7. https://doi.org/10.1109/SYSMART.2016.7894491<br>
[6] Eskicioglu AM, Fisher PS. Image Quality Measures and Their Performance. *IEEE Transactions on Communications* 1995;43:2959–65. https://doi.org/10.1109/26.477498


