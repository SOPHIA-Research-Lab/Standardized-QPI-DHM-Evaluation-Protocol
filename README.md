# Standardized-QPI-DHM-Evaluation-Protocol

A Python-based GUI for evaluating the efficiency and validity of phase-reconstruction algorithms in **Digital Holographic Microscopy (DHM)**.  
This repository provides a **standardized protocol** for Quantitative Phase Imaging (QPI) evaluation, structured into four main modules (*cores*) that cover background stability, global phase distortions, ground-truth comparisons, and computational complexity.  

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
- Root Mean Square (RMS)  
- Peak-to-Valley (P–V) Value in background  
- Full Width at Half Maximum (FWHM) of the phase histogram  
- Spatial frequency content of background  
- Entropy of background phase map
- Background phase tilt/curvature residuals (Legendre/Zernike coefficients)  


---

##  Nucleo II – Global Phase Distortion Metrics  

For samples without large empty regions. Evaluates distortions across the entire phase map.  

**Metrics:**  
- Peak-to-Valley (P–V)
- Global phase gradient 
- TSM (Total Sharpness Metric)  
- Phase curvature coefficients (low-order polynomial fitting)  
- Laplacian energy (Curvature Energy)  
- Spatial frequency content  
- Entropy of phase map  
- Sharpness/Contrast metrics  

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

## Test Holograms  

The repository includes a diverse set of test holograms to ensure robust evaluation:  
Included sample holograms correspond to calibration patterns (USAF and Star Target) and a biological sample. File name for each hologram indicates the parameters used to record it:

**T_Usaf_20x_632_3.75.bmp** — USAF pattern, 20x objective, 632 nm wavelength, 3.75 µm pixel size.
**N_Star_20x_532_5.86_-4cm.tiff** — Star Target pattern, 20x objective, 532 nm wavelength, 5.86 µm pixel size, -4 cm defocus.
**T_Star_10x_632_3.75.bmp** — Star Target pattern, 10x objective, 632 nm wavelength, 3.75 µm pixel size.
**T_Probiotics_20x_632_3.75.bmp** — Biological sample (probiotics), 20x objective, 632 nm wavelength, 3.75 µm pixel size.

Naming convention: [Prefix]_[Sample]_[Objective]_[Wavelength in nm]_[Pixel size in µm]_[Additional parameters].[extension]

Where:

**Prefix:** indicates the hologram type (T = Telecentric configuration, N = No telecentric configuration)<br>
**Sample:** type of object recorded (Usaf, Star, Probiotics, etc.)<br>
**Objective:** microscope objective magnification used (10x, 20x, 40x)<br>
**Wavelength**: in nanometers (532, 632)<br>
**Pixel size:** in micrometers<br>
Additional parameters (optional): such as defocus distance (-4cm).<br>

---
## Repository structure
 
```
MY_APP/
├── analysis/
│   ├── module1/
│   │   ├── residual_background.py
│   │   └── utilitiesRBPV.py
│   ├── module2/
│   │   └── global_phase.py
│   ├── module3/
│   │   └── ground_truth_comparison.py
│   └── __init__.py
│
├── complexity_algorithm/
│   ├── anidado_1.py
│   ├── anidado_2.py
│   ├── anidado.py
│   ├── pyDHM_methods.py
│   ├── SHPC.py
│   ├── test_functions.py
│   ├── test.py
│   ├── Gray_images/
│   │   ├── 128x128/
│   │   ├── 256x256/
│   │   ├── 512x512/
│   │   ├── 640x480/
│   │   ├── 800x600/
│   │   ├── 1024x768/
│   │   ├── 1600x1200/
│   │   ├── 1920x1440/
│   │   ├── 2048x2048/
│   │   ├── 2560x1920/
│   │   ├── 3840x2880/
│   │   └── 4096x4096/
│   │       └── (img1.png, img2.png, img3.png per resolution)
│   │
│   ├── Hologram_stack/
│   │   ├── 128x128/
│   │   ├── 256x256/
│   │   ├── 512x512/
│   │   ├── 640x480/
│   │   ├── 800x600/
│   │   ├── 1024x768/
│   │   ├── 1600x1200/
│   │   ├── 1920x1440/
│   │   ├── 2048x2048/
│   │   ├── 2560x1920/
│   │   ├── 3840x2880/
│   │   └── 4096x4096/
│   │       └── (hologram_01.png ... hologram_10.png per resolution)
│   │
│   ├── Test_algorithm/
│   │   ├── pyDHM_methods.py
│   │   └── SHPC.py
│   │
│   ├── Test_files/
│   │   ├── SHPC_Perfomance.txt
│   │   ├── test_functions.py
│   │   └── Vortex_Performance.txt
│   │
│   └── User_manual/
│       └── User_Manual.docx
│
├── core/
│   └── file_manager.py
│
├── menus/
│   ├── analysis_menu.py
│   ├── edit_menu.py
│   ├── file_menu.py
│   ├── help_menu.py
│   └── samples_menu.py
│
├── Samples Hologram/
│   ├── N_Star_20x_532_5.86_-4cm.tiff
│   ├── T_Probiotics_20x_632_3.75.bmp
│   ├── T_Star_10x_632_3.75.bmp
│   └── T_Usaf_20x_632_3.75.bmp
│
├── ui/
│   ├── image_selection_dialog.py
│   ├── main_frame.py
│   └── metric_selection_dialog.py
│
├── app.py              # ⭐ Main entry point — run this file to ├── launch the application
├── detached_notebook.py
├── launcher.py
├── README.md
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

With Python 3.13.9 or higher already installed, install all dependencies using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```
**Note**: As an alternative method (without using the file), dependencies can also be installed manually by running:

```bash
pip install wxPython numpy pandas matplotlib pillow scikit-image scipy openpyxl scikit-learn opencv-python
```
 
---
 
## Quick start — GUI
 
```bash
python app.py
```
## How the Application Works

### 1. Launching the Application
After running `app.py`, the main window opens, giving access to the workspace and the menu bar.

### 2. Loading an Image
1. Go to **File → Open**.
2. Select the phase map/image (.mat,.jpeg,.png,.bmp) to analyze.
3. The image is displayed in the main workspace.

### 3. Analysis Menu
The **Analysis** menu contains four modules:

- **Module 1** – Residual Background Phase Variance
- **Module 2** – Global Phase
- **Module 3** – Ground Truth Comparison
- **Module 4** – Computational complexity

---

### Module 1 – Residual Background Phase Variance

When this module is selected, a metric-selection window appears, listing all available metrics (STD, MAD, RMS, PV, FWHM, Entropy, Legendre — each for Unwrapped Background, Background, and Background Zones). The user can select one or several metrics and click **Apply**.

Each metric offers **3 calculation options**:

**Option 1 — Calculate with phase unwrapping**
- A warning is shown first, since unwrapping may jeopardize phase measurements.
- If the user chooses to continue, a **segmentation step** is required:
  - The user adjusts the threshold to separate sample from background.
  - Once satisfied, the user clicks **OK** and the metric is calculated.

**Option 2 — Calculate whit or without phase unwrapping**
- Skips the unwrapping step and warning.
- Starts directly at the threshold/segmentation step; the rest of the process is identical to Option 1.

**Option 3 — Zone-based analysis**
1. The same unwrapping warning is shown (with the option to unwrap or not).
2. The user specifies the number of zones to analyze.
3. The image is displayed, and the user draws a bounding box for each zone of interest.
4. The metric is calculated per zone, along with the overall average across all zones.

---

### Module 2 – Global Phase

For each available metric, the user chooses between:
- Calculate **without** unwrapping the phase, or
- Calculate **with** unwrapping the phase.

Once an option is selected, the metric is calculated automatically for the entire sample — no additional steps (segmentation or zone selection) are required.

---

### Module 3 – Ground Truth Comparison

1. The module first requires a **reference image**. If none is loaded, the user is prompted to load one.
2. The user selects the desired metrics.
3. As in Module 2, each metric can be calculated **with or without** phase unwrapping.
4. The selected metrics are then calculated by comparing the loaded image against the reference.

---

### Results Table and Export

As metrics are calculated (from any module), their values are appended to a results table shown at the bottom of the workspace, labeled with the metric name.

Results can be exported in two ways:
- **Right-click** on the table → export option, or
- **File → Save**, choosing between **.xlsx** or **.csv** format.

---

### Computational complexity Module

This module evaluates the **computational complexity** of phase-compensation algorithms, including execution time, number of operations, and related performance metrics.

## References

[1] Lloret T, Navarro-Fuster V, Ramírez MG, Morales-Vidal M, Beléndez A, Pascual I. Aberration-Based Quality Metrics in Holographic Lenses. *Polymers* 2020;12:993. https://doi.org/10.3390/POLYM12040993<br>
[2] Godden TM, Muñiz A, Claverley JD, Yacoot A, Humphry MJ, Seaberg MD, et al. Phase calibration target for quantitative phase imaging with ptychography. *Optics Express* 2016;24:7679–92. https://doi.org/10.1364/OE.24.007679<br>
[3] Yang J, Li F, Du J, et al. Automatic aberration compensation for digital holographic microscopy based on bicubic downsampling and improved minimization of global phase gradients. *Optics Express* 2023;31:36188–201. https://doi.org/10.1364/OE.496840<br>
[4] Castaneda R, Trujillo C, Doblas A. Video-rate quantitative phase imaging using a digital holographic microscope and a generative adversarial network. *Sensors* 2021;21:8021. https://doi.org/10.3390/S21238021<br>
[5] Bansal R, Raj G, Choudhury T. Blur image detection using Laplacian operator and Open-CV. *Proceedings of the 5th International Conference on System Modeling and Advancement in Research Trends, SMART* 2017:63–7. https://doi.org/10.1109/SYSMART.2016.7894491<br>
[6] Eskicioglu AM, Fisher PS. Image Quality Measures and Their Performance. *IEEE Transactions on Communications* 1995;43:2959–65. https://doi.org/10.1109/26.477498


