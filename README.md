# Standardized-QPI-DHM-Evaluation-Protocol
A Python-based GUI for evaluating the efficiency and validity of phase-reconstruction algorithms in Digital Holographic Microscopy (DHM).
This repository provides a standardized protocol for Quantitative Phase Imaging (QPI) evaluation, structured into four main modules (núcleos) that cover background stability, global phase distortions, ground-truth comparisons, and computational complexity.

# Standardized-QPI-DHM-Evaluation-Protocol  

A Python-based GUI for evaluating the efficiency and validity of phase-reconstruction algorithms in **Digital Holographic Microscopy (DHM)**.  
This repository provides a **standardized protocol** for Quantitative Phase Imaging (QPI) evaluation, structured into four main modules (*núcleos*) that cover background stability, global phase distortions, ground-truth comparisons, and computational complexity.  

---

## Overview  

Digital Holographic Microscopy (DHM) enables quantitative phase imaging, but the reliability of reconstruction algorithms varies depending on the evaluation criteria. This repository defines a **benchmark protocol** and provides tools to assess the quality of reconstructed phase maps using standardized metrics.  

The protocol is organized into **four evaluation cores (núcleos)**:  

1. Residual Background Phase Variance  
2. Global Phase Distortion Metrics  
3. Ground-Truth Comparisons  
4. Computational Complexity  

Additionally, a set of **test holograms** is included to benchmark performance across different types of samples.  

---

## Nucleo I – Residual Background Phase Variance  

Focuses on phase flatness in object-free regions of reconstructed phase images. Requires segmentation to isolate background regions.  

**Metrics:**  
- Standard deviation (STD) or Mean Absolute Deviation (MAD)  
- Root Mean Square (RMS)  
- Peak-to-Valley (P–V) Value in background  
- Background phase tilt/curvature residuals (Legendre/Zernike coefficients)  
- Full Width at Half Maximum (FWHM) of the phase histogram  
- Spatial frequency content of background  
- Entropy of background phase map  

---

##  Nucleo II – Global Phase Distortion Metrics  

For samples without large empty regions. Evaluates distortions across the entire phase map.  

**Metrics:**  
- Maximum–Minimum and Maximum–Minimum–Average–α-STD
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
- Percent Error (PE) – differences with manufacturer values in %  
- SSIM (Structural Similarity Index)  
- pSNR (peak Signal-to-Noise Ratio)  
- Self-consistency approaches:  
  - Double-exposure  
  - Conjugate wavefront reference  
  - Complex-field subtraction/multiplication for aberration-free reconstruction  

---

##  Nucleo IV – Computational Complexity  

Estimates computational cost of each algorithm:  
- Operation counts  
- Hardware-dependent benchmarks  
- Execution time and memory profiling  

---

## 🧪 Test Holograms  

The repository includes (or plans to include) a diverse set of test holograms to ensure robust evaluation:  

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Xxxxxxxxxxxxxxxxxxxxxxxxxxxx 
xxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxx

---

## Getting Started  

### Requirements  
- Python xxxxxxxxxxxxxxxx 


pip install -r requirements.txt
