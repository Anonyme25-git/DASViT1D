**DASViT1D** is a framework for intrusion/event recognition with **Distributed Acoustic Sensing (DAS)**. It fuses **multi-domain features** — **MFCC**, **RDFT**, and **DWT** — and feeds them to a **1D Vision Transformer (ViT-1D)** to capture long-range temporal dependencies and classify activities (walking, running, vehicle, construction, etc.).

> **Highlights**
> - Complementary fusion: **MFCC + RDFT + DWT**
> - **1D Transformer** encoder (multi-head self-attention)
> - Outperforms comparable CNN baselines on the same features
> - Low inference latency (near real-time)

---

## Table of Contents
- [Context](#context)
- [Data](#data)
- [Method](#method)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Reference Hyperparameters](#reference-hyperparameters)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Context
Fiber-optic **ϕ-OTDR/DAS** turns standard fiber into a dense acoustic sensor array. The goal is to **recognize intrusion patterns** from DAS signals by combining complementary time-frequency representations and a **1D Transformer** classifier.

---

## Data
- **Dataset**: public ϕ-OTDR dataset with **9 classes** (vehicle, fence/impact, longboard, manipulation, open/close, construction, regular/idle, running, walking).
- link : https://doi.org/10.6084/m9.figshare.27004732 
- **Recommended splits**: 80/10/10 for train/val/test.  
- Avoid committing raw data to Git; prefer `.gitignore` or external storage (or Git LFS if needed).

---

## Method
1) **Feature extraction**
   - **MFCC** (e.g., `n_mfcc=20`, `n_fft=1024`, `hop=128`, `n_mels=128`, `f_max=500 Hz`).
   - **RDFT** (redundant FFT, e.g., redundancy `r=3`) for improved frequency resolution.
   - **DWT** (e.g., Daubechies-4, 3 levels) for time-frequency localization.
   - **Fusion**: concatenate MFCC + RDFT + DWT; **normalize** (fit scaler on train only).

2) **Model**
   - **1D patching** of fused sequences → linear projection + positional encoding.  
   - **Transformer encoder** (MHSA) → classification head (9 classes).

---

## Results
- **Accuracy**: ~**93.5%**  
- **FNR**: ~**0.9%** • **NAR**: ~**0.5%**  
- **Inference latency**: ~**0.13 s / sample**

> Figures above are reported from the accompanying paper (see citation) with fixed splits and identical preprocessing.

---

## Installation
> Python ≥ 3.9 recommended

```bash
# create & activate a virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -U pip
pip install torch numpy scipy scikit-learn librosa pywt matplotlib tqdm
# or:
# pip install -r requirements.txt
