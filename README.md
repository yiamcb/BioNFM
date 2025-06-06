# BioNFM
Official code for the paper "Neuro-Biomarker Driven Hybrid Neural Architecture for Parkinson’s Disease Detection" This repository contains the full implementation of the BioNFM model, along with preprocessing pipelines, training scripts, and evaluation tools.



# BioNFM: Biologically-Informed Neural Feature Modulation Network

This repository contains the official implementation of **BioNFM**, a deep learning framework for disease classification using EEG data. The model incorporates biologically meaningful features, attention mechanisms, and advanced signal processing techniques to distinguish between Healthy and Parkinson’s Disease (PD) subjects.

## Key Features

- Extraction and labeling of EEG data from BDF files
- Temporal windowing and preprocessing of EEG signals
- Feature extraction including:
  - Beta band inter-hemispheric coherence
  - Entropy
  - Fractal Dimension
  - Phase-Amplitude Coupling (PAC)
- Biologically informed model architecture combining:
  - Graph-inspired feature encoding
  - Transformer-based temporal attention
  - Reservoir dynamics via RNN
  - Custom biologically inspired loss function
- Statistical analysis and visualizations for result interpretation

---

## Files Overview

| File | Description |
|------|-------------|
| `Data_Extraction.py` | Loads EEG `.bdf` files, assigns labels (Healthy or PD), and stores raw signals. |
| `Pre_Processing.py` | Aligns EEG data lengths and splits into overlapping frames for temporal analysis. |
| `Features_Extraction_Routine.py` | Calculates inter-hemispheric coherence in the Beta band and compares groups. |
| `Visualizations.py` | Generates publication-ready plots and performs statistical tests (t-test, ANOVA) on entropy, PAC, and fractal features. |
| `Model_Architecture.py` | Defines and compiles the BioNFM model with GNN, Transformer, and RNN blocks, using a custom biologically-aware loss. |

---


## Outputs

- `Beta1.pdf`, `Beta2.pdf` – Coherence plots
- `fractalDims.pdf` – Fractal Dimension comparison
- `Entropy Features.pdf` – Entropy across EEG channels
- `PAC.pdf` – PAC distributions
- Printed ANOVA tables – Region-wise significance of features

---
