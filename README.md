# 🧠 NeuroHologram: Sparse EEG Reconstruction

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

**NeuroHologram** is a Deep Learning & Mathematical Optimization system designed to reconstruct high-resolution medical EEG (32-channel) from sparse consumer-grade headsets (4-channel).



## 🚀 Project Overview
Medical EEG caps are expensive and cumbersome. Consumer headsets are cheap but lack the spatial resolution required for diagnosis. This project solves that gap by treating brain signal upsampling as an **Inverse Problem**.

**Key Technologies:**
* **Deep Learning:** 1D U-Net with Skip Connections for high-fidelity signal recovery.
* **Mathematics:** ADMM (Alternating Direction Method of Multipliers) for Total Variation Denoising.
* **Visualization:** Interactive Streamlit Dashboard for real-time holographic analysis.

## 📊 Results
* **Reconstruction Accuracy:** ~96% (Relative to Signal Range)
* **MAE:** ~2.0 - 2.7 μV (Within standard EEG noise margins)
* **Visual Fidelity:** Successfully captures high-frequency Gamma waves and seizure-like spikes.

## 🛠️ Installation

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/NeuroHologram.git](https://github.com/YOUR_USERNAME/NeuroHologram.git)
   cd NeuroHologram