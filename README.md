# 🧠 NeuroHologram: Sparse EEG Reconstruction System

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Turning Consumer Wearables into Medical-Grade Diagnostic Tools."**

**NeuroHologram** is a hybrid AI/Mathematical framework designed to solve the **EEG Inverse Upsampling Problem**. It reconstructs high-fidelity, 32-channel medical EEG signals from sparse, 4-channel consumer-grade inputs (e.g., Muse/Emotiv headsets) using a **1D U-Net** architecture refined by **ADMM Convex Optimization**.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Mathematical Foundation](#-mathematical-foundation)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Performance Results](#-performance-results)
- [Project Structure](#-project-structure)

---

## 🚀 Project Overview

### The Problem
Medical diagnosis (Epilepsy, Tumors, BCI) requires high-density EEG caps (32-64 sensors), which are expensive, wired, and require a technician. Consumer headsets are wireless and cheap but only have 4 sensors, making them useless for medical analysis due to low spatial resolution.

### The Solution
**NeuroHologram** treats the brain as a **Volume Conductor**. Since electrical signals propagate through the skull, data at the "corners" of the head contains latent information about the "center."
* **Input:** 4 Channels (Fp1, Fp2, O1, O2) - 87.5% Missing Data.
* **Output:** 32 Channels (Full 10-20 Standard Montage).
* **Method:** We use Deep Learning to learn the non-linear mapping and Convex Optimization to enforce biological smoothness.

---

## ✨ Key Features
* **Holographic Reconstruction:** Rebuilds the full brain map from just 4 data points.
* **Physics-Informed Optimization:** Uses **ADMM (Alternating Direction Method of Multipliers)** to minimize Total Variation (TV), ensuring reconstructed signals are biologically plausible and noise-free.
* **U-Net Architecture:** Utilizes **Skip Connections** to preserve high-frequency features (Gamma waves, Seizure spikes) that are typically lost in standard Autoencoders.
* **Interactive Dashboard:** A professional **Streamlit** UI allowing real-time patient simulation, sensor failure testing, and signal analysis.

---

## 🏗 System Architecture

The pipeline consists of three distinct stages:

1.  **Data Sparsification:**
    * **Source:** DEAP Dataset (32-Channel, 128Hz).
    * **Process:** We apply a binary mask $M$, setting 28 channels to zero and keeping only the 4 corner sensors.
2.  **Deep Learning Prior (U-Net):**
    * The sparse input is fed into a **1D U-Net**.
    * **Encoder:** Compresses temporal dynamics into a latent space.
    * **Decoder:** Upsamples the features back to 32 channels.
    * **Skip Connections:** Directly transfer high-frequency details from input to output.
3.  **Mathematical Refinement (ADMM):**
    * The AI output serves as the "Prior" ($x_{AI}$).
    * The ADMM solver optimizes the final signal to minimize noise while staying close to the AI prediction.

---

## 📐 Mathematical Foundation

We solve the **Inverse Problem** defined as $Y = M \cdot X + \epsilon$, where $M$ is the sparse masking matrix.

The reconstruction is obtained by solving the following **Convex Optimization Problem**:

$$\hat{x} = \arg\min_x \left( \underbrace{\frac{1}{2} \|x - x_{AI}\|_2^2}_{\text{Fidelity Term}} + \underbrace{\lambda \|\nabla x\|_1}_{\text{Sparsity Prior (TV)}} \right)$$

* **Term 1:** Forces the output to respect the learned patterns from the U-Net.
* **Term 2:** The **Total Variation (TV)** norm penalizes jagged, non-biological noise, enforcing smoothness consistent with volume conduction physics.

---

## 💻 Installation & Setup

### Prerequisites
* Python 3.8+
* RAM: 8GB+ (16GB Recommended)

### Step 1: Clone Repository
```bash
git clone [https://github.com/YOUR_USERNAME/NeuroHologram.git](https://github.com/YOUR_USERNAME/NeuroHologram.git)
cd NeuroHologram
