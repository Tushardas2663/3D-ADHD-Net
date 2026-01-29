# 3D ADHD-Net & DeepTrace: Interpretable EEG-Based ADHD Diagnosis

Official implementation of the paper **"3D ADHD-Net: Decoding ADHD from EEG with Neurophysiological Insights"**.

##  Overview
This repository contains the source code for **3D ADHD-Net**, a topology-aware deep learning framework for diagnosing ADHD from EEG data, and **DeepTrace**, a novel hierarchical explainability method designed to discover neurophysiological biomarkers.

### Key Features
* **3D Topology-Aware Architecture**: Preserves the non-Euclidean geometry of the brain using Azimuthal Equidistant Projection.
* **DeepTrace Explainability**: A signed Spearman correlation-based backtracking algorithm that identifies directional biomarkers (Hypo/Hyper-activation) without zero-signal hallucinations.
* **Rigorous Validation**: Implements strict **5-fold Subject-Independent Cross-Validation** to prevent data leakage.

---

##  Repository Structure

The project consists of the following files:

| File | Description |
| :--- | :--- |
| `preprocessing.py` | **Data Pipeline.** Interpolates 19-channel EEG signals onto 8x8 topographical grids (preserving zero-padding) and handles subject-wise splitting. |
| `3D-ADHD_Net.py` | **Model Architecture.** Defines the hybrid 3D CNN-BiLSTM-Transformer model and provides information about the training loop with subject-independent cross-validation. |
| `DeepTrace.py` | **Explainability Algorithm.** The class implementation of the DeepTrace algorithm, including balanced cohort selection and recursive sign-aligned tracing. |

---
