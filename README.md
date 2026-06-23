# Parallel PINV Neural Network for MedMNIST Classification

A **Parallel Pseudo-Inverse Neural Network (PINV)** framework for **medical image classification** using the **MedMNIST** benchmark datasets. This project replaces conventional backpropagation with a **closed-form Moore-Penrose Pseudo-Inverse (PINV)** learning algorithm and combines multiple image feature extraction techniques through a **parallel ensemble architecture**.

---

## Overview

Traditional Artificial Neural Networks (ANNs) rely on iterative backpropagation for training, which can be computationally expensive and time-consuming. This project investigates an alternative approach by employing **Pseudo-Inverse Neural Networks (PINV)** that compute the output weights analytically without gradient descent.

The framework extracts multiple complementary image representations and trains an independent PINV classifier for each feature branch. The outputs are then combined using an ensemble voting strategy to improve classification performance.

---

## Features

* Closed-form PINV learning (No Backpropagation)
* Parallel Multi-Feature Architecture
* Fast Training using Moore-Penrose Pseudo-Inverse
* Multi-class Medical Image Classification
* Ensemble Voting
* MATLAB & Python Implementations
* Compatible with MedMNIST datasets

---

## Feature Extraction Methods

The current implementation supports multiple feature representations:

* Original Image
* Histogram Equalization
* Sobel Edge Features
* Binary Threshold Features
* DCT Features
* CLAHE (Adaptive Histogram Equalization)
* Laplacian Features

Additional feature extraction methods can be integrated easily.

---

## Network Architecture

```text
                     Input Image
                          │
      ┌──────────┬──────────┬──────────┬──────────┐
      │          │          │          │
 Original    Histogram    Sobel    Threshold
      │          │          │          │
    PINV       PINV       PINV       PINV
      └──────────┬──────────┬──────────┘
                 │
         Voting / Fusion Layer
                 │
        Final Classification
```

---

## Dataset

This project is designed for the **MedMNIST** benchmark datasets.

Examples:

* BreastMNIST
* BloodMNIST
* PathMNIST
* OrganMNIST
* DermaMNIST
* TissueMNIST

Dataset website:

https://medmnist.com/

---

## Repository Structure

```text
.
├── MATLAB/
│   ├── PINV_Parallel_BreastMNIST.m
│   ├── PMF_PINV_Framework.m
│   └── Utility Functions
│
├── Python/
│   ├── PINV_PNW_BloodMNIST.py
│   └── Supporting Scripts
│
├── Dataset/
│   └── *.mat
│
└── README.md
```

---

## Requirements

### Python

* Python 3.9+
* NumPy
* OpenCV
* SciPy
* scikit-learn
* MedMNIST

Install dependencies

```bash
pip install numpy scipy opencv-python scikit-learn medmnist
```

---

### MATLAB

Required Toolboxes

* Image Processing Toolbox
* Statistics and Machine Learning Toolbox

---

## Results

The framework evaluates the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Experiments were conducted on MedMNIST datasets using multiple feature extraction techniques and parallel PINV classifiers.

---

## Future Work

* Weighted Soft Voting
* Automatic Feature Selection
* Validation-Based Weight Optimization
* Ridge-Regularized PINV
* Deep Parallel PINV Networks
* Additional Feature Extraction Methods
* GPU Acceleration
* Explainable AI (Grad-CAM)

---

## Citation

If you use this work in your research, please cite this repository.

```bibtex
@misc{ParallelPINV,
  title={Parallel PINV Neural Network for Medical Image Classification},
  author={G. Arun},
  year={2026},
  publisher={GitHub}
}
```

---

## Author

**G. Arun**
AI Researcher

GitHub: https://github.com/Arun2517

---


