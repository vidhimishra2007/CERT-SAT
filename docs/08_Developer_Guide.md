# Developer Guide

> This guide explains how to set up the project, train new models, run predictions, evaluate performance, understand the project structure, and contribute to CERT-SAT.

---

# Overview

This guide is intended for developers who want to:

- Run the project locally
- Train new anomaly detection models
- Evaluate model performance
- Integrate the model into another application
- Extend the framework with additional algorithms
- Contribute to future development

---

# Development Environment

## Requirements

- Python 3.10+
- Git
- pip
- Virtual Environment (recommended)

---

## Clone Repository

```bash
git clone https://github.com/<username>/CERT-SAT.git

cd CERT-SAT
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Project Structure

```
CERT-SAT/

├── README.md

├── docs/

├── data/

│   ├── raw/

│   ├── processed/

│   └── archive/

├── models_saved/

├── notebooks/

├── src/

│   ├── preprocessing/

│   ├── training/

│   ├── inference/

│   ├── evaluation/

│   └── utils/

├── tests/

├── examples/

└── requirements.txt
```

---

# Running the Project

## Train the Model

```bash
python train_model.py
```

Training performs:

- data loading
- preprocessing
- feature extraction
- scaling
- model training
- model serialization

The trained model is stored inside

```
models_saved/
```

---

## Run Prediction

```bash
python predict.py
```

Prediction performs

- load model
- preprocess telemetry
- extract features
- generate anomaly scores
- return prediction

---

## Evaluate the Model

```bash
python evaluate.py
```

Evaluation generates

- Precision
- Recall
- F1-score
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix

---

# Replacing the Dataset

To use a different telemetry dataset:

1. Copy the dataset into

```
data/raw/
```

2. Update the dataset configuration.

3. Retrain the model.

No other changes should be required if the dataset follows the expected format.

---

# Adding a New Algorithm

The project is designed to support additional anomaly detection algorithms.

Example workflow:

```
New Algorithm

↓

Feature Extraction

↓

Training Module

↓

Evaluation Module

↓

Inference Module
```

Possible algorithms include:

- Local Outlier Factor
- One-Class SVM
- Autoencoder
- LSTM Encoder–Decoder
- Transformer

Only the model-specific training and inference logic needs to be implemented; the preprocessing pipeline can be reused.

---

# Coding Standards

Developers contributing to the project should follow these guidelines:

- Use descriptive variable names.
- Write modular functions.
- Add docstrings for public methods.
- Keep functions focused on a single responsibility.
- Avoid duplicated code.
- Follow PEP 8 style conventions.

---

# Common Issues

## Model File Not Found

Cause

The trained model has not been generated.

Solution

Run

```bash
python train_model.py
```

---

## Incorrect Input Shape

Cause

Telemetry data does not match the expected format.

Solution

Verify:

- number of channels
- window size
- feature dimensions

---

## Missing Dependencies

Cause

Required Python packages are not installed.

Solution

```bash
pip install -r requirements.txt
```

---

## Prediction Errors

Possible reasons include:

- corrupted model files
- incorrect preprocessing
- mismatched StandardScaler
- incompatible dataset format

---

# Best Practices

For reliable results:

- Always use the same preprocessing pipeline for training and inference.
- Retrain the model after changing feature engineering.
- Keep configuration files under version control.
- Validate new datasets before training.
- Store trained models separately from source code.

---

# Future Improvements

Future contributions may include:

- Deep learning models
- Online anomaly detection
- Explainable AI
- Ensemble methods
- Hyperparameter optimization
- Interactive dashboards
- Real-time telemetry streaming

---

# Contributing

Contributions are welcome.

Recommended workflow:

1. Fork the repository.
2. Create a new feature branch.
3. Implement changes.
4. Add tests.
5. Update documentation.
6. Submit a Pull Request.

Please ensure that all code is documented and follows the project's coding standards.

---

# Documentation Index

| Document | Description |
|-----------|-------------|
| README | Project overview |
| Project Overview | Objectives and workflow |
| System Architecture | Architecture and modules |
| Data Pipeline | Data preparation and feature engineering |
| Model Development | Multi-Channel Isolation Forest |
| Model Evaluation | Metrics and validation |
| Deployment & Inference | Model deployment and API |
| Research Background | Literature review |
| Developer Guide | Setup and contribution |

---

# Final Notes

CERT-SAT has been designed as a modular and extensible anomaly detection framework. The project structure separates data processing, model development, evaluation, and deployment into independent components, making it straightforward to maintain, extend, and integrate into future applications.
