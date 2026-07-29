# CERT-SAT: Multi-Channel Anomaly Detection for Spacecraft Telemetry

> An end-to-end machine learning framework for detecting anomalies in multivariate spacecraft telemetry using a Multi-Channel Isolation Forest architecture.

---

## Overview

CERT-SAT is an anomaly detection framework designed to identify abnormal behavior in spacecraft telemetry. The project processes raw telemetry data, extracts statistical features from sliding windows, and applies a dedicated Isolation Forest model to each telemetry channel.

Unlike traditional single-model approaches, CERT-SAT treats every telemetry channel independently, allowing the framework to capture channel-specific behavior while maintaining scalability across high-dimensional datasets.

The project is built around the NASA SMAP and MSL benchmark datasets and provides a complete pipeline covering data preprocessing, feature engineering, model training, inference, evaluation, and backend integration.

---

## Features

- Multi-Channel Isolation Forest architecture
- Statistical feature extraction from telemetry windows
- Sliding window preprocessing
- Per-channel anomaly detection
- StandardScaler-based feature normalization
- Model persistence using Joblib
- Evaluation using Precision, Recall, and F1-score
- API-ready inference pipeline
- Modular and extensible project structure

---

## Repository Structure

```text
CERT-SAT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── archive/
│
├── models_saved/
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
│
├── examples/
│
├── notebooks/
│
├── docs/
│   ├── 01_Project_Overview.md
│   ├── 02_System_Architecture.md
│   ├── 03_Data_and_Preprocessing.md
│   ├── 04_Model_Development.md
│   ├── 05_Model_Evaluation.md
│   ├── 06_API_and_Inference.md
│   ├── 07_Research_Background.md
│   └── 08_Appendix.md
│
└── README.md
```

---

## System Workflow

```text
Raw Telemetry
      │
      ▼
Data Preprocessing
      │
      ▼
Sliding Window Generation
      │
      ▼
Statistical Feature Extraction
      │
      ▼
Feature Scaling
      │
      ▼
Multi-Channel Isolation Forest
      │
      ▼
Anomaly Scores
      │
      ▼
Prediction & Evaluation
```

---

## Dataset

This project uses the publicly available NASA spacecraft telemetry datasets:

- SMAP (Soil Moisture Active Passive)
- MSL (Mars Science Laboratory)

These datasets contain multivariate telemetry streams with labelled anomaly intervals and are widely used for benchmarking anomaly detection algorithms.

---

## Model Architecture

CERT-SAT trains an independent Isolation Forest model for every telemetry channel.

Each channel follows the pipeline:

```text
Telemetry Channel
        │
        ▼
Sliding Window
        │
        ▼
Statistical Features
        │
        ▼
StandardScaler
        │
        ▼
Isolation Forest
        │
        ▼
Anomaly Score
```

This architecture enables channel-specific anomaly detection while remaining computationally efficient.

---

## Feature Engineering

Each sliding window is converted into a compact feature vector using statistical descriptors such as:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Range / Trend-based statistics

These features summarize the temporal behavior within each window and provide meaningful input for the Isolation Forest models.

---

## Training Pipeline

The training process consists of:

1. Load raw telemetry.
2. Generate sliding windows.
3. Extract statistical features.
4. Normalize features.
5. Train one Isolation Forest per telemetry channel.
6. Save trained models and preprocessing objects.

---

## Inference Pipeline

For unseen telemetry:

1. Load trained model.
2. Apply preprocessing.
3. Extract window features.
4. Scale features.
5. Predict anomaly scores.
6. Aggregate channel predictions.
7. Return anomaly labels.

---

## Evaluation

Model performance is assessed using:

- Precision
- Recall
- F1-score
- ROC analysis
- Precision–Recall analysis
- Error analysis
- Qualitative prediction reports

---

## Documentation

| Document | Description |
|----------|-------------|
| Project Overview | Project goals, datasets, and workflow |
| System Architecture | Internal architecture and design |
| Data & Preprocessing | Window generation and feature extraction |
| Model Development | Isolation Forest implementation |
| Model Evaluation | Metrics, experiments, and analysis |
| API & Inference | Backend integration and inference |
| Research Background | Literature review and related work |
| Appendix | Hyperparameters, references, and supplementary material |

---

## Future Improvements

Potential future enhancements include:

- Deep learning–based anomaly detection models
- Online/streaming anomaly detection
- Dynamic threshold optimization
- Explainable AI techniques
- Real-time dashboard visualization
- Model ensemble approaches

---

## Acknowledgements

This project builds upon publicly available NASA telemetry datasets and established anomaly detection research, including Isolation Forest, LSTM Encoder–Decoder methods, and related literature in time-series anomaly detection.

---

## License

Specify the appropriate open-source license for your repository (e.g., MIT, Apache-2.0).
