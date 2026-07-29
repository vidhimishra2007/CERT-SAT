# System Architecture

> This document describes the overall architecture of CERT-SAT, the interaction between its components, and the complete data flow from raw telemetry to anomaly prediction.

---

# Architecture Overview

CERT-SAT is designed as a modular machine learning pipeline. Each module performs one specific task, making the system easy to understand, maintain, and extend.

Rather than processing raw telemetry directly, the system transforms continuous sensor data into statistical feature vectors before applying a dedicated Isolation Forest model to each telemetry channel.

The architecture separates the workflow into independent stages:

1. Data Loading
2. Data Preprocessing
3. Sliding Window Generation
4. Feature Engineering
5. Feature Scaling
6. Model Training
7. Model Inference
8. Evaluation
9. Backend Integration

---

# High-Level Architecture

```text
                     NASA Telemetry Dataset
                              │
                              ▼
                     Data Loading Module
                              │
                              ▼
                    Data Preprocessing Module
                              │
                              ▼
                  Sliding Window Generation
                              │
                              ▼
                 Statistical Feature Extraction
                              │
                              ▼
                      Feature Normalization
                              │
                              ▼
                Multi-Channel Isolation Forest
                              │
                              ▼
                  Channel-wise Anomaly Scores
                              │
                              ▼
                     Prediction Aggregation
                              │
                              ▼
                    Backend/API Integration
```

Every stage has a single responsibility and communicates with the next stage through well-defined inputs and outputs.

---

# System Components

## 1. Data Loading Module

The Data Loading module is responsible for importing raw spacecraft telemetry from the dataset.

### Responsibilities

- Read telemetry files
- Load sensor values
- Maintain temporal ordering
- Prepare data for preprocessing

### Input

Raw telemetry sequences

### Output

Structured multivariate time-series

---

## 2. Data Preprocessing Module

Raw telemetry cannot be used directly for machine learning.

This module prepares the data by performing operations such as:

- Handling missing values
- Removing invalid records
- Standardizing input format
- Organizing channel data

The output is a clean and consistent dataset suitable for feature extraction.

---

## 3. Sliding Window Generator

Machine learning models require fixed-size observations.

Instead of processing an entire telemetry stream at once, CERT-SAT divides the signal into overlapping windows.

Example:

```text
Telemetry

□□□□□□□□□□□□□□□□□□□□□□□□□□□□

↓

Window 1

□□□□□□□□

↓

Window 2

    □□□□□□□□

↓

Window 3

        □□□□□□□□
```

Each window captures the local behavior of a telemetry channel.

Benefits include:

- Preserving temporal information
- Increasing the number of training samples
- Detecting localized anomalies
- Supporting continuous monitoring

---

# 4. Feature Engineering

Each sliding window is transformed into a compact statistical representation.

Instead of using hundreds of raw values, the model receives a small feature vector describing the window.

Typical statistical features include:

- Mean
- Standard deviation
- Minimum
- Maximum
- Trend
- Range

Example

```text
Window

[10 11 12 13 14 15 ...]

↓

Feature Vector

Mean = 12.5

Std = 1.9

Min = 10

Max = 15

Trend = +5
```

Feature engineering reduces computational complexity while preserving meaningful information.

---

# 5. Feature Scaling

Different statistical features have different numerical ranges.

Before model training, all features are normalized using **StandardScaler**.

Benefits:

- Faster learning
- Comparable feature magnitudes
- Improved numerical stability
- Consistent inference

The same scaler used during training is reused during prediction to ensure consistency.

---

# 6. Multi-Channel Isolation Forest

This is the core of the CERT-SAT framework.

Unlike traditional approaches that train a single model on all telemetry channels, CERT-SAT trains an independent Isolation Forest for each channel.

```text
Telemetry

├── Channel 1 → Isolation Forest 1

├── Channel 2 → Isolation Forest 2

├── Channel 3 → Isolation Forest 3

│

└── Channel N → Isolation Forest N
```

Advantages:

- Channel-specific learning
- Better scalability
- Independent anomaly detection
- Simplified maintenance

Each model learns the normal behavior of its corresponding telemetry channel.

---

# 7. Prediction Pipeline

When new telemetry arrives, the inference process follows the same sequence used during training.

```text
New Telemetry

↓

Sliding Window

↓

Feature Extraction

↓

StandardScaler

↓

Isolation Forest

↓

Anomaly Score

↓

Prediction
```

This guarantees consistency between training and deployment.

---

# 8. Evaluation Module

The evaluation module measures model performance using benchmark datasets.

Metrics include:

- Precision
- Recall
- F1-score
- ROC Curve
- Precision–Recall Curve
- Error Analysis

The evaluation process helps identify strengths, weaknesses, and opportunities for improvement.

---

# 9. Backend Integration

The trained model can be integrated into backend services through an inference API.

Typical workflow:

```text
Client

↓

REST API

↓

Load Model

↓

Run Prediction

↓

Return JSON Response
```

Example response:

```json
{
  "channel": "A-1",
  "anomaly": true,
  "score": -0.42
}
```

This modular design allows the anomaly detection engine to be integrated into web applications, dashboards, or monitoring systems.

---

# Design Principles

CERT-SAT follows several software engineering principles.

## Modularity

Each module performs one well-defined task.

## Reusability

Components can be reused independently in other projects.

## Scalability

Additional telemetry channels can be added without changing the overall architecture.

## Maintainability

Independent modules simplify debugging and future development.

## Consistency

Training and inference use the same preprocessing pipeline to avoid discrepancies.

---

# End-to-End Data Flow

```text
Raw Telemetry
        │
        ▼
Load Dataset
        │
        ▼
Preprocessing
        │
        ▼
Sliding Windows
        │
        ▼
Feature Extraction
        │
        ▼
StandardScaler
        │
        ▼
Isolation Forest Models
        │
        ▼
Anomaly Scores
        │
        ▼
Prediction
        │
        ▼
Evaluation / API Response
```

---

# Summary

The architecture of CERT-SAT is designed around a modular pipeline that transforms raw telemetry into meaningful anomaly predictions. By separating data preparation, feature engineering, model training, and inference into independent components, the framework remains scalable, maintainable, and easy to integrate into production environments. The next document explains the preprocessing pipeline and feature engineering process in detail.
