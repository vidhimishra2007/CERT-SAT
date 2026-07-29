# Data Pipeline and Feature Engineering

> This document describes how raw spacecraft telemetry is transformed into machine learning-ready feature vectors. It covers dataset organization, preprocessing, sliding window generation, statistical feature engineering, and feature normalization.

---

# Overview

Machine learning models cannot directly consume continuous telemetry streams. Raw spacecraft telemetry consists of long sequences of sensor measurements collected over time, often containing noise, varying scales, and temporal dependencies.

CERT-SAT converts this raw telemetry into compact numerical feature vectors through a structured preprocessing pipeline. This transformation improves computational efficiency while preserving the statistical characteristics required for anomaly detection.

The preprocessing pipeline consists of five major stages:

1. Dataset Loading
2. Data Preparation
3. Sliding Window Generation
4. Statistical Feature Extraction
5. Feature Scaling

---

# Data Pipeline

```text
Raw Telemetry
      │
      ▼
Load Dataset
      │
      ▼
Data Preparation
      │
      ▼
Sliding Window Generation
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Machine Learning Features
```

Each stage prepares the data for the next stage, ensuring consistency between model training and inference.

---

# Dataset

CERT-SAT uses the publicly available NASA spacecraft telemetry benchmark datasets.

## SMAP

The **Soil Moisture Active Passive (SMAP)** dataset contains telemetry collected from the SMAP Earth observation satellite. The telemetry consists of multiple sensor channels that monitor spacecraft health and operational status.

---

## MSL

The **Mars Science Laboratory (MSL)** dataset contains telemetry from the Curiosity rover mission. Like SMAP, it includes multiple telemetry channels with labelled anomaly intervals for evaluation.

---

## Why These Datasets?

The NASA datasets are widely used because they:

- represent real spacecraft telemetry
- contain multivariate time-series
- include realistic anomaly intervals
- are commonly used for benchmarking anomaly detection algorithms
- enable comparison with previous research

---

# Data Structure

Each telemetry channel is represented as a continuous numerical sequence.

Example:

```text
Time

0
1
2
3
4
5
6
...

Temperature

20.3
20.2
20.4
20.7
21.1
20.9
20.8
...
```

Each channel is processed independently throughout the pipeline.

---

# Data Preparation

Before feature extraction, the telemetry is prepared for analysis.

Typical preprocessing tasks include:

- loading telemetry files
- preserving chronological order
- separating telemetry channels
- validating input dimensions
- handling missing or invalid values (if present)

The goal is to ensure that every channel is represented as a clean and continuous time series.

---

# Why Sliding Windows?

A telemetry stream may contain thousands of observations.

Instead of treating the entire sequence as a single sample, CERT-SAT divides it into overlapping windows.

This enables the model to analyse local behaviour over time.

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

Each window becomes one training sample.

---

# Window Parameters

Sliding windows are defined using two parameters.

## Window Size

The number of consecutive observations contained in each window.

Example

```text
Window Size = 120

□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
```

Larger windows capture long-term behaviour but increase computation.

Smaller windows provide more localized analysis but may lose long-term context.

---

## Stride

Stride determines how far the window moves after each observation.

Example

```text
Stride = 10

Window 1

□□□□□□□□

Window 2

          □□□□□□□□

Window 3

                    □□□□□□□□
```

Overlapping windows increase the number of training samples while preserving temporal continuity.

---

# Feature Engineering

Machine learning models perform better when each window is represented by descriptive statistics instead of hundreds of raw values.

CERT-SAT computes statistical features for every sliding window.

Typical features include:

| Feature | Description |
|----------|-------------|
| Mean | Average sensor value |
| Standard Deviation | Signal variability |
| Minimum | Lowest value |
| Maximum | Highest value |
| Trend / Difference | Overall change within the window |

Example:

Window

```text
[10, 11, 13, 12, 14]
```

Extracted features

```text
Mean = 12.0

Std = 1.58

Min = 10

Max = 14

Trend = +4
```

The resulting feature vector is much smaller than the original telemetry window while preserving its overall behaviour.

---

# Why Statistical Features?

Using statistical summaries provides several advantages.

- reduces dimensionality
- lowers computational cost
- removes redundant information
- improves model efficiency
- captures the overall characteristics of each window

These features are particularly effective for tree-based anomaly detection algorithms such as Isolation Forest.

---

# Feature Scaling

The extracted features have different numerical ranges.

Example:

```text
Mean = 425

Std = 7

Range = 63
```

Without normalization, larger numerical values may dominate the feature space.

CERT-SAT therefore applies **StandardScaler**.

The scaler computes

```text
Scaled Feature

=

(Value − Mean)

──────────────

Standard Deviation
```

After scaling, every feature has approximately:

- mean = 0
- standard deviation = 1

This ensures that all features contribute equally during training.

---

# Training vs Inference

Exactly the same preprocessing pipeline is used during prediction.

```text
Training

Telemetry

↓

Windows

↓

Features

↓

StandardScaler (Fit)

↓

Isolation Forest



Inference

Telemetry

↓

Windows

↓

Features

↓

Saved StandardScaler (Transform)

↓

Saved Isolation Forest
```

Reusing the same scaler prevents inconsistencies between training and deployment.

---

# Output of the Pipeline

After preprocessing, every telemetry window is converted into a compact feature vector.

Example

```text
Input

120 Raw Observations

↓

Output

[Mean,
Std,
Min,
Max,
Trend]
```

These feature vectors become the input to the machine learning model.

---

# Design Considerations

The preprocessing pipeline was designed with the following goals:

- preserve temporal behaviour
- reduce computational complexity
- generate consistent feature vectors
- support scalable multi-channel processing
- ensure reproducible inference

---

# Summary

The data pipeline transforms continuous spacecraft telemetry into standardized statistical feature vectors suitable for anomaly detection. By combining sliding window generation, statistical feature engineering, and feature normalization, CERT-SAT creates compact representations that retain meaningful signal characteristics while enabling efficient training and inference. The next chapter explains how these feature vectors are used to train the Multi-Channel Isolation Forest model.
