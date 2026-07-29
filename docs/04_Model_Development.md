# Model Development

> This document describes the design, implementation, training process, and engineering decisions behind the Multi-Channel Isolation Forest model used in CERT-SAT.

---

# Overview

The objective of the model is to detect anomalous behaviour in spacecraft telemetry without requiring labelled anomaly data during training.

Instead of relying on predefined thresholds or supervised classification, CERT-SAT uses **Isolation Forest**, an unsupervised anomaly detection algorithm capable of identifying observations that differ significantly from normal operating behaviour.

To improve scalability and interpretability, the project adopts a **Multi-Channel Isolation Forest architecture**, where each telemetry channel is modelled independently.

---

# Why Isolation Forest?

Anomaly detection differs from traditional classification problems because anomalous events are:

- Rare
- Diverse
- Difficult to label
- Often previously unseen

These characteristics make supervised learning unsuitable in many real-world telemetry applications.

Isolation Forest addresses this problem by learning the structure of **normal data** instead of learning examples of anomalies.

---

# Algorithm Intuition

Isolation Forest is based on a simple observation:

> Anomalies are easier to isolate than normal observations.

Normal observations belong to dense regions of the feature space.

Anomalies usually appear far from these dense regions.

Instead of measuring distance or estimating probability distributions, Isolation Forest repeatedly partitions the feature space using randomly selected features and split values.

Points that become isolated after only a few splits are likely to be anomalous.

---

# Isolation Process

```
Entire Dataset
      │
      ▼
Random Feature Selection
      │
      ▼
Random Split Value
      │
      ▼
Partition Dataset
      │
      ▼
Repeat Until Observation Isolated
      │
      ▼
Path Length
      │
      ▼
Anomaly Score
```

The average path length across many random trees determines how anomalous a sample is.

---

# Why Path Length Matters

Consider two observations:

```
Normal Observation

□□□□□□□□□□□□□□

Requires many splits

↓

Long path

↓

Normal



Anomalous Observation

■

Isolated quickly

↓

Short path

↓

Anomaly
```

The shorter the average path length, the more anomalous the observation.

---

# Why Multi-Channel Instead of One Global Model?

Spacecraft telemetry contains many independent sensor channels.

Each channel measures different physical properties.

Examples include:

- Battery voltage
- Temperature
- Pressure
- Current
- Power
- Communication signals

These sensors exhibit very different statistical behaviour.

Training one global model would force all channels to share a common representation, making it harder to detect channel-specific anomalies.

CERT-SAT instead trains one Isolation Forest per channel.

---

# Multi-Channel Architecture

```
Telemetry

├── Channel 1
│      │
│      ▼
│ Isolation Forest
│
├── Channel 2
│      │
│      ▼
│ Isolation Forest
│
├── Channel 3
│      │
│      ▼
│ Isolation Forest
│
└── Channel N
       │
       ▼
 Isolation Forest
```

Each model learns the normal operating behaviour of only one telemetry channel.

This design improves both scalability and interpretability.

---

# Model Pipeline

For every telemetry channel, the following steps are performed.

```
Raw Telemetry

↓

Sliding Window

↓

Feature Engineering

↓

StandardScaler

↓

Isolation Forest

↓

Anomaly Score
```

---

# Input Representation

The Isolation Forest does not receive raw telemetry values.

Instead, each sliding window is converted into a statistical feature vector.

Example

```
Window

[10,11,12,13,15]

↓

Feature Vector

Mean

Standard Deviation

Minimum

Maximum

Trend
```

This representation captures the behaviour of the signal while reducing dimensionality.

---

# Training Procedure

The training process consists of the following stages.

## Step 1

Load telemetry data.

---

## Step 2

Generate sliding windows.

---

## Step 3

Extract statistical features.

---

## Step 4

Normalize features using StandardScaler.

---

## Step 5

Train an independent Isolation Forest for each channel.

---

## Step 6

Save:

- trained models
- fitted scalers
- configuration

using Joblib.

---

# Hyperparameters

The behaviour of Isolation Forest depends on several important parameters.

| Hyperparameter | Purpose |
|---------------|----------|
| n_estimators | Number of isolation trees |
| contamination | Expected anomaly proportion |
| max_samples | Samples used to build each tree |
| random_state | Ensures reproducibility |
| bootstrap | Controls sampling strategy |

The selected values were determined through experimentation and validation.

---

# Feature Scaling

Although Isolation Forest is tree-based and relatively insensitive to feature scaling, CERT-SAT standardizes statistical features before training.

Reasons include:

- consistent preprocessing
- stable numerical ranges
- reusable inference pipeline
- compatibility with future models

---

# Model Persistence

Training can be computationally expensive.

Instead of retraining the model every time the application starts, CERT-SAT stores trained artifacts using Joblib.

Saved artifacts include:

- Isolation Forest models
- StandardScaler objects
- configuration metadata

This allows predictions to be performed immediately after loading the model.

---

# Inference Workflow

```
New Telemetry

↓

Window Generation

↓

Feature Extraction

↓

Saved StandardScaler

↓

Saved Isolation Forest

↓

Anomaly Score

↓

Prediction
```

The inference pipeline mirrors the training pipeline to ensure consistent predictions.

---

# Advantages of the Architecture

The Multi-Channel design offers several advantages.

## Scalability

New telemetry channels can be added independently.

---

## Maintainability

Each channel can be retrained without affecting the others.

---

## Interpretability

Anomalies can be traced back to individual sensors.

---

## Fault Isolation

Problems within one subsystem do not interfere with anomaly detection in unrelated subsystems.

---

## Computational Efficiency

Channels can be processed in parallel, making the architecture suitable for large telemetry systems.

---

# Limitations

While Isolation Forest performs well in many scenarios, it also has limitations.

- Does not explicitly model temporal dependencies.
- Sensitive to feature quality.
- Requires careful window selection.
- May struggle with extremely subtle anomalies.

These limitations motivate future exploration of sequence-based models such as LSTM Encoder–Decoder architectures.

---

# Future Improvements

Potential enhancements include:

- Transformer-based anomaly detection
- Autoencoder architectures
- Online learning
- Adaptive contamination estimation
- Ensemble anomaly detection
- Explainable AI techniques

---

# Summary

The Multi-Channel Isolation Forest architecture forms the core of CERT-SAT. By combining statistical feature engineering, channel-specific modelling, and unsupervised learning, the framework provides an efficient and scalable solution for detecting anomalies in spacecraft telemetry. The modular design also enables straightforward deployment, maintenance, and future extension.
