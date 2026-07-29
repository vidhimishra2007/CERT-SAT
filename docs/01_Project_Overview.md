# Project Overview

## Introduction

CERT-SAT is an end-to-end machine learning framework designed to detect anomalies in multivariate spacecraft telemetry. It analyzes large volumes of sensor data collected from spacecraft, identifies unusual behavior, and helps engineers detect potential system failures before they become critical.

The project combines statistical feature engineering with a Multi-Channel Isolation Forest architecture, allowing each telemetry channel to be modeled independently while maintaining a scalable and efficient anomaly detection pipeline.

Although the framework is demonstrated using NASA spacecraft telemetry datasets, its modular design allows it to be adapted to other multivariate time-series anomaly detection problems such as industrial monitoring, IoT systems, network telemetry, and predictive maintenance.

---

# Problem Statement

Modern spacecraft continuously generate telemetry from hundreds of onboard sensors.

These sensors monitor parameters such as:

- Battery status
- Temperature
- Pressure
- Power consumption
- Communication systems
- Navigation systems
- Instrument health

Since telemetry is collected continuously, manual monitoring is practically impossible.

The challenge is to automatically identify abnormal behaviour without requiring human intervention.

Traditional rule-based monitoring systems depend heavily on manually designed thresholds and often fail to capture subtle or previously unseen anomalies.

CERT-SAT addresses this challenge using an unsupervised machine learning approach capable of detecting unusual patterns directly from telemetry data.

---

# Objectives

The primary objectives of this project are:

- Detect anomalous spacecraft behaviour automatically.
- Learn normal operational patterns without requiring labelled training data.
- Process high-dimensional telemetry efficiently.
- Provide a scalable architecture for multi-channel sensor data.
- Generate anomaly scores for each telemetry channel.
- Enable easy integration into backend applications through a reusable inference pipeline.

---

# Why Anomaly Detection?

Failures in spacecraft systems are often rare, making it difficult to collect sufficient examples of abnormal behaviour for supervised learning.

Furthermore,

- New failure types may never have been observed before.
- Sensor behaviour changes over time.
- Manual threshold selection is difficult.
- High-dimensional telemetry contains complex relationships.

Because of these challenges, unsupervised anomaly detection is particularly well suited for telemetry monitoring.

Instead of learning from labelled anomalies, the model learns what normal behaviour looks like and identifies observations that significantly deviate from it.

---

# Project Scope

CERT-SAT covers the complete anomaly detection lifecycle.

The project includes:

- Dataset preparation
- Telemetry preprocessing
- Sliding window generation
- Statistical feature extraction
- Feature normalization
- Model training
- Model persistence
- Prediction
- Performance evaluation
- Backend integration

This makes the project suitable both for research purposes and production deployment.

---

# Project Workflow

The overall workflow of the system is illustrated below.

```text
NASA Telemetry
       │
       ▼
Raw Data Loading
       │
       ▼
Preprocessing
       │
       ▼
Sliding Window Generation
       │
       ▼
Feature Extraction
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
Evaluation
       │
       ▼
API Integration
```

Each stage of the pipeline is discussed in detail in the subsequent documentation.

---

# Key Features

CERT-SAT provides several capabilities that make it suitable for multivariate anomaly detection.

## Multi-Channel Architecture

Instead of training one model on the entire dataset, an independent Isolation Forest is trained for each telemetry channel.

This allows each sensor to learn its own definition of normal behaviour.

---

## Statistical Feature Engineering

Rather than directly processing raw telemetry sequences, each sliding window is summarized using statistical features.

This reduces dimensionality while preserving important characteristics of the signal.

---

## Sliding Window Processing

Continuous telemetry streams are divided into overlapping windows.

Each window becomes an independent observation that can be processed by the machine learning model.

---

## Unsupervised Learning

The framework does not require labelled anomalies for training.

This makes it practical for real-world environments where anomaly labels are scarce or unavailable.

---

## Modular Design

Each stage of the pipeline is implemented independently.

Examples include:

- Data preprocessing
- Feature extraction
- Model training
- Inference
- Evaluation

This modular architecture simplifies maintenance and future enhancements.

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Dataset | NASA SMAP / MSL |

---

# Project Directory

```text
CERT-SAT/
│
├── data/
├── docs/
├── examples/
├── models_saved/
├── notebooks/
├── src/
├── tests/
└── README.md
```

The source code is organised into independent modules, making the project easy to extend and maintain.

---

# Documentation Guide

The documentation is organised into the following sections:

| Document | Purpose |
|----------|---------|
| 01_Project_Overview | Introduction and project goals |
| 02_System_Architecture | Internal architecture and design |
| 03_Data_and_Preprocessing | Data preparation and feature engineering |
| 04_Model_Development | Isolation Forest implementation |
| 05_Model_Evaluation | Metrics and performance analysis |
| 06_API_and_Inference | Backend integration and prediction |
| 07_Research_Background | Literature review and related work |
| 08_Appendix | Hyperparameters and references |

Readers unfamiliar with the project are encouraged to continue with **02_System_Architecture.md** before exploring the implementation details.

---

# Summary

CERT-SAT is a complete anomaly detection framework for multivariate spacecraft telemetry. By combining sliding-window feature engineering with a Multi-Channel Isolation Forest architecture, it provides an efficient and scalable solution for detecting unusual system behaviour. The following chapters describe the system architecture, preprocessing pipeline, model development, evaluation methodology, and deployment process in detail.
