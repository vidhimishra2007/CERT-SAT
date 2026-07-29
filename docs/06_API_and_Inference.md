# Deployment and Inference Pipeline

> This document explains how the trained CERT-SAT model is deployed, how predictions are generated, and how the anomaly detection pipeline can be integrated into backend applications.

---

# Overview

After training, the Multi-Channel Isolation Forest models are serialized and stored for later use. During deployment, these saved models are loaded into memory and used to analyze incoming telemetry without retraining.

The deployment pipeline ensures that the same preprocessing, feature extraction, and scaling operations used during training are applied during inference. This consistency is essential for producing reliable predictions.

---

# Deployment Architecture

The deployment workflow consists of the following components.

```text
                 Incoming Telemetry
                         │
                         ▼
                 Preprocessing Module
                         │
                         ▼
             Sliding Window Generation
                         │
                         ▼
              Statistical Feature Extraction
                         │
                         ▼
              Load Saved StandardScaler
                         │
                         ▼
         Transform Feature Vectors
                         │
                         ▼
        Load Multi-Channel Isolation Forest
                         │
                         ▼
             Channel-wise Predictions
                         │
                         ▼
              Aggregate Results
                         │
                         ▼
                 API Response
```

Unlike the training pipeline, deployment only performs inference using previously trained artifacts.

---

# Saved Model Artifacts

Training produces several reusable artifacts.

| Artifact | Purpose |
|-----------|---------|
| Isolation Forest Models | Predict anomalies for each telemetry channel |
| StandardScaler Objects | Normalize incoming features |
| Configuration File | Stores model metadata and parameters |

These artifacts are loaded during application startup and reused for every prediction request.

---

# Why Save the Model?

Retraining a model every time the application starts would be inefficient.

Saving the trained artifacts provides several benefits:

- Faster application startup
- Consistent predictions
- Lower computational cost
- Easier deployment
- Improved reproducibility

---

# Inference Workflow

The inference process mirrors the preprocessing steps used during training.

```text
Incoming Telemetry
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
Isolation Forest Prediction
        │
        ▼
Anomaly Score
        │
        ▼
Prediction Label
```

Maintaining identical preprocessing during training and inference ensures that the model receives data in the same format.

---

# Step 1 — Receive Telemetry

The application receives telemetry data from an external source.

Possible sources include:

- REST API
- CSV files
- Streaming services
- IoT gateways
- Telemetry databases

Each telemetry record is grouped by channel before processing.

---

# Step 2 — Generate Sliding Windows

Incoming telemetry is divided into overlapping windows.

Each window represents a short segment of the telemetry signal and forms a single prediction sample.

---

# Step 3 — Extract Statistical Features

For every window, statistical descriptors are calculated.

Typical features include:

- Mean
- Standard deviation
- Minimum
- Maximum
- Trend

The resulting feature vector becomes the input to the trained model.

---

# Step 4 — Normalize Features

The saved StandardScaler transforms the extracted features.

**Important:** During inference, the scaler is **not fitted again**. It only applies the parameters learned during training.

This guarantees consistency between training and prediction.

---

# Step 5 — Generate Predictions

Each telemetry channel has its own trained Isolation Forest model.

```text
Channel A
        │
        ▼
Isolation Forest A
        │
        ▼
Score A


Channel B
        │
        ▼
Isolation Forest B
        │
        ▼
Score B
```

Each model independently computes an anomaly score.

---

# Step 6 — Interpret Scores

The anomaly score indicates how unusual the observation is.

| Score | Interpretation |
|--------|----------------|
| Very Low | Strong anomaly |
| Near Threshold | Borderline behaviour |
| High | Normal behaviour |

The application converts these scores into prediction labels using the configured decision threshold.

---

# API Integration

CERT-SAT can be integrated into a backend service through a prediction endpoint.

A typical request contains:

```json
{
  "telemetry": [
    [12.3, 12.5, 12.8, 13.1],
    [44.2, 44.0, 43.9, 43.8]
  ]
}
```

A typical response:

```json
{
  "predictions": [
    {
      "channel": "Temperature",
      "score": -0.58,
      "prediction": "Anomaly"
    },
    {
      "channel": "Pressure",
      "score": 0.27,
      "prediction": "Normal"
    }
  ]
}
```

---

# Error Handling

The inference service should validate incoming data before generating predictions.

Recommended checks include:

- Missing telemetry values
- Invalid channel counts
- Incorrect feature dimensions
- Empty input
- Corrupted model files

Providing clear error messages simplifies debugging and integration.

---

# Performance Considerations

Several design choices improve runtime performance:

- Load models once during application startup.
- Reuse the fitted StandardScaler.
- Process channels independently.
- Batch predictions where possible.
- Cache reusable resources.

These optimizations reduce latency and improve scalability.

---

# Deployment Scenarios

The modular architecture allows CERT-SAT to be deployed in multiple environments.

Examples include:

- Research notebooks
- Command-line tools
- REST APIs
- Cloud services
- Real-time monitoring systems
- Edge computing environments

---

# Security Considerations

When exposing the model through an API:

- Validate all user input.
- Restrict access to prediction endpoints.
- Log prediction requests responsibly.
- Protect serialized model files.
- Monitor API performance.

These practices improve reliability and operational security.

---

# Summary

The deployment pipeline transforms incoming telemetry into anomaly predictions using the trained Multi-Channel Isolation Forest models. By reusing the preprocessing pipeline and serialized artifacts from training, CERT-SAT delivers efficient, reproducible, and scalable anomaly detection suitable for integration into production applications.
