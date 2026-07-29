# Model Evaluation

> This document presents the evaluation methodology, performance metrics, prediction interpretation, comparative analysis, and limitations of the CERT-SAT anomaly detection framework.

---

# Overview

A machine learning model is only valuable if its predictions are reliable. Therefore, evaluating the anomaly detection model is a critical part of the CERT-SAT pipeline.

Unlike supervised classification tasks, anomaly detection presents unique evaluation challenges because anomalies are rare, often imbalanced, and may vary significantly in duration and severity.

The evaluation process aims to answer the following questions:

- Does the model correctly identify anomalous behaviour?
- How often does it generate false alarms?
- How many anomalies are missed?
- Is Isolation Forest an appropriate choice for this problem?
- How should prediction scores be interpreted?

---

# Evaluation Pipeline

The evaluation process follows these stages:

```text
Test Dataset
      │
      ▼
Generate Predictions
      │
      ▼
Compare with Ground Truth
      │
      ▼
Calculate Metrics
      │
      ▼
Analyze Errors
      │
      ▼
Interpret Results
```

---

# Evaluation Metrics

Since anomaly detection is an imbalanced classification problem, relying solely on accuracy is misleading. Instead, CERT-SAT evaluates performance using Precision, Recall, and F1-score.

## Precision

Precision measures the proportion of predicted anomalies that are actually anomalous.

\[
Precision = \frac{TP}{TP + FP}
\]

Where:

- TP = True Positives
- FP = False Positives

A high Precision indicates that when the model predicts an anomaly, it is usually correct.

---

## Recall

Recall measures the proportion of actual anomalies that are successfully detected.

\[
Recall = \frac{TP}{TP + FN}
\]

Where:

- FN = False Negatives

High Recall means fewer anomalies are missed.

---

## F1-score

The F1-score is the harmonic mean of Precision and Recall.

\[
F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}
\]

The F1-score provides a balanced measure when both false positives and false negatives are important.

---

# Why Not Accuracy?

Consider the following example:

- Total observations: 10,000
- Actual anomalies: 100
- Normal observations: 9,900

A model that predicts every observation as normal achieves:

```
Accuracy = 99%
```

Despite this impressive accuracy, the model completely fails to detect anomalies.

This illustrates why Precision, Recall, and F1-score are more meaningful for anomaly detection.

---

# Confusion Matrix

The confusion matrix summarizes the model's predictions.

|                | Predicted Normal | Predicted Anomaly |
|----------------|-----------------:|------------------:|
| Actual Normal  | True Negative (TN) | False Positive (FP) |
| Actual Anomaly | False Negative (FN) | True Positive (TP) |

Each outcome provides insight into the model's strengths and weaknesses.

---

# Understanding Prediction Outcomes

## True Positive (TP)

An anomaly is correctly detected.

Example:

A spacecraft sensor exhibits abnormal behaviour, and the model correctly flags it.

---

## False Positive (FP)

A normal observation is incorrectly classified as anomalous.

Consequence:

- unnecessary investigation
- increased operational cost
- alarm fatigue

---

## False Negative (FN)

A real anomaly is missed.

Consequence:

- undetected system failure
- potential mission risk
- delayed intervention

False negatives are generally considered more critical than false positives in spacecraft monitoring.

---

## True Negative (TN)

Normal behaviour is correctly identified.

---

# Prediction Interpretation

Isolation Forest produces an anomaly score for every observation.

Conceptually:

```text
Very Negative Score
        │
        ▼
Highly Anomalous


Near Threshold
        │
        ▼
Borderline


Positive Score
        │
        ▼
Likely Normal
```

The exact threshold depends on the model configuration and contamination parameter.

Lower scores generally indicate observations that are easier to isolate and are therefore more likely to be anomalous.

---

# Example Prediction

| Window | Anomaly Score | Prediction |
|--------:|--------------:|-----------|
| 1 | -0.62 | Anomaly |
| 2 | -0.47 | Anomaly |
| 3 | -0.08 | Borderline |
| 4 | 0.15 | Normal |
| 5 | 0.32 | Normal |

The anomaly score provides a measure of confidence rather than a simple binary label.

---

# Error Analysis

No anomaly detection model is perfect. Understanding incorrect predictions is essential for improving performance.

## Causes of False Positives

False positives may occur when:

- normal operating conditions differ significantly from the training data
- sensors experience temporary fluctuations
- statistical features resemble anomalous patterns

Reducing false positives helps prevent unnecessary alerts.

---

## Causes of False Negatives

False negatives may occur when:

- anomalies are subtle
- abnormal behaviour closely resembles normal operation
- the selected window size smooths short-duration anomalies
- statistical features fail to capture temporal patterns

Reducing false negatives is particularly important in safety-critical systems.

---

# Isolation Forest vs Local Outlier Factor (LOF)

During experimentation, both Isolation Forest and Local Outlier Factor (LOF) were considered.

| Criterion | Isolation Forest | LOF |
|-----------|------------------|-----|
| Learning Type | Unsupervised | Unsupervised |
| Scalability | High | Moderate |
| High-dimensional Data | Good | Less effective |
| Training Speed | Faster | Slower |
| Memory Usage | Lower | Higher |
| Prediction on New Data | Efficient | Less suitable for deployment |

In some experiments, LOF may achieve higher Precision, Recall, or F1-score. However, metric performance alone does not determine the best production model.

Isolation Forest was selected because it:

- scales better to large telemetry datasets
- supports efficient inference
- has lower computational overhead
- is easier to deploy
- aligns well with the project's multi-channel architecture

The final choice balances predictive performance with scalability and maintainability.

---

# Model Strengths

The evaluation demonstrates several strengths of the proposed approach:

- Effective detection of unusual telemetry behaviour
- Efficient processing of high-dimensional sensor data
- Independent modelling of telemetry channels
- Scalable architecture suitable for production deployment
- Fast prediction after training

---

# Current Limitations

The current implementation also has limitations:

- Does not explicitly model temporal dependencies
- Performance depends on feature quality
- Fixed window size may not suit all anomaly types
- Threshold selection may require further tuning
- Extremely subtle anomalies remain challenging

These limitations motivate future research into sequence-based deep learning methods.

---

# Future Evaluation Improvements

Potential future enhancements include:

- Cross-validation across additional datasets
- Adaptive threshold optimization
- Ensemble evaluation
- Explainability techniques
- Real-time performance benchmarking
- Comparison with Transformer and Autoencoder models

---

# Summary

The evaluation confirms that the Multi-Channel Isolation Forest architecture provides an effective and scalable solution for spacecraft telemetry anomaly detection. By analysing Precision, Recall, F1-score, confusion matrices, and error patterns, the project demonstrates both the strengths and limitations of the proposed approach. Although alternative algorithms such as LOF may outperform Isolation Forest on certain metrics, Isolation Forest offers a better balance between predictive performance, computational efficiency, and production readiness.
