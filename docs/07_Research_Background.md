# Research Background and Literature Review

> This document reviews the evolution of anomaly detection techniques, compares major approaches used for multivariate time-series data, and explains the motivation behind the design decisions adopted in CERT-SAT.

---

# Overview

Anomaly detection has been an active research area for several decades and is widely used in domains such as aerospace, cybersecurity, finance, manufacturing, healthcare, and Internet of Things (IoT).

Unlike conventional classification tasks, anomaly detection focuses on identifying observations that significantly deviate from normal behaviour. Since anomalous events are rare and often unlabeled, supervised learning is not always practical. Consequently, researchers have developed statistical, machine learning, and deep learning techniques to detect unusual patterns without relying heavily on labelled data.

CERT-SAT builds upon this body of research by adopting a scalable machine learning approach for spacecraft telemetry.

---

# Evolution of Anomaly Detection

The development of anomaly detection methods can be broadly divided into three generations.

```text
Statistical Methods
        │
        ▼
Machine Learning Methods
        │
        ▼
Deep Learning Methods
```

Each generation addresses different challenges and introduces new capabilities.

---

# Statistical Methods

Early anomaly detection techniques relied on statistical assumptions about the underlying data distribution.

Examples include:

- Z-score
- Gaussian models
- Control charts
- Moving averages

These methods classify observations as anomalous when they deviate significantly from expected statistical behaviour.

### Advantages

- Simple to implement
- Fast execution
- Minimal computational requirements
- Easy interpretation

### Limitations

- Assume known probability distributions
- Sensitive to noise
- Perform poorly on complex multivariate datasets
- Limited ability to detect nonlinear relationships

For modern spacecraft telemetry containing many correlated sensors, statistical methods are generally insufficient.

---

# Machine Learning Approaches

Machine learning techniques overcome many limitations of statistical models by learning patterns directly from data.

Popular algorithms include:

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- DBSCAN
- K-Means based methods

These algorithms are capable of detecting complex anomalies without explicitly modelling probability distributions.

---

# Isolation Forest

Isolation Forest is an ensemble-based anomaly detection algorithm introduced by Liu et al. in 2008.

Unlike density-based methods, Isolation Forest identifies anomalies by recursively partitioning the feature space using random splits.

Observations requiring fewer partitions to become isolated are considered anomalous.

### Advantages

- Computationally efficient
- Scales well to large datasets
- Suitable for high-dimensional features
- Minimal parameter tuning
- Fast prediction

### Limitations

- Does not model temporal dependencies
- Performance depends on feature quality
- Sensitive to preprocessing choices

Because of its scalability and efficiency, Isolation Forest forms the foundation of CERT-SAT.

---

# Local Outlier Factor (LOF)

LOF identifies anomalies by comparing the local density of each observation with that of its neighbours.

Points located in regions with significantly lower density receive higher anomaly scores.

### Advantages

- Effective for local anomalies
- Handles irregular data distributions
- Captures neighbourhood relationships

### Limitations

- Computationally expensive
- Sensitive to neighbourhood size
- Less scalable for very large datasets
- Higher memory requirements

Although LOF produced competitive results during experimentation, its computational cost and deployment limitations made Isolation Forest a more suitable choice for CERT-SAT.

---

# Deep Learning Approaches

Recent research has increasingly focused on deep learning models capable of learning temporal patterns directly from sequential data.

Common architectures include:

- Autoencoders
- Variational Autoencoders
- LSTM Networks
- GRU Networks
- Transformer Models

These approaches eliminate much of the manual feature engineering required by traditional machine learning.

---

# LSTM Encoder–Decoder

The LSTM Encoder–Decoder architecture is designed to reconstruct sequential input data.

The model consists of two components:

```text
Input Sequence

↓

Encoder

↓

Latent Representation

↓

Decoder

↓

Reconstructed Sequence
```

If the reconstruction error exceeds a predefined threshold, the observation is classified as anomalous.

### Strengths

- Captures temporal dependencies
- Suitable for sequential data
- Learns complex nonlinear relationships

### Weaknesses

- Requires extensive training
- Computationally intensive
- More difficult to deploy
- Larger memory footprint

These characteristics make LSTM models attractive for future work but less practical for lightweight deployment.

---

# Hundman et al. (2018)

Hundman et al. introduced one of the most influential deep learning approaches for spacecraft telemetry anomaly detection.

Their work proposed an LSTM Encoder–Decoder model trained on NASA telemetry streams.

The key contributions include:

- Sequence reconstruction
- Dynamic thresholding
- Handling multivariate telemetry
- Evaluation using real spacecraft data

### Relevance to CERT-SAT

Hundman's work demonstrates the effectiveness of sequence-based deep learning models for spacecraft telemetry.

However, the approach requires significantly more computational resources than Isolation Forest.

CERT-SAT instead focuses on a lightweight machine learning solution that is easier to train and deploy.

---

# Chandola et al. (2009)

Chandola et al. published one of the most comprehensive surveys on anomaly detection.

The paper categorizes anomaly detection into:

- Point anomalies
- Contextual anomalies
- Collective anomalies

It also reviews statistical, proximity-based, clustering, classification, and information-theoretic techniques.

This survey provides the theoretical foundation for understanding anomaly detection and influenced the overall design of CERT-SAT.

---

# Comparative Analysis

| Method | Training Data | Temporal Learning | Scalability | Deployment Complexity |
|---------|---------------|------------------|-------------|----------------------|
| Statistical Methods | No | No | High | Low |
| Isolation Forest | Normal Data | No | High | Low |
| LOF | Normal Data | No | Medium | Medium |
| One-Class SVM | Normal Data | No | Medium | Medium |
| LSTM Encoder–Decoder | Large Dataset | Yes | Medium | High |
| Transformer | Large Dataset | Yes | Medium | Very High |

---

# Why CERT-SAT Uses Isolation Forest

Several factors influenced the selection of Isolation Forest.

### Scalability

Each telemetry channel can be processed independently.

---

### Fast Training

Isolation Forest trains significantly faster than deep learning models.

---

### Efficient Inference

Prediction latency is low, making the model suitable for deployment.

---

### Low Computational Requirements

The algorithm performs well without requiring GPU acceleration.

---

### Modular Architecture

Independent models simplify maintenance and future improvements.

---

### Compatibility

Isolation Forest integrates naturally with the sliding-window feature engineering pipeline adopted by CERT-SAT.

---

# Future Research Directions

Potential future improvements include:

- Transformer-based anomaly detection
- Graph Neural Networks for sensor relationships
- Hybrid Isolation Forest + LSTM models
- Adaptive threshold estimation
- Explainable AI techniques
- Online learning
- Self-supervised representation learning

---

# Key Takeaways

The literature demonstrates that no single anomaly detection technique is universally optimal.

Classical statistical methods offer simplicity but struggle with complex telemetry.

Deep learning methods capture temporal dependencies but require greater computational resources.

Isolation Forest provides an effective compromise between predictive performance, computational efficiency, scalability, and ease of deployment. These characteristics make it well suited for the objectives of CERT-SAT while leaving opportunities for future exploration of more advanced sequence-based models.

---

# References

This section should include the full bibliographic references for all research papers, datasets, and software libraries cited throughout the documentation.

Examples include:

- Liu, Ting & Zhou (Isolation Forest)
- Chandola et al. (2009)
- Hundman et al. (2018)
- NASA SMAP/MSL Dataset
- Scikit-learn Documentation
