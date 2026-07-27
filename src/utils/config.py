"""
Shared configuration for CERT-SAT anomaly detection project.
Centralizing these means Isolation Forest, LSTM, and Autoencoder
all use the same window size / stride / paths, so results stay comparable.
"""

import os

# ---- Dataset paths ----
# Set via kagglehub.dataset_download(...) at runtime, or override here
# if you've already downloaded the dataset locally.
DATASET_SLUG = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"

DOWNLOAD_PATH = None  # populated at runtime by data/loader.py

def get_paths(download_path):
    """Given the kagglehub download path, return standard sub-paths."""
    labels_csv = os.path.join(download_path, "labeled_anomalies.csv")
    base_npy_dir = os.path.join(download_path, "data", "data")
    train_dir = os.path.join(base_npy_dir, "train")
    test_dir = os.path.join(base_npy_dir, "test")
    return {
        "labels_csv": labels_csv,
        "base_npy_dir": base_npy_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
    }

# ---- Windowing ----
# WINDOW_SIZE is derived from the median anomaly sequence length (~120
# timesteps) found during EDA. STRIDE=10 keeps runtime reasonable across
# 82 channels while still giving decent temporal resolution.
WINDOW_SIZE = 120
STRIDE = 10

# ---- Isolation Forest ----
IF_N_ESTIMATORS = 100
IF_RANDOM_STATE = 42
IF_DEFAULT_CONTAMINATION = 0.05  # fallback when a channel has no lookup value
IF_CONTAMINATION_MIN = 0.001
IF_CONTAMINATION_MAX = 0.5

# ---- Evaluation ----
# NOTE: contamination is currently derived from imbalance_df, which uses
# TRUE test-set anomaly fractions. This is a form of label leakage into a
# hyperparameter and should be flagged as a known limitation, or replaced
# with IF_DEFAULT_CONTAMINATION for a leakage-free baseline comparison.