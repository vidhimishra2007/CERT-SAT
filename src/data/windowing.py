"""
Sliding-window utilities shared across Isolation Forest, LSTM, and Autoencoder.

Two windowing strategies are provided:
- create_windows: flattens raw values in each window (high-dimensional,
  rarely used directly for IF since it scales poorly with window_size * n_features)
- create_windows_stats: extracts summary statistics per window (mean/std/min/max/delta),
  which is what the Isolation Forest baseline actually uses

Also includes conversion helpers between window-level predictions/labels
and point-level (per-timestep) predictions/labels, since ground truth
anomaly_sequences are given as point-level ranges.
"""

import numpy as np


def create_windows(arr, window_size, stride):
    """Flatten each window into a single feature vector. Shape: (n_windows, window_size * n_features)."""
    n_timesteps = arr.shape[0]
    windows = []
    for start in range(0, n_timesteps - window_size + 1, stride):
        windows.append(arr[start:start + window_size].flatten())
    return np.array(windows)


def create_windows_stats(arr, window_size, stride=1):
    """
    Summary-statistic windowing: for each window, compute mean/std/min/max
    per feature plus the first-to-last delta. Much lower-dimensional than
    create_windows, and what the IF baseline is trained on.
    """
    n_timesteps = arr.shape[0]
    windows = []
    for start in range(0, n_timesteps - window_size + 1, stride):
        w = arr[start:start + window_size]
        feats = np.concatenate([
            w.mean(axis=0),
            w.std(axis=0),
            w.min(axis=0),
            w.max(axis=0),
            (w[-1] - w[0]),
        ])
        windows.append(feats)
    return np.array(windows)


def sequences_to_point_labels(anomaly_sequences, test_len):
    """Convert ground-truth [start, end] anomaly sequences into a point-level binary label array."""
    point_true = np.zeros(test_len)
    for start, end in anomaly_sequences:
        point_true[start:min(end, test_len)] = 1
    return point_true


def windows_to_point_labels(preds, test_len, window_size, stride):
    """
    Convert window-level IsolationForest predictions (-1 = anomaly, 1 = normal)
    back to point-level binary labels by marking every timestep in an
    anomalous window's span.
    """
    point_pred = np.zeros(test_len)
    for i, p in enumerate(preds):
        if p == -1:
            start = i * stride
            end = min(start + window_size, test_len)
            point_pred[start:end] = 1
    return point_pred
