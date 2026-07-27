"""
Data loading for the NASA SMAP/MSL anomaly detection dataset.

Handles:
- downloading the dataset via kagglehub
- loading and parsing labeled_anomalies.csv
- loading per-channel .npy train/test arrays
- building the shape_df summary (lengths, feature counts, NaN/Inf checks)
"""

import os
import ast
import numpy as np
import pandas as pd
import kagglehub

from src.utils.config import DATASET_SLUG, get_paths


def download_dataset():
    """Downloads (or retrieves cached) SMAP/MSL dataset. Returns path dict."""
    download_path = kagglehub.dataset_download(DATASET_SLUG)
    return get_paths(download_path)


def load_labels(labels_csv):
    """Load labeled_anomalies.csv and parse anomaly_sequences from string to list."""
    labels = pd.read_csv(labels_csv)
    labels["anomaly_sequences"] = labels["anomaly_sequences"].apply(ast.literal_eval)
    return labels


def load_channel_arrays(ch_id, train_dir, test_dir):
    """Load a single channel's train/test .npy arrays. Returns (train_arr, test_arr) or (None, None) if missing."""
    train_path = os.path.join(train_dir, f"{ch_id}.npy")
    test_path = os.path.join(test_dir, f"{ch_id}.npy")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None, None
    return np.load(train_path), np.load(test_path)


def build_shape_summary(labels, train_dir, test_dir):
    """
    Per-channel summary: train/test lengths, feature count, anomaly seq count,
    and NaN/Inf flags. Used to sanity-check the dataset before modeling and
    to decide on windowing strategy.
    """
    shape_summary = []
    for _, row in labels.iterrows():
        ch_id = row["chan_id"]
        train_arr, test_arr = load_channel_arrays(ch_id, train_dir, test_dir)
        if train_arr is None:
            continue

        nan_inf_flags = {
            "train_has_nan": np.isnan(train_arr.astype(float)).any(),
            "train_has_inf": np.isinf(train_arr.astype(float)).any(),
            "test_has_nan": np.isnan(test_arr.astype(float)).any(),
            "test_has_inf": np.isinf(test_arr.astype(float)).any(),
        }
        shape_summary.append({
            "ch_id": ch_id,
            "spacecraft": row["spacecraft"],
            "train_len": train_arr.shape[0],
            "test_len": test_arr.shape[0],
            "n_features": train_arr.shape[1],
            "n_anomaly_seqs": len(row["anomaly_sequences"]),
            **nan_inf_flags,
        })
    return pd.DataFrame(shape_summary)


def build_imbalance_summary(labels, shape_df):
    """
    Per-channel fraction of test timesteps labeled anomalous.
    Useful for understanding class imbalance severity across channels.

    CAUTION: if used to set IsolationForest's `contamination` parameter,
    this leaks true test-set label info into a hyperparameter. Fine for
    EDA / oracle-upper-bound comparisons, but flag this if used in the
    "real" baseline run.
    """
    def anomaly_fraction(row, test_len):
        total_anom_points = sum(end - start for start, end in row["anomaly_sequences"])
        return total_anom_points / test_len

    imbalance = []
    for _, row in labels.iterrows():
        match = shape_df[shape_df["ch_id"] == row["chan_id"]]
        if match.empty:
            continue
        test_len = match["test_len"].values[0]
        frac = anomaly_fraction(row, test_len)
        imbalance.append({"ch_id": row["chan_id"], "anomaly_fraction": frac})
    return pd.DataFrame(imbalance)
