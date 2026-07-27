"""
Exploratory data analysis utilities for SMAP/MSL channels: distribution
shift between train/test, dead/constant feature detection, and
train-vs-test visualization with anomaly windows highlighted.

These are diagnostic tools, not part of the model pipeline -- use them in
01_eda notebook/script to decide on windowing strategy and per-channel
scaling before modeling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def describe_channel_all_features(ch_id, train_dir, test_dir):
    """
    Quantify train/test distribution shift for one channel, averaged across
    all its features. Large mean_shift or std_ratio far from 1.0 signals
    that a model trained only on nominal (train) data may struggle to
    generalize to test-time distribution.
    """
    train_arr = np.load(os.path.join(train_dir, f"{ch_id}.npy"))
    test_arr = np.load(os.path.join(test_dir, f"{ch_id}.npy"))
    return {
        "ch_id": ch_id,
        "train_mean_avg": train_arr.mean(axis=0).mean(),
        "train_std_avg": train_arr.std(axis=0).mean(),
        "test_mean_avg": test_arr.mean(axis=0).mean(),
        "test_std_avg": test_arr.std(axis=0).mean(),
        "train_min": train_arr.min(),
        "train_max": train_arr.max(),
        "test_min": test_arr.min(),
        "test_max": test_arr.max(),
    }


def build_scale_summary(labels, train_dir, test_dir):
    """Run describe_channel_all_features across all channels and derive mean_shift / std_ratio columns."""
    scale_summary = pd.DataFrame(
        [describe_channel_all_features(c, train_dir, test_dir) for c in labels["chan_id"]]
    )
    scale_summary["mean_shift"] = (
        scale_summary["train_mean_avg"] - scale_summary["test_mean_avg"]
    ).abs()
    scale_summary["std_ratio"] = scale_summary["train_std_avg"] / scale_summary["test_std_avg"]
    return scale_summary


def check_feature_variance(ch_id, train_dir, top_n=5):
    """
    Flag potential dead/constant features (near-zero variance in train).
    These contribute no signal and can be dropped or down-weighted.
    """
    train_arr = np.load(os.path.join(train_dir, f"{ch_id}.npy"))
    variances = pd.DataFrame(train_arr).var().sort_values()
    print(f"\n{ch_id} - lowest variance features (potential dead/constant features):")
    print(variances.head(top_n))
    return variances


def plot_channel(ch_id, labels, train_dir, test_dir, feature_idx=0, save_dir=None):
    """
    Plot a channel's train (nominal-only) and test (anomaly windows shaded
    red) series side by side for a single feature index.
    """
    row = labels[labels["chan_id"] == ch_id].iloc[0]
    train_arr = np.load(os.path.join(train_dir, f"{ch_id}.npy"))
    test_arr = np.load(os.path.join(test_dir, f"{ch_id}.npy"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(train_arr[:, feature_idx], color="steelblue", linewidth=0.8)
    axes[0].set_title(f"{ch_id} - TRAIN (nominal only) - feature {feature_idx}")

    axes[1].plot(test_arr[:, feature_idx], color="steelblue", linewidth=0.8)
    for start, end in row["anomaly_sequences"]:
        axes[1].axvspan(start, end, color="red", alpha=0.3)
    axes[1].set_title(f"{ch_id} - TEST (anomaly windows shaded red) - feature {feature_idx}")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{ch_id}_train_test_plot.png"), dpi=150)
    plt.show()


def diagnose_channel(ch_id, labels, train_dir, test_dir, window_size, stride,
                      contam_lookup, scale_channel_fn, create_windows_stats_fn,
                      sequences_to_point_labels_fn):
    """
    Inspect how well Isolation Forest's anomaly scores separate true-anomaly
    windows from normal windows for one channel. Prints score distribution
    by true label and the rank of true-anomaly windows (lower rank = more
    anomalous = better separation).
    """
    from sklearn.ensemble import IsolationForest

    row = labels[labels["chan_id"] == ch_id].iloc[0]
    train_arr = np.load(os.path.join(train_dir, f"{ch_id}.npy"))
    test_arr = np.load(os.path.join(test_dir, f"{ch_id}.npy"))
    train_scaled, test_scaled = scale_channel_fn(train_arr, test_arr)

    X_train = create_windows_stats_fn(train_scaled, window_size, stride)
    X_test = create_windows_stats_fn(test_scaled, window_size, stride)

    contam_rate = contam_lookup.get(ch_id, 0.05)
    clf = IsolationForest(n_estimators=100, contamination=contam_rate, random_state=42)
    clf.fit(X_train)
    scores = clf.decision_function(X_test)  # lower = more anomalous

    y_true_point = sequences_to_point_labels_fn(row["anomaly_sequences"], test_arr.shape[0])
    window_is_true_anomaly = []
    for i in range(len(X_test)):
        start = i * stride
        end = min(start + window_size, test_arr.shape[0])
        window_is_true_anomaly.append(y_true_point[start:end].max())

    df = pd.DataFrame({"score": scores, "is_true_anomaly": window_is_true_anomaly})
    print(f"\n{ch_id} - score stats by true label:")
    print(df.groupby("is_true_anomaly")["score"].describe())
    print(f"\nRank of true-anomaly windows (0=most anomalous):")
    ranked = df["score"].rank()
    print(ranked[df["is_true_anomaly"] == 1].describe())
    return df
