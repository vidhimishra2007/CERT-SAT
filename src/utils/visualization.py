"""
Visualization utilities for model OUTPUTS (anomaly scores, reconstruction
errors, predicted vs. true anomaly windows).

This is distinct from src/evaluation/eda_utils.py's plot_channel(), which
plots raw telemetry before any model is involved. Use this module once
you have scores/predictions from IsolationForest, LSTM, or Autoencoder.

Works for all three methods since it just takes a 1D score/error array
plus true/predicted point-labels -- IF's decision_function, LSTM's
prediction error, and AE's reconstruction error all fit this shape.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_scores_with_anomalies(scores, y_true_point, ch_id, threshold=None,
                                lower_is_anomalous=True, save_dir=None):
    """
    Plot a continuous anomaly score/error series over time, with true
    anomaly windows shaded and an optional threshold line.

    Parameters
    ----------
    scores : 1D array, one value per test timestep (or per window --
        see plot_window_scores below if scores are window-level, not point-level)
    y_true_point : 1D binary array, same length as scores, ground truth
    ch_id : str, channel id for the plot title
    threshold : float, optional decision threshold to draw as a horizontal line
    lower_is_anomalous : bool. IsolationForest.decision_function: lower = more
        anomalous. Reconstruction/prediction error (LSTM, AE): higher = more
        anomalous. Set accordingly so shading direction reads correctly.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(scores, color="steelblue", linewidth=0.8, label="anomaly score")

    # shade true anomaly regions
    in_anomaly = False
    start = 0
    for i, v in enumerate(y_true_point):
        if v == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif v == 0 and in_anomaly:
            in_anomaly = False
            ax.axvspan(start, i, color="red", alpha=0.25)
    if in_anomaly:
        ax.axvspan(start, len(y_true_point), color="red", alpha=0.25)

    if threshold is not None:
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")

    direction = "lower = more anomalous" if lower_is_anomalous else "higher = more anomalous"
    ax.set_title(f"{ch_id} - anomaly score over time ({direction}); true anomalies shaded red")
    ax.set_xlabel("timestep")
    ax.set_ylabel("score")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{ch_id}_score_plot.png"), dpi=150)
    plt.show()


def plot_prediction_overlay(raw_series, y_true_point, y_pred_point, ch_id,
                             feature_idx=0, save_dir=None):
    """
    Plot raw telemetry with TWO shaded overlays: true anomalies (red) and
    predicted anomalies (blue, semi-transparent), so you can visually see
    false positives/negatives and near-misses.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw_series[:, feature_idx] if raw_series.ndim > 1 else raw_series,
             color="black", linewidth=0.7)

    def shade_regions(mask, color, alpha, label):
        in_region = False
        start = 0
        first = True
        for i, v in enumerate(mask):
            if v == 1 and not in_region:
                in_region = True
                start = i
            elif v == 0 and in_region:
                in_region = False
                ax.axvspan(start, i, color=color, alpha=alpha, label=label if first else None)
                first = False
        if in_region:
            ax.axvspan(start, len(mask), color=color, alpha=alpha, label=label if first else None)

    shade_regions(y_true_point, "red", 0.25, "true anomaly")
    shade_regions(y_pred_point, "blue", 0.20, "predicted anomaly")

    ax.set_title(f"{ch_id} - true vs predicted anomalies (feature {feature_idx})")
    ax.set_xlabel("timestep")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{ch_id}_prediction_overlay.png"), dpi=150)
    plt.show()


def plot_score_distribution_by_label(scores, y_true, ch_id, save_dir=None):
    """
    Histogram comparing score distributions for normal vs. true-anomaly
    points/windows. Good separation (little overlap) = model is finding
    useful signal for this channel; heavy overlap = model is struggling.
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[y_true == 0], bins=30, alpha=0.6, label="normal", color="steelblue")
    ax.hist(scores[y_true == 1], bins=30, alpha=0.6, label="true anomaly", color="crimson")
    ax.set_title(f"{ch_id} - score distribution by true label")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{ch_id}_score_distribution.png"), dpi=150)
    plt.show()


def plot_method_comparison_bar(per_channel_dfs, metric="f1", labels=None, save_dir=None):
    """
    Grouped bar chart comparing mean metric (e.g. F1 or F0.5) across
    methods -- pass a dict {method_name: per_channel_df} once you have
    results from IF, LSTM, and AE.

    Example
    -------
    plot_method_comparison_bar({
        "Isolation Forest": if_per_channel_df,
        "LSTM": lstm_per_channel_df,
        "Autoencoder": ae_per_channel_df,
    }, metric="f1")
    """
    method_names = list(per_channel_dfs.keys())
    means = [df[metric].mean() for df in per_channel_dfs.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(method_names, means, color=["steelblue", "seagreen", "darkorange"][:len(method_names)])
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(f"Method comparison - mean {metric} across channels")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"method_comparison_{metric}.png"), dpi=150)
    plt.show()
