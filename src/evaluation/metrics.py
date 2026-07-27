"""
Evaluation utilities shared across all three detection methods (IF, LSTM, AE).

Includes:
- point_adjust: the standard time-series anomaly detection adjustment where,
  if a model detects ANY point within a true anomaly segment, the whole
  segment counts as detected (common practice in Telemanom / SMAP-MSL papers,
  but be aware it inflates recall vs. strict point-wise scoring)
- compute_channel_metrics: precision/recall/F1 (swap to fbeta for F0.5) for
  a single channel
- aggregate_results: run metrics across all channels and produce both
  per-channel and micro-aggregated summaries
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data.windowing import sequences_to_point_labels, windows_to_point_labels


def point_adjust(y_true, y_pred):
    """
    If a predicted anomaly point falls anywhere inside a true anomaly segment,
    mark the entire segment as correctly predicted. Standard practice for
    SMAP/MSL-style benchmarks (Hundman et al.), but inflates recall relative
    to strict point-wise evaluation -- report this choice explicitly in your
    writeup.
    """
    y_pred_adjusted = y_pred.copy()
    anomaly_state = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            j = i
            while j >= 0 and y_true[j] == 1:
                y_pred_adjusted[j] = 1
                j -= 1
        elif y_true[i] == 0:
            anomaly_state = False
        if anomaly_state:
            y_pred_adjusted[i] = 1
    return y_pred_adjusted


def compute_channel_metrics(y_true, y_pred, beta=1.0):
    """
    Precision/recall/F-beta for one channel's point-level predictions.
    Set beta=0.5 to match the project's primary F0.5 metric (precision-weighted).
    """
    precision, recall, fbeta, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, beta=beta
    )
    return precision, recall, fbeta


def aggregate_results(results, window_size, stride, beta=1.0):
    """
    Given a list of per-channel result dicts (each with preds/scores/
    anomaly_sequences/test_len/ch_id/spacecraft/class), compute:
    - per_channel_df: precision/recall/F-beta per channel
    - overall (micro-aggregated precision/recall/F-beta across all points, all channels)

    `results` format matches what src.models.isolation_forest.run_isolation_forest
    returns; for LSTM/AE you'll adapt the preds->point-label conversion but
    can reuse this aggregation logic.
    """
    per_channel_metrics = []
    all_true, all_pred = [], []

    for r in results:
        y_true = sequences_to_point_labels(r["anomaly_sequences"], r["test_len"])
        y_pred = windows_to_point_labels(r["preds"], r["test_len"], window_size, stride)
        y_pred_adj = point_adjust(y_true, y_pred)

        precision, recall, fbeta = compute_channel_metrics(y_true, y_pred_adj, beta=beta)
        per_channel_metrics.append({
            "ch_id": r["ch_id"],
            "spacecraft": r["spacecraft"],
            "class": r["class"],
            "precision": precision,
            "recall": recall,
            f"f{beta}".replace(".0", ""): fbeta,
        })

        all_true.extend(y_true)
        all_pred.extend(y_pred_adj)

    per_channel_df = pd.DataFrame(per_channel_metrics)
    overall_p, overall_r, overall_fbeta, _ = precision_recall_fscore_support(
        all_true, all_pred, average="binary", zero_division=0, beta=beta
    )
    overall = {"precision": overall_p, "recall": overall_r, f"f{beta}".replace(".0", ""): overall_fbeta}

    return per_channel_df, overall