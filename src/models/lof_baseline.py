"""
src/models/lof_baseline.py

Local Outlier Factor (LOF) baseline detector for CERT-SAT.

Added as a classical-baseline comparison point for the Isolation Forest
model (see BENCHMARK_COMPARISON.md). Mirrors the structure of
src/models/isolation_forest.py so the two can be run through the exact
same data loading / windowing / scaling / metrics pipeline, making the
comparison apples-to-apples:

- Same per-channel StandardScaler fit on train only (scale_channel_with_scaler)
- Same summary-statistic windowing (create_windows_stats, window_size=120, stride=10)
- Same point-adjust + precision/recall/F1 aggregation (src/evaluation/metrics.py)

LOF requires `novelty=True` to support fit-on-train / predict-on-test,
since the default LOF mode only scores the data it was fit on.

- LOFDetector: single-channel model, conforms to BaseDetector (fit/predict/score).
- run_lof(): same signature/return shape as run_isolation_forest() so it can
  be dropped into the same benchmark script and aggregate_results() call.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.models.base_detector import BaseDetector
from src.data.preprocessing import scale_channel_with_scaler
from src.data.windowing import create_windows_stats
from src.data.loader import load_channel_arrays
from src.utils.config import IF_DEFAULT_CONTAMINATION, IF_CONTAMINATION_MIN, IF_CONTAMINATION_MAX


# --------------------------------------------------------------------- #
# Single-channel detector — conforms to BaseDetector
# --------------------------------------------------------------------- #
class LOFDetector(BaseDetector):
    """
    Local Outlier Factor for a single SMAP/MSL channel.

    Uses novelty=True so the model can be fit on training (nominal-only)
    windows and then score unseen test windows, matching how the
    Isolation Forest baseline is used.

    Score/label convention (sklearn-native, same direction as
    IsolationForestDetector):
        predict() -> 1 = normal, -1 = anomaly
        score()   -> decision_function output; LOWER = more anomalous
    """

    def __init__(self, ch_id: str, n_neighbors: int = 20,
                 contamination: float = IF_DEFAULT_CONTAMINATION):
        self.ch_id = ch_id
        self.n_neighbors = n_neighbors
        self.contamination = min(max(contamination, IF_CONTAMINATION_MIN), IF_CONTAMINATION_MAX)
        self.model: Optional[LocalOutlierFactor] = None
        self._is_fitted = False

    def fit(self, X_train: np.ndarray) -> "LOFDetector":
        # n_neighbors can't exceed the number of training samples.
        n_neighbors = min(self.n_neighbors, max(1, len(X_train) - 1))
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )
        self.model.fit(X_train)
        self._is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Sklearn-native predictions: 1 = normal, -1 = anomaly."""
        self._check_fitted()
        return self.model.predict(X_test)

    def score(self, X_test: np.ndarray) -> np.ndarray:
        """decision_function output. LOWER = more anomalous."""
        self._check_fitted()
        return self.model.decision_function(X_test)

    def is_anomaly(self, X_test: np.ndarray) -> np.ndarray:
        """predict() normalized to a plain bool array, True = anomaly."""
        return self.predict(X_test) == -1

    def _check_fitted(self):
        if not self._is_fitted or self.model is None:
            raise RuntimeError(f"Detector for channel '{self.ch_id}' is not fitted yet.")


# --------------------------------------------------------------------- #
# Batch-evaluation entrypoint — same shape as run_isolation_forest()
# --------------------------------------------------------------------- #
def run_lof(labels, shape_df, train_dir, test_dir, window_size, stride,
            contam_lookup=None, n_neighbors: int = 20):
    """
    Fit + predict a Local Outlier Factor per channel.

    Parameters mirror src.models.isolation_forest.run_isolation_forest
    exactly (labels, shape_df, train_dir, test_dir, window_size, stride,
    contam_lookup), so both models can be driven from the same benchmark
    loop.

    Returns
    -------
    results : list of dicts, one per successfully modeled channel, with
        keys: ch_id, spacecraft, class, preds, scores, anomaly_sequences,
        test_len. Same format as run_isolation_forest -- feeds directly
        into src.evaluation.metrics.aggregate_results.
    skipped_channels : list of ch_ids skipped for being too short.
    """
    contam_lookup = contam_lookup or {}
    results = []
    skipped_channels = []

    for _, row in labels.iterrows():
        ch_id = row["chan_id"]

        match = shape_df[shape_df["ch_id"] == ch_id]
        if match.empty or match["train_len"].values[0] < window_size or match["test_len"].values[0] < window_size:
            skipped_channels.append(ch_id)
            continue

        train_arr, test_arr = load_channel_arrays(ch_id, train_dir, test_dir)
        if train_arr is None:
            skipped_channels.append(ch_id)
            continue

        train_scaled, test_scaled, _scaler = scale_channel_with_scaler(train_arr, test_arr)
        X_train = create_windows_stats(train_scaled, window_size, stride)
        X_test = create_windows_stats(test_scaled, window_size, stride)

        if len(X_train) == 0 or len(X_test) == 0:
            skipped_channels.append(ch_id)
            continue

        contam_rate = contam_lookup.get(ch_id, IF_DEFAULT_CONTAMINATION)

        det = LOFDetector(ch_id, n_neighbors=n_neighbors, contamination=contam_rate)
        det.fit(X_train)
        preds = det.predict(X_test)
        scores = det.score(X_test)

        results.append({
            "ch_id": ch_id,
            "spacecraft": row["spacecraft"],
            "class": row["class"],
            "preds": preds,
            "scores": scores,
            "anomaly_sequences": row["anomaly_sequences"],
            "test_len": test_arr.shape[0],
        })

    return results, skipped_channels