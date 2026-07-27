"""
src/models/isolation_forest.py

Isolation Forest baseline detector for CERT-SAT.

Trains one IsolationForest per channel (SMAP/MSL channels have very
different behavior, so a single global model would underperform), using
summary-statistic windows as input features.

Refactored for backend integration:
- IsolationForestDetector: single-channel model, conforms to BaseDetector
  (fit/predict/score) so evaluation/compare_models.py can treat it like the
  upcoming LSTM/Autoencoder detectors.
- MultiChannelIsolationForest: owns one IsolationForestDetector per channel,
  exposes a single predict(ch_id, X) call, and saves/loads every channel's
  model as one artifact. This is the class the backend should use.
- run_isolation_forest(): unchanged public signature/behavior (same 2-tuple
  return: results, skipped_channels) so existing evaluation code keeps
  working. Internally it now builds channel models via
  IsolationForestDetector/MultiChannelIsolationForest instead of raw
  sklearn calls. Pass a `manager` in if you want the fitted per-channel
  models back afterward to save() for deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base_detector import BaseDetector
from src.models.versioning import build_artifact_metadata, check_interface_version
from src.data.preprocessing import scale_channel_with_scaler
from src.data.windowing import create_windows_stats
from src.data.loader import load_channel_arrays
from src.utils.config import (
    IF_N_ESTIMATORS, IF_RANDOM_STATE,WINDOW_SIZE, STRIDE,
    IF_DEFAULT_CONTAMINATION, IF_CONTAMINATION_MIN, IF_CONTAMINATION_MAX,
)


# --------------------------------------------------------------------- #
# Single-channel detector — conforms to BaseDetector
# --------------------------------------------------------------------- #
class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest for a single SMAP/MSL channel.

    Score/label convention (per BaseDetector's note that this differs by
    implementation): sklearn-native.
        predict() -> 1 = normal, -1 = anomaly
        score()   -> decision_function output; LOWER = more anomalous
    This is the OPPOSITE convention to reconstruction-error models
    (LSTM/Autoencoder), where HIGHER error = more anomalous.
    compare_models.py will need to account for this when combining or
    ranking scores across detector types.
    """

    def __init__(self, ch_id: str, n_estimators: int = IF_N_ESTIMATORS,
                 contamination: float = IF_DEFAULT_CONTAMINATION,
                 random_state: int = IF_RANDOM_STATE):
        self.ch_id = ch_id
        self.n_estimators = n_estimators
        self.contamination = min(max(contamination, IF_CONTAMINATION_MIN), IF_CONTAMINATION_MAX)
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
        

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
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
# Multi-channel manager — this is the integration surface for the backend
# --------------------------------------------------------------------- #
class MultiChannelIsolationForest:
    """
    Owns one IsolationForestDetector per channel and exposes a single
    predict(ch_id, X) call, so the backend never has to manage a
    dict of models itself.

    IMPORTANT — input to predict() must already be windowed + scaled the
    same way training data was (scale_channel + create_windows_stats).
    This class does not do that preprocessing itself; see the
    "Known gap" note in README_isolation_forest.md about persisting the
    scaler for live/new data.
    """

    def __init__(self):
        self.detectors: dict[str, IsolationForestDetector] = {}
        self.scalers: dict = {}
        

    def fit_channel(self, ch_id: str, X_train: np.ndarray,
                     contamination: float = IF_DEFAULT_CONTAMINATION) -> IsolationForestDetector:
        det = IsolationForestDetector(ch_id, contamination=contamination)
        det.fit(X_train)
        self.detectors[ch_id] = det
        return det

    def predict(self, ch_id: str, X_test: np.ndarray) -> dict:
        """
        Integration entrypoint.

        Parameters
        ----------
        ch_id : channel id, e.g. "A-1" — must have been fit already.
        X_test : ndarray, shape (n_windows, n_features)
            Windowed summary-stat features, scaled the same way as training
            (see scale_channel / create_windows_stats).

        Returns
        -------
        dict:
            "ch_id"       : str
            "is_anomaly"  : bool[], shape (n_windows,) -- True = anomaly
            "label"       : int[], shape (n_windows,) -- sklearn-native, 1/-1
            "score"       : float[], shape (n_windows,) -- LOWER = more anomalous
            "n_samples"   : int
        """
        if ch_id not in self.detectors:
            raise KeyError(
                f"No trained model for channel '{ch_id}'. "
                f"Trained channels: {list(self.detectors)}"
            )
        det = self.detectors[ch_id]
        label = det.predict(X_test)
        return {
            "ch_id": ch_id,
            "is_anomaly": label == -1,
            "label": label,
            "score": det.score(X_test),
            "n_samples": int(np.asarray(X_test).shape[0]),
        }

    def predict_raw_window(self, ch_id: str, raw_buffer: np.ndarray) -> dict:
        """
        Backend-facing entrypoint: score ONE raw (unscaled) rolling window
        directly, without the caller having to scale/window it first. This
        is the method examples/CERT-SAT_BACKEND_CONTRACT.md documents as
        `manager.predict_raw_window(ch_id, raw_buffer)`.

        Parameters
        ----------
        ch_id : channel id, e.g. "P-1" -- must have been fit already.
        raw_buffer : ndarray, shape (window_size, n_raw_features)
            The most recent WINDOW_SIZE raw telemetry timesteps for this
            channel, chronological order, unscaled.

        Returns
        -------
        Same dict shape as predict(): ch_id, is_anomaly, label, score, n_samples.
        """
        # Local import avoids a circular import (live_window -> this module).
        from src.inference.live_window import preprocess_live_window

        X_window_features = preprocess_live_window(ch_id, raw_buffer, self)
        return self.predict(ch_id, X_window_features)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save every channel's fitted detector into one .joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "detectors": self.detectors,
            "scalers": self.scalers,
            "meta": build_artifact_metadata(
                model_type="isolation_forest",
                channel_ids=list(self.detectors.keys()),
            ),
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultiChannelIsolationForest":
        payload = joblib.load(Path(path))
        obj = cls()
        obj.detectors = payload["detectors"]
        obj.scalers = payload.get("scalers", {})
        return obj


# --------------------------------------------------------------------- #
# Original batch-evaluation entrypoint — signature/behavior unchanged
# --------------------------------------------------------------------- #
def run_isolation_forest(labels, shape_df, train_dir, test_dir,
                          window_size, stride, contam_lookup=None,
                          manager: Optional[MultiChannelIsolationForest] = None):
    """
    Fit + predict an IsolationForest per channel.

    Parameters
    ----------
    labels : DataFrame from load_labels()
    shape_df : DataFrame from build_shape_summary(), used to skip channels
        too short for the chosen window_size
    contam_lookup : dict {ch_id: anomaly_fraction}, e.g. from
        build_imbalance_summary(). If None, IF_DEFAULT_CONTAMINATION is used
        for every channel (leakage-free baseline). If provided, contamination
        is set per-channel from true test anomaly fraction -- useful as an
        "oracle upper bound" comparison, but note this leaks test labels
        into a hyperparameter; report this choice explicitly.
    manager : MultiChannelIsolationForest, optional
        If provided, every channel's fitted detector is also registered
        into this manager, so you can call manager.save(path) afterward to
        get a deployable artifact. If omitted, a manager is created and
        discarded internally -- behavior is identical to before.

    Returns
    -------
    results : list of dicts, one per successfully modeled channel, with
        keys: ch_id, spacecraft, class, preds, scores, anomaly_sequences, test_len.
        Matches the format expected by src.evaluation.metrics.aggregate_results.
        (unchanged: preds/scores are the same sklearn-native arrays as before)
    skipped_channels : list of ch_ids skipped for being too short.
    """
    contam_lookup = contam_lookup or {}
    manager = manager if manager is not None else MultiChannelIsolationForest()
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

        train_scaled, test_scaled, scaler = scale_channel_with_scaler(train_arr, test_arr)
        X_train = create_windows_stats(train_scaled, window_size, stride)
        X_test = create_windows_stats(test_scaled, window_size, stride)

        if len(X_train) == 0 or len(X_test) == 0:
            skipped_channels.append(ch_id)
            continue

        contam_rate = contam_lookup.get(ch_id, IF_DEFAULT_CONTAMINATION)

        det = manager.fit_channel(ch_id, X_train, contamination=contam_rate)
        manager.scalers[ch_id] = scaler
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