"""
src/inference/live_window.py

Live inference preprocessing for CERT-SAT.

Batch evaluation (src/models/isolation_forest.py, run_isolation_forest)
scales and windows an entire train/test array at once. Live inference is
different: the backend receives one raw telemetry packet at a time and
needs to score a single window using the SAME scaler that was fit during
training for that channel.

This module bridges that gap: given a rolling buffer of the last
WINDOW_SIZE raw timesteps for one channel, it produces the exact
summary-statistic feature vector (mean/std/min/max/delta) that
MultiChannelIsolationForest.predict(ch_id, X_test) expects.

Usage (backend side)
---------------------
    manager = MultiChannelIsolationForest.load("isoforest_all_channels.joblib")

    # maintain a rolling buffer per channel, e.g. a collections.deque
    # with maxlen=WINDOW_SIZE, popping oldest as new packets arrive
    raw_buffer = ...  # ndarray, shape (WINDOW_SIZE, n_features)

    X_test = preprocess_live_window("P-1", raw_buffer, manager)
    result = manager.predict("P-1", X_test)

    if result["is_anomaly"][0]:
        # raise an alert
        ...
"""

import numpy as np

from src.utils.config import WINDOW_SIZE


def preprocess_live_window(ch_id, raw_buffer, manager, window_size: int = WINDOW_SIZE):
    """
    Convert one raw rolling window of telemetry into the feature format
    expected by MultiChannelIsolationForest.predict().

    Parameters
    ----------
    ch_id : str
        Channel id, e.g. "P-1". Must already have a fitted scaler in
        manager.scalers (i.e. this channel was trained and manager.save()
        included scalers -- see the updated isolation_forest.py).
    raw_buffer : ndarray, shape (window_size, n_features)
        The most recent `window_size` RAW (unscaled) telemetry timesteps
        for this channel, in chronological order. The caller (backend) is
        responsible for maintaining this rolling buffer -- e.g. a
        collections.deque(maxlen=window_size) that pops the oldest packet
        as each new one arrives.
    manager : MultiChannelIsolationForest
        Loaded via MultiChannelIsolationForest.load(path).
    window_size : int
        Must match WINDOW_SIZE used during training (120 by default).

    Returns
    -------
    X_test : ndarray, shape (1, n_features * 5)
        Ready to pass directly into manager.predict(ch_id, X_test).

    Raises
    ------
    ValueError
        If raw_buffer does not have exactly `window_size` rows.
    KeyError
        If no scaler has been fitted/stored for this ch_id.
    """
    raw_buffer = np.asarray(raw_buffer)

    if raw_buffer.shape[0] != window_size:
        raise ValueError(
            f"Expected exactly {window_size} timesteps for channel '{ch_id}', "
            f"got {raw_buffer.shape[0]}. Maintain a rolling buffer of size "
            f"{window_size} before calling this function."
        )

    if ch_id not in manager.scalers:
        raise KeyError(
            f"No fitted scaler found for channel '{ch_id}'. This manager was "
            f"likely saved before scaler persistence was added -- retrain "
            f"and re-save using the updated isolation_forest.py."
        )

    scaler = manager.scalers[ch_id]
    # transform ONLY -- never re-fit on live data, or you leak live
    # statistics into what should be a fixed, training-time normalization
    scaled = scaler.transform(raw_buffer)

    feats = np.concatenate([
        scaled.mean(axis=0),
        scaled.std(axis=0),
        scaled.min(axis=0),
        scaled.max(axis=0),
        (scaled[-1] - scaled[0]),
    ])

    return feats.reshape(1, -1)