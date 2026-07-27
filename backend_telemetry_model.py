"""
src/backend_telemetry_model.py

Demo backend telemetry model for the synthetic 4-feature satellite CSV
(data/backend_telemetry_history.csv).

This used to duplicate windowing/scaling/IsolationForest logic that
already lives elsewhere in src/. It now just wires those pieces together:

- src.data.preprocessing.scale_channel_with_scaler -> fit/apply StandardScaler
- src.data.windowing.create_windows_stats           -> mean/std/min/max/delta windows
- src.models.isolation_forest.MultiChannelIsolationForest -> fit/predict/save/load
- src.inference.live_window.preprocess_live_window  -> raw buffer -> window features

The demo CSV only has one logical model (no real per-channel split like
SMAP/MSL), so every satellite_id is scored against a single internal
"channel" trained in MultiChannelIsolationForest. That keeps the saved
artifact in the same format as the production multi-channel model, and
lets live inference reuse preprocess_live_window() unchanged instead of
re-implementing the scaler-then-window step here.

Public interface (train_global_model, BackendTelemetryScorer,
FEATURE_NAMES, WINDOW_SIZE, STRIDE) is unchanged, so
train_backend_telemetry_model.py and examples/example_live_inference.py
keep working as-is. NOTE: the saved artifact's internal format changed
(it's now a MultiChannelIsolationForest payload, not a raw
{"model", "scaler", ...} dict) -- retrain and re-save before running
example_live_inference.py against a model trained with the old code.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.preprocessing import scale_channel_with_scaler
from src.data.windowing import create_windows_stats
from src.inference.live_window import preprocess_live_window
from src.models.isolation_forest import MultiChannelIsolationForest
from src.utils.config import WINDOW_SIZE, STRIDE


FEATURE_NAMES = [
    "battery_voltage",
    "temperature",
    "cpu_usage",
    "signal_strength",
]


# Internal MultiChannelIsolationForest key this demo trains under. Every
# satellite_id shares this one model -- there's no per-satellite training
# split in the demo CSV the way there is per-channel for SMAP/MSL.
_GLOBAL_CHANNEL_ID = "backend_global"


def train_global_model(
    csv_path: str | Path,
    output_path: str | Path,
    contamination: float = 0.05,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> dict:
    feature_names = list(feature_names)
    df = pd.read_csv(csv_path)

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {missing}")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    X_raw = df[feature_names].to_numpy(dtype=float)
    if len(X_raw) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} rows, got {len(X_raw)}")

    # No held-out split for this demo CSV -- fit and window on the same
    # array. scale_channel_with_scaler still hands back the fitted scaler
    # so it gets persisted alongside the model, same as the production path.
    X_scaled, _, scaler = scale_channel_with_scaler(X_raw, X_raw)
    X_train = create_windows_stats(X_scaled, WINDOW_SIZE, STRIDE)

    manager = MultiChannelIsolationForest()
    manager.fit_channel(_GLOBAL_CHANNEL_ID, X_train, contamination=contamination)
    manager.scalers[_GLOBAL_CHANNEL_ID] = scaler

    output_path = Path(output_path)
    manager.save(output_path)

    return {
        "manager": manager,
        "feature_names": feature_names,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "raw_feature_count": len(feature_names),
        "model_feature_count": X_train.shape[1],
        "contamination": contamination,
    }


class BackendTelemetryScorer:
    """
    Streaming scorer -- owns one rolling buffer per satellite_id and scores
    it once WINDOW_SIZE raw packets have arrived, reusing
    preprocess_live_window() + MultiChannelIsolationForest.predict() for
    the actual scaling/windowing/inference (see
    examples/CERT-SAT_BACKEND_CONTRACT.md for the same pattern applied to
    real SMAP/MSL channels).
    """

    def __init__(self, model_path: str | Path):
        self.manager = MultiChannelIsolationForest.load(model_path)
        self.feature_names = FEATURE_NAMES
        self.window_size = WINDOW_SIZE
        self.buffers = defaultdict(lambda: deque(maxlen=self.window_size))

    def handle_packet(self, satellite_id: str, telemetry: dict) -> dict:
        values = [telemetry[name] for name in self.feature_names]
        self.buffers[satellite_id].append(values)

        if len(self.buffers[satellite_id]) < self.window_size:
            return {
                "satellite_id": satellite_id,
                "status": "warming_up",
                "buffer_size": len(self.buffers[satellite_id]),
                "required_buffer_size": self.window_size,
                "prediction": None,
            }

        raw_buffer = np.asarray(self.buffers[satellite_id], dtype=float)
        X = preprocess_live_window(
            _GLOBAL_CHANNEL_ID, raw_buffer, self.manager, window_size=self.window_size
        )
        result = self.manager.predict(_GLOBAL_CHANNEL_ID, X)

        return {
            "satellite_id": satellite_id,
            "status": "scored",
            "is_anomaly": bool(result["is_anomaly"][0]),
            "label": int(result["label"][0]),
            "score": float(result["score"][0]),
            "n_samples": int(result["n_samples"]),
        }