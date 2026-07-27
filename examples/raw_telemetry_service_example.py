"""Backend-style raw telemetry inference wrapper for CERT-SAT."""

from collections import defaultdict, deque
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.inference.live_window import preprocess_live_window
from src.models.isolation_forest import MultiChannelIsolationForest
from src.utils.config import WINDOW_SIZE


class CertSatTelemetryScorer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = REPO_ROOT / "models_saved/isoforest_all_channels_with_scalers.joblib"
        self.manager = MultiChannelIsolationForest.load(model_path)
        self.buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def handle_packet(self, ch_id, telemetry_vector):
        if ch_id not in self.manager.detectors:
            raise KeyError(f"Unsupported channel: {ch_id}")

        self.buffers[ch_id].append(telemetry_vector)

        if len(self.buffers[ch_id]) < WINDOW_SIZE:
            return {
                "ch_id": ch_id,
                "status": "warming_up",
                "buffer_size": len(self.buffers[ch_id]),
                "required_buffer_size": WINDOW_SIZE,
                "prediction": None,
            }

        raw_buffer = np.asarray(self.buffers[ch_id], dtype=float)
        X_window_features = preprocess_live_window(ch_id, raw_buffer, self.manager)
        result = self.manager.predict(ch_id, X_window_features)

        return {
            "ch_id": result["ch_id"],
            "status": "scored",
            "is_anomaly": bool(result["is_anomaly"][0]),
            "label": int(result["label"][0]),
            "score": float(result["score"][0]),
            "n_samples": int(result["n_samples"]),
        }
