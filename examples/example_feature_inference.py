"""Smoke test for the current CERT-SAT model artifact.

This example uses feature-level inference because the included artifact does
not contain fitted scalers for raw telemetry preprocessing.
"""

import os
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.loader import download_dataset
from src.inference.live_window import preprocess_live_window
from src.models.isolation_forest import MultiChannelIsolationForest
from src.utils.config import WINDOW_SIZE
from src.data.windowing import create_windows_stats


MODEL_PATH = REPO_ROOT / "models_saved/isoforest_all_channels.joblib"
CHANNEL_ID = "P-1"


# MODEL_PATH = "models_saved/isoforest_all_channels_with_scalers.joblib"
# CHANNEL_ID = "T-1"

# Local extracted SMAP/MSL dataset root (same layout kagglehub produces).
# Leave as-is even if you don't have this folder -- resolve_test_dir()
# below falls back to downloading via kagglehub automatically.
LOCAL_SMAP_ROOT = Path("archive")

# Which 120-timestep window to pull from the test file. Change START_IDX
# to test a different slice (e.g. a window inside a known anomaly sequence).
START_IDX = 0


def resolve_test_dir():
    """Find the SMAP/MSL test dir locally, or download it via kagglehub."""
    local_test_dir = os.path.join(LOCAL_SMAP_ROOT, "data", "data", "test")
    if os.path.isdir(local_test_dir):
        print(f"Using local test data at {local_test_dir}/")
        return local_test_dir
    print(f"No local '{LOCAL_SMAP_ROOT}/' folder found -- downloading via kagglehub...")
    paths = download_dataset()
    return paths["test_dir"]


def load_new_window(ch_id, test_dir, start_idx, window_size):
    """Pull one real (unseen) raw window from a channel's test .npy file."""
    test_arr = np.load(os.path.join(test_dir, f"{ch_id}.npy"))
    end_idx = start_idx + window_size
    if end_idx > len(test_arr):
        raise ValueError(
            f"START_IDX={start_idx} + WINDOW_SIZE={window_size} exceeds "
            f"test array length ({len(test_arr)}). Pick a smaller START_IDX."
        )
    return test_arr[start_idx:end_idx]

def main():
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    test_dir = resolve_test_dir()
    # detector = manager.detectors[CHANNEL_ID].model
    # n_features = detector.n_features_in_

    raw_window = load_new_window(CHANNEL_ID, test_dir, START_IDX, WINDOW_SIZE)
    # # Replace this with real scaled/windowed summary features from backend.
    # X_window_features = np.zeros((1, n_features), dtype=float)
    X_window_features = create_windows_stats(raw_window, WINDOW_SIZE, WINDOW_SIZE)
    result = manager.predict(CHANNEL_ID, X_window_features)

    serializable = {
        "ch_id": result["ch_id"],
        "is_anomaly": result["is_anomaly"].astype(bool).tolist(),
        "label": result["label"].astype(int).tolist(),
        "score": result["score"].astype(float).tolist(),
        "n_samples": result["n_samples"],
    }
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
