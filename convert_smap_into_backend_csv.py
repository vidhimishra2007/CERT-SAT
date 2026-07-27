"""
convert_smap_into_backend_csv.py

Converts real NASA SMAP/MSL channels into the backend_telemetry_history.csv
format expected by train_backend_telemetry_model.py.

Uses the same path resolution and .npy loading as the rest of the
pipeline (src.utils.config.get_paths, src.data.loader.load_channel_arrays,
src.data.loader.download_dataset) instead of hand-rolling directory joins.

Dataset resolution: if a local "archive" folder (same layout as
kagglehub.dataset_download() produces) exists, use it. Otherwise fall
back to downloading it via kagglehub -- same pattern as
examples/retrain_and_save_with_scalers.py's resolve_dataset_paths().

Usage:
    python3 convert_smap_into_backend_csv.py
"""

from pathlib import Path

import pandas as pd
import os

from src.data.loader import download_dataset, load_channel_arrays
from src.utils.config import get_paths

# Local extracted SMAP/MSL dataset root -- get_paths() expects the folder
# that CONTAINS "data/data/train" and "data/data/test" (same layout
# kagglehub.dataset_download() produces), not the train dir directly.
LOCAL_SMAP_ROOT = Path()
# Path("archive")

# Map your 4 backend features to 4 real SMAP/MSL channels
CHANNEL_MAP = {
    "battery_voltage": "P-1",
    "temperature": "S-1",
    "cpu_usage": "E-1",
    "signal_strength": "E-2",
}

SATELLITE_ID = "SAT-001"
OUTPUT_PATH = Path("data/backend_telemetry_history.csv")


def resolve_dataset_paths():
    if LOCAL_SMAP_ROOT is not None and LOCAL_SMAP_ROOT.is_dir():
        candidate = get_paths(str(LOCAL_SMAP_ROOT))
        if os.path.isdir(candidate["train_dir"]):
            print(f"Using local dataset at {LOCAL_SMAP_ROOT}/")
            return candidate
    print("No valid local dataset found -- downloading via kagglehub...")
    return download_dataset()


def main():
    paths = resolve_dataset_paths()

    data = {}
    min_len = None

    for feature, ch in CHANNEL_MAP.items():
        train_arr, _test_arr = load_channel_arrays(ch, paths["train_dir"], paths["test_dir"])
        if train_arr is None:
            raise FileNotFoundError(
                f"Channel '{ch}' (for feature '{feature}') not found under "
                f"{paths['train_dir']} / {paths['test_dir']}"
            )

        arr = train_arr[:, 0]  # column 0 = raw telemetry value
        data[feature] = arr
        min_len = len(arr) if min_len is None else min(min_len, len(arr))
        print(f"Loaded {ch}.npy -> {feature} ({len(arr)} rows)")

    # trim all channels to the same length
    for k in data:
        data[k] = data[k][:min_len]

    df = pd.DataFrame(data)
    df.insert(0, "satellite_id", SATELLITE_ID)
    df.insert(0, "timestamp", pd.date_range("2026-07-01", periods=min_len, freq="min"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()