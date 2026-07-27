"""
scripts/validate_backend_integration.py

Run this anytime after backend integration to confirm the live API path
(preprocess_live_window + manager.predict) produces the SAME predictions
as the offline notebook path (scale_channel_with_scaler + create_windows_stats).

Usage:
    python scripts/validate_backend_integration.py
    python scripts/validate_backend_integration.py --channels P-1 S-1 E-1
"""

import argparse
import time
import sys
import os
import numpy as np

# Make sure the repo root (parent of scripts/) is importable, regardless
# of where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.isolation_forest import MultiChannelIsolationForest
from src.inference.live_window import preprocess_live_window
from src.data.windowing import create_windows_stats
from src.utils.config import WINDOW_SIZE, STRIDE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models_saved/isoforest_all_channels_with_scalers.joblib")
TEST_DIR = os.path.join(REPO_ROOT, "archive/data/data/test")

DEFAULT_CHANNELS = ["P-1", "S-1", "E-1", "E-2", "A-1", "D-1",
                     "T-1", "M-1", "F-1", "G-1", "R-1", "C-1"]


def validate_channel(ch_id, manager, max_windows=150):
    test_arr = np.load(f"{TEST_DIR}/{ch_id}.npy")
    if test_arr.shape[0] < WINDOW_SIZE:
        return None

    scaler = manager.scalers[ch_id]
    det = manager.detectors[ch_id]

    # --- notebook / batch path ---
    test_scaled = scaler.transform(test_arr)
    X_batch = create_windows_stats(test_scaled, WINDOW_SIZE, STRIDE)
    if len(X_batch) == 0:
        return None
    n = min(len(X_batch), max_windows)
    X_batch = X_batch[:n]
    batch_preds = det.predict(X_batch)
    batch_scores = det.score(X_batch)

    # --- API / live path, same window positions ---
    api_preds, api_scores = [], []
    for i in range(n):
        start = i * STRIDE
        raw_buffer = test_arr[start:start + WINDOW_SIZE]
        X_live = preprocess_live_window(ch_id, raw_buffer, manager)
        r = manager.predict(ch_id, X_live)
        api_preds.append(r["label"][0])
        api_scores.append(r["score"][0])

    mismatches = int(np.sum(batch_preds != np.array(api_preds)))
    max_diff = float(np.max(np.abs(batch_scores - np.array(api_scores))))
    return {
        "ch_id": ch_id,
        "n_windows": n,
        "mismatches": mismatches,
        "max_score_diff": max_diff,
        "batch_preds": batch_preds,
        "batch_scores": batch_scores,
        "api_preds": np.array(api_preds),
        "api_scores": np.array(api_scores),
    }


def measure_latency(ch_id, manager, n_calls=100):
    test_arr = np.load(f"{TEST_DIR}/{ch_id}.npy")
    windows = [test_arr[i:i + WINDOW_SIZE] for i in range(n_calls)]
    for w in windows[:10]:  # warm-up
        X = preprocess_live_window(ch_id, w, manager)
        manager.predict(ch_id, X)
    times = []
    for w in windows:
        t0 = time.perf_counter()
        X = preprocess_live_window(ch_id, w, manager)
        manager.predict(ch_id, X)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--max-windows", type=int, default=150)
    args = parser.parse_args()

    print(f"Loading {MODEL_PATH} ...")
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    print(f"Loaded {len(manager.detectors)} channels.\n")

    total_windows, total_mismatches = 0, 0
    for ch_id in args.channels:
        if ch_id not in manager.detectors:
            print(f"  {ch_id}: SKIPPED (not in model)")
            continue
        result = validate_channel(ch_id, manager, args.max_windows)
        if result is None:
            print(f"  {ch_id}: SKIPPED (not enough test data)")
            continue
        total_windows += result["n_windows"]
        total_mismatches += result["mismatches"]
        status = "OK" if result["mismatches"] == 0 else "MISMATCH FOUND"
        print(f"  {ch_id}: {result['n_windows']} windows, "
              f"{result['mismatches']} mismatches, "
              f"max score diff {result['max_score_diff']:.6f}  [{status}]")

    print(f"\nTOTAL: {total_windows} windows checked, {total_mismatches} mismatches")
    if total_mismatches == 0:
        print("PASS -- live API path matches notebook predictions exactly.")
    else:
        print("FAIL -- investigate preprocess_live_window / scaler for the flagged channel(s).")

    # latency = measure_latency(args.channels[0], manager)
    # print(f"\nSingle-packet API latency (channel {args.channels[0]}): {latency:.1f} ms avg")
    print()
    for ch_id in args.channels:
        if ch_id not in manager.detectors:
            print(f"Single-packet API latency (channel {ch_id}): SKIPPED (not in model)")
            continue
        latency = measure_latency(ch_id, manager)
        print(f"Single-packet API latency (channel {ch_id}): {latency:.1f} ms avg")

if __name__ == "__main__":
    main()