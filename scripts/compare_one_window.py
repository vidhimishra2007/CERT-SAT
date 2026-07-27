"""
scripts/compare_one_window.py

Quick single-window sanity check: does the live API path
(preprocess_live_window + manager.predict) produce the same label/score
as the offline notebook/batch path (create_windows_stats) for one
channel's first window?

This is a minimal special case of validate_channel() in
validate_backend_integration.py (max_windows=1) -- reuses that function
instead of re-implementing the same batch-vs-live comparison here. For
multi-channel / multi-window validation with latency numbers, use
validate_backend_integration.py directly.

Usage:
    python scripts/compare_one_window.py
    python scripts/compare_one_window.py P-1
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.isolation_forest import MultiChannelIsolationForest
from scripts.validate_backend_integration import validate_channel, MODEL_PATH


def resolve_channels(args, manager):
    if not args:
        return ["P-1"]
    if len(args) == 1 and args[0] == "all":
        return sorted(manager.detectors)
    return args


def main():
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    channels = resolve_channels(sys.argv[1:], manager)
 

    for ch_id in channels:
        if ch_id not in manager.detectors:
            print(f"{ch_id}: not in model. Trained channels include: {sorted(manager.detectors)[:10]}...")
            continue

    if ch_id not in manager.detectors:
        print(f"{ch_id}: not in model. Trained channels include: {sorted(manager.detectors)[:10]}...")
        return

    result = validate_channel(ch_id, manager, max_windows=1)
    if result is None:
        print(f"{ch_id}: not enough test data to build one window.")
        return

    print(f"{ch_id}: window comparison:")
    print("Notebook:", result["batch_preds"], result["batch_scores"])
    print("API:     ", result["api_preds"], result["api_scores"])

    status = "MATCH" if result["mismatches"] == 0 else "MISMATCH"
    print(f"[{status}] max score diff = {result['max_score_diff']:.6f}")
   


if __name__ == "__main__":
    main()