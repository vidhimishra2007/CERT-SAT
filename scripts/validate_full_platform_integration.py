"""
scripts/validate_full_platform_integration.py

End-to-end validation of the generic model interface against the full
platform (notebook path + backend/API path), run as one command.

This ties together the checks that previously lived as separate scripts
into a single pass/fail report suitable for CI or a pre-release gate:

  Stage 1 - Generic model interface contract (src/models/base_detector.py)
            Every registered detector class (IsolationForest, LOF, ...)
            fits/predicts/saves/loads through the same MultiChannelDetector
            code path. Delegates to scripts/test_generic_pipeline.py.

  Stage 2 - Notebook vs. live API prediction parity
            For every channel in the recommended production artifact,
            confirms that scoring a window through the offline/batch
            (notebook) path -- scale_channel_with_scaler + create_windows_stats
            -- produces IDENTICAL label + score to scoring the same window
            through the live/API path -- preprocess_live_window + predict().
            Delegates to scripts/validate_backend_integration.py.

  Stage 3 - Backend integration smoke test
            Feeds one channel's real telemetry through the streaming
            backend wrapper (examples/raw_telemetry_service_example.py,
            the pattern documented in
            examples/CERT-SAT_BACKEND_CONTRACT.md) packet-by-packet and
            checks the final scored response has the exact contract shape
            (ch_id/status/is_anomaly/label/score/n_samples) and matches the
            direct manager.predict() result for the same window.

Usage:
    python scripts/validate_full_platform_integration.py
    python scripts/validate_full_platform_integration.py --channels P-1 S-1 E-1
    python scripts/validate_full_platform_integration.py --skip-stage1

Exit code is 0 iff every stage passes.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.models.isolation_forest import MultiChannelIsolationForest
from scripts.validate_backend_integration import (
    MODEL_PATH,
    TEST_DIR,
    validate_channel,
)
from examples.raw_telemetry_service_example import CertSatTelemetryScorer

DEFAULT_CHANNELS = ["P-1", "S-1", "E-1", "A-1", "T-1", "C-1"]


def stage1_generic_interface():
    print("=" * 70)
    print("STAGE 1: generic model interface contract")
    print("=" * 70)
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "test_generic_pipeline.py")],
        cwd=REPO_ROOT,
    )
    ok = proc.returncode == 0
    print(f"\nSTAGE 1 {'PASSED' if ok else 'FAILED'}\n")
    return ok


def stage2_notebook_vs_api(channels, max_windows):
    print("=" * 70)
    print("STAGE 2: notebook path vs. live API path parity")
    print("=" * 70)
    manager = MultiChannelIsolationForest.load(MODEL_PATH)

    total_windows, total_mismatches = 0, 0
    for ch_id in channels:
        if ch_id not in manager.detectors:
            print(f"  {ch_id}: SKIPPED (not in model)")
            continue
        result = validate_channel(ch_id, manager, max_windows)
        if result is None:
            print(f"  {ch_id}: SKIPPED (not enough test data)")
            continue
        total_windows += result["n_windows"]
        total_mismatches += result["mismatches"]
        status = "OK" if result["mismatches"] == 0 else "MISMATCH FOUND"
        print(f"  {ch_id}: {result['n_windows']} windows, "
              f"{result['mismatches']} mismatches, "
              f"max score diff {result['max_score_diff']:.6f}  [{status}]")

    ok = total_mismatches == 0 and total_windows > 0
    print(f"\nSTAGE 2: {total_windows} windows checked, {total_mismatches} mismatches")
    print(f"STAGE 2 {'PASSED' if ok else 'FAILED'}\n")
    return ok


def stage3_backend_smoke_test(channels):
    print("=" * 70)
    print("STAGE 3: backend streaming integration smoke test")
    print("=" * 70)
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    scorer = CertSatTelemetryScorer(MODEL_PATH)

    all_ok = True
    for ch_id in channels:
        if ch_id not in manager.detectors:
            print(f"  {ch_id}: SKIPPED (not in model)")
            continue
        test_path = os.path.join(TEST_DIR, f"{ch_id}.npy")
        if not os.path.exists(test_path):
            print(f"  {ch_id}: SKIPPED (no test data)")
            continue

        test_arr = np.load(test_path)
        window_size = scorer.buffers[ch_id].maxlen if ch_id in scorer.buffers else None

        result = None
        for row in test_arr[:120]:
            result = scorer.handle_packet(ch_id, row.tolist())

        contract_keys = {"ch_id", "status", "is_anomaly", "label", "score", "n_samples"}
        shape_ok = result is not None and contract_keys.issubset(result.keys())
        scored_ok = shape_ok and result["status"] == "scored"

        # Cross-check against calling manager.predict() directly on the
        # same 120-row window through preprocess_live_window.
        from src.inference.live_window import preprocess_live_window
        X = preprocess_live_window(ch_id, test_arr[:120], manager)
        direct = manager.predict(ch_id, X)
        matches_direct = (
            scored_ok
            and result["label"] == int(direct["label"][0])
            and abs(result["score"] - float(direct["score"][0])) < 1e-9
        )

        ok = shape_ok and scored_ok and matches_direct
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"  {ch_id}: contract_shape={shape_ok} scored={scored_ok} "
              f"matches_direct_predict={matches_direct}  [{status}]")

    print(f"\nSTAGE 3 {'PASSED' if all_ok else 'FAILED'}\n")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--max-windows", type=int, default=150)
    parser.add_argument("--skip-stage1", action="store_true",
                         help="Skip the generic-interface pipeline test (slow, needs full archive/ data)")
    args = parser.parse_args()

    results = {}
    if not args.skip_stage1:
        results["stage1_generic_interface"] = stage1_generic_interface()
    else:
        print("STAGE 1 skipped by flag.\n")

    results["stage2_notebook_vs_api"] = stage2_notebook_vs_api(args.channels, args.max_windows)
    results["stage3_backend_smoke_test"] = stage3_backend_smoke_test(args.channels)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    overall_ok = all(results.values())
    print(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()