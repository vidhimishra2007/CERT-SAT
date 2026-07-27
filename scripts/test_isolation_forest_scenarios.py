"""
scripts/test_isolation_forest_scenarios.py

Scenario + edge-case test suite for the CERT-SAT Isolation Forest model,
covering both integration entrypoints:
    - manager.predict(ch_id, X_features)          (pre-windowed input)
    - predict_raw(manager, ch_id, raw_buffer) (raw telemetry input) --
      a thin test-local wrapper around the two existing library calls
      (preprocess_live_window + manager.predict), matching the pattern
      backend integrators should use for raw telemetry.

Run:
    python scripts/test_isolation_forest_scenarios.py

Each check prints PASS/FAIL. Exit code is nonzero if anything fails.
"""

import sys
import os
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.isolation_forest import MultiChannelIsolationForest
from src.inference.live_window import preprocess_live_window
from src.utils.config import WINDOW_SIZE

MODEL_PATH = "models_saved/isoforest_all_channels_with_scalers.joblib"
TEST_DIR = "archive/data/data/test"

results = []


def predict_raw(manager, ch_id, raw_buffer):
    """
    Test-local helper: the two-step raw-telemetry path that already exists
    in the codebase (preprocess_live_window + predict). Not a new library
    method -- this just saves repeating the two calls in every scenario
    below, the same way a backend integrator should chain them.
    """
    X = preprocess_live_window(ch_id, raw_buffer, manager)
    return manager.predict(ch_id, X)


def check(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"  PASS  {name}")
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def main():
    print(f"Loading {MODEL_PATH} ...")
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    ch_id = "P-1"
    n_features = manager.detectors[ch_id].model.n_features_in_ // 5  # 5 stats per raw feature
    test_arr = np.load(os.path.join(TEST_DIR, f"{ch_id}.npy"))
    good_window = test_arr[:WINDOW_SIZE]

    print(f"\nChannel {ch_id}: raw n_features={n_features}, test_arr shape={test_arr.shape}\n")
    print("=== 1. Normal / happy-path scenarios ===")

    def normal_prediction():
        r = predict_raw(manager, ch_id, good_window)
        assert r["label"][0] in (1, -1)
        assert r["n_samples"] == 1
        assert r["ch_id"] == ch_id

    check("normal raw window scores without error", normal_prediction)

    def known_anomaly_window():
        # Try several offsets; at least confirm scoring runs and returns
        # a finite score everywhere (doesn't crash on real anomalous data).
        for start in range(0, len(test_arr) - WINDOW_SIZE, 500):
            r = predict_raw(manager, ch_id, test_arr[start:start + WINDOW_SIZE])
            assert np.isfinite(r["score"][0]), f"non-finite score at start={start}"

    check("scores finite across many real windows", known_anomaly_window)

    def live_path_matches_offline_batch_path():
        from src.data.preprocessing import scale_channel_with_scaler
        from src.data.windowing import create_windows_stats

        # Offline/batch path: fit-independent, uses this channel's saved
        # scaler via .transform() only (never refit), mirroring how
        # run_isolation_forest() scales+windows a whole array at once.
        scaler = manager.scalers[ch_id]
        # transform-only, same as preprocess_live_window does internally
        offline_scaled = scaler.transform(good_window)
        X_offline = create_windows_stats(offline_scaled, WINDOW_SIZE)
        r_offline = manager.predict(ch_id, X_offline)

        # Live/streaming path: raw buffer -> preprocess_live_window -> predict
        r_live = predict_raw(manager, ch_id, good_window)

        assert r_offline["label"][0] == r_live["label"][0]
        assert np.isclose(r_offline["score"][0], r_live["score"][0])

    check(
        "live path (preprocess_live_window+predict) matches offline batch path",
        live_path_matches_offline_batch_path,
    )

    print("\n=== 2. Malformed input edge cases ===")

    def wrong_row_count_short():
        try:
            predict_raw(manager, ch_id, good_window[:119])
            assert False, "expected ValueError for 119-row buffer, got no error"
        except ValueError:
            pass

    check("buffer with 119 rows (one short) raises ValueError", wrong_row_count_short)

    def wrong_row_count_long():
        try:
            predict_raw(manager, ch_id, test_arr[:121])
            assert False, "expected ValueError for 121-row buffer, got no error"
        except ValueError:
            pass

    check("buffer with 121 rows (one extra) raises ValueError", wrong_row_count_long)

    def empty_buffer():
        try:
            predict_raw(manager, ch_id, np.empty((0, n_features)))
            assert False, "expected ValueError for empty buffer"
        except ValueError:
            pass

    check("empty buffer raises ValueError", empty_buffer)

    def unknown_channel():
        try:
            predict_raw(manager, "NOT-A-REAL-CHANNEL", good_window)
            assert False, "expected KeyError for unknown channel"
        except KeyError:
            pass

    check("unsupported channel id raises KeyError", unknown_channel)

    def wrong_feature_count():
        bad_window = good_window[:, :-1]  # drop one feature column
        try:
            predict_raw(manager, ch_id, bad_window)
            assert False, "expected an error for wrong feature count, got a silent result"
        except (ValueError, KeyError) as e:
            pass

    check("wrong raw feature count raises (not a silent wrong answer)", wrong_feature_count)

    print("\n=== 3. Degenerate / extreme value scenarios ===")

    def constant_window():
        # std = 0 across the whole window -- watch for div-by-zero / NaN
        const_window = np.tile(good_window[0], (WINDOW_SIZE, 1))
        r = predict_raw(manager, ch_id, const_window)
        assert np.isfinite(r["score"][0]), "constant window produced non-finite score"

    check("constant-value window (zero variance) does not produce NaN/Inf", constant_window)

    def all_zero_window():
        zero_window = np.zeros_like(good_window)
        r = predict_raw(manager, ch_id, zero_window)
        assert np.isfinite(r["score"][0])

    check("all-zero raw window does not crash or produce NaN", all_zero_window)

    def extreme_value_window():
        spike_window = good_window.copy()
        spike_window[-1] = spike_window[-1] * 1e6 + 1e6  # simulate a sensor spike
        r = predict_raw(manager, ch_id, spike_window)
        assert np.isfinite(r["score"][0])
        # A window ending in a massive spike should score more anomalous
        # (lower score) than the unmodified window.
        r_normal = predict_raw(manager, ch_id, good_window)
        assert r["score"][0] < r_normal["score"][0], (
            "extreme spike window did not score more anomalous than baseline"
        )

    check("extreme sensor-spike window scores more anomalous than baseline", extreme_value_window)

    def nan_in_buffer():
        nan_window = good_window.copy()
        nan_window[5, 0] = np.nan
        try:
            r = predict_raw(manager, ch_id, nan_window)
            # sklearn IsolationForest raises on NaN by default in recent
            # versions; if it doesn't, the score must at least not be NaN
            # silently accepted as a valid "normal" reading.
            assert not np.isnan(r["score"][0]), (
                "NaN telemetry value silently produced a NaN score "
                "(should raise or be rejected upstream)"
            )
        except ValueError:
            pass  # acceptable: sklearn rejects NaN input

    check("NaN in raw telemetry is rejected or does not silently pass through", nan_in_buffer)

    def inf_in_buffer():
        inf_window = good_window.copy()
        inf_window[10, 0] = np.inf
        try:
            r = predict_raw(manager, ch_id, inf_window)
            assert np.isfinite(r["score"][0]), "Inf telemetry value silently produced a non-finite score"
        except ValueError:
            pass

    check("Inf in raw telemetry is rejected or does not silently pass through", inf_in_buffer)

    print("\n=== 4. Batch / vectorized scenarios ===")

    def multiple_windows_batch():
        from src.data.windowing import create_windows_stats
        from src.utils.config import STRIDE
        scaler = manager.scalers[ch_id]
        scaled = scaler.transform(test_arr[:1000])
        X = create_windows_stats(scaled, WINDOW_SIZE, STRIDE)
        r = manager.predict(ch_id, X)
        assert r["n_samples"] == len(X)
        assert len(r["label"]) == len(X)
        assert np.all(np.isfinite(r["score"]))

    check("batch of many windows scores correctly with finite scores", multiple_windows_batch)

    def single_row_2d_shape_required():
        # raw_buffer must be 2D (window_size, n_features); a flattened 1D
        # array of the right total length should NOT silently work.
        flat = good_window.flatten()
        try:
            predict_raw(manager, ch_id, flat)
            assert False, "expected an error for 1D flattened input"
        except (ValueError, IndexError):
            pass

    check("1D flattened buffer is rejected rather than silently misinterpreted", single_row_2d_shape_required)

    print("\n=== 5. Model artifact integrity ===")

    def all_channels_loadable():
        assert len(manager.detectors) == len(manager.scalers), (
            f"detector/scaler count mismatch: "
            f"{len(manager.detectors)} detectors vs {len(manager.scalers)} scalers"
        )

    check("every trained channel has a matching persisted scaler", all_channels_loadable)

    def spot_check_multiple_channels():
        import os
        sample_channels = ["P-1", "S-1", "E-1", "A-1", "T-1"]
        for cid in sample_channels:
            if cid not in manager.detectors:
                continue
            arr = np.load(os.path.join(TEST_DIR, f"{cid}.npy"))
            if len(arr) < WINDOW_SIZE:
                continue
            r = predict_raw(manager, cid, arr[:WINDOW_SIZE])
            assert np.isfinite(r["score"][0])

    check("spot-check across multiple channels (P-1, S-1, E-1, A-1, T-1)", spot_check_multiple_channels)

    def raw_buffer_passed_directly_to_predict_fails_loud():
        # Regression guard: examples/raw_telemetry_service_example.py was
        # once edited to call manager.predict(ch_id, raw_buffer) directly,
        # skipping preprocess_live_window(). predict() expects a windowed
        # feature vector (1, n_raw_features*5), not a raw (WINDOW_SIZE,
        # n_raw_features) buffer. Confirm this fails loudly (shape
        # mismatch) rather than silently returning a meaningless score --
        # integrators must always call preprocess_live_window() first for
        # raw telemetry (see predict_raw() helper above).
        try:
            manager.predict(ch_id, good_window)
            assert False, (
                "manager.predict() accepted a raw (unwindowed) buffer "
                "without error -- this should fail loudly on shape mismatch"
            )
        except ValueError:
            pass

    check(
        "regression guard: predict() rejects raw buffers (use preprocess_live_window first)",
        raw_buffer_passed_directly_to_predict_fails_loud,
    )
    
    print("\n=== 6. Additional edge cases ===")

    def my_new_scenario():
    # 1. set up an input
        weird_window = good_window.copy()
        weird_window[0] = weird_window[0] * -1   # example: flip first row's sign

    # 2. run it through the model
        r = predict_raw(manager, ch_id, weird_window)

    # 3. assert what "correct" looks like
        assert np.isfinite(r["score"][0]), "score should not be NaN/Inf"

    check("first row sign-flipped does not crash", my_new_scenario)

    def sudden_dropout_to_zero():
        dropout = good_window.copy()
        dropout[90:] = 0.0  # sensor drops to 0 for the last 30 timesteps
        r = predict_raw(manager, ch_id, dropout)
        assert np.isfinite(r["score"][0])

    check("sensor dropout to zero in tail of window", sudden_dropout_to_zero)

    def repeated_calls_are_deterministic():
        r1 = predict_raw(manager, ch_id, good_window)
        r2 = predict_raw(manager, ch_id, good_window)
        assert r1["label"][0] == r2["label"][0]
        assert np.isclose(r1["score"][0], r2["score"][0])

    check("same input scored twice gives identical result", repeated_calls_are_deterministic)

    def cross_channel_misuse():
        other_ch = "T-1"
        n_p1 = good_window.shape[1]
        n_other = manager.detectors[other_ch].model.n_features_in_ // 5
        try:
            X = preprocess_live_window(other_ch, good_window, manager)
            r = manager.predict(other_ch, X)
            assert n_p1 != n_other or np.isfinite(r["score"][0])
        except (ValueError, KeyError):
            pass  # also acceptable: shape mismatch caught

    check("cross-channel misuse (P-1 data through T-1 model) does not silently corrupt output type", cross_channel_misuse)

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"TOTAL: {len(results)} scenarios, {n_pass} passed, {n_fail} failed")
    if n_fail:
        print("\nFAILED SCENARIOS:")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("ALL SCENARIOS PASSED")


if __name__ == "__main__":
    main()