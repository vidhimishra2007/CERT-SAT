"""
scripts/test_generic_pipeline.py

Smoke test for the generic model interface (src/models/base_detector.py)
and pipeline (src/models/multi_channel.py), run against REAL SMAP/MSL
channel data (not synthetic) -- the same dataset and windowing settings
used everywhere else in this project. Proves that IsolationForest and LOF
both fit/predict/save/load through the exact same MultiChannelDetector
code path. Any future model (One-Class SVM, LSTM, ...) that conforms to
BaseDetector can be added to DETECTOR_CLASSES below to test it the
same way.

Requires the dataset to already be present locally at archive/ (same
layout src/data/loader.py expects: archive/labeled_anomalies.csv,
archive/data/data/train/*.npy, archive/data/data/test/*.npy) -- same
requirement as scripts/test_isolation_forest_scenarios.py.

Run:
    python scripts/test_generic_pipeline.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_channel_arrays, load_labels
from src.data.preprocessing import scale_channel_with_scaler
from src.data.windowing import create_windows_stats
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof_baseline import LOFDetector
from src.models.multi_channel import MultiChannelDetector
from src.utils.config import WINDOW_SIZE, STRIDE, get_paths

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")
TEST_CHANNEL = "P-1"  # has 3 labeled contextual anomalies -- see archive/labeled_anomalies.csv
DETECTOR_CLASSES = [IsolationForestDetector, LOFDetector]

results = []


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


def flagged_point_overlap(is_anomaly, test_len, window_size, stride, anomaly_sequences):
    """
    Diagnostic only (not a pass/fail check): what fraction of TRUE anomalous
    timesteps fall inside a window this detector flagged. Mirrors
    src.data.windowing.windows_to_point_labels, adapted for a plain bool
    is_anomaly array instead of sklearn's {-1, 1} convention.
    """
    point_pred = np.zeros(test_len)
    for i, flagged in enumerate(is_anomaly):
        if flagged:
            start = i * stride
            point_pred[start:min(start + window_size, test_len)] = 1
    point_true = np.zeros(test_len)
    for start, end in anomaly_sequences:
        point_true[start:min(end, test_len)] = 1
    true_positive_points = int(((point_pred == 1) & (point_true == 1)).sum())
    total_true_points = int(point_true.sum())
    return true_positive_points, total_true_points


def main():
    paths = get_paths(ARCHIVE_DIR)
    labels = load_labels(paths["labels_csv"])
    row = labels[labels["chan_id"] == TEST_CHANNEL].iloc[0]
    anomaly_sequences = row["anomaly_sequences"]

    train_raw, test_raw = load_channel_arrays(TEST_CHANNEL, paths["train_dir"], paths["test_dir"])
    if train_raw is None:
        print(f"Channel {TEST_CHANNEL} not found under {ARCHIVE_DIR} -- is the dataset present?")
        sys.exit(1)

    train_scaled, test_scaled, scaler = scale_channel_with_scaler(train_raw, test_raw)
    X_train = create_windows_stats(train_scaled, WINDOW_SIZE, STRIDE)
    X_test = create_windows_stats(test_scaled, WINDOW_SIZE, STRIDE)

    print(f"Channel {TEST_CHANNEL}: train {train_raw.shape}, test {test_raw.shape}, "
          f"WINDOW_SIZE={WINDOW_SIZE}, STRIDE={STRIDE} (from src.utils.config)")
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Ground-truth anomaly sequences: {anomaly_sequences}\n")

    for detector_cls in DETECTOR_CLASSES:
        print(f"--- {detector_cls.__name__} ({detector_cls.MODEL_TYPE}) ---")

        manager = MultiChannelDetector(detector_cls, window_size=WINDOW_SIZE)

        check(f"[{detector_cls.__name__}] fit_channel", lambda m=manager: (
            m.fit_channel(TEST_CHANNEL, X_train, contamination=0.1),
            m.scalers.__setitem__(TEST_CHANNEL, scaler),
        ))

        def _predict_shape(m=manager):
            result = m.predict(TEST_CHANNEL, X_test)
            assert set(result) == {"ch_id", "model_type", "is_anomaly", "anomaly_score", "n_samples"}, result.keys()
            assert result["is_anomaly"].dtype == bool, result["is_anomaly"].dtype
            assert result["is_anomaly"].shape == (X_test.shape[0],)
            assert result["anomaly_score"].shape == (X_test.shape[0],)
            assert result["n_samples"] == X_test.shape[0]
            assert result["is_anomaly"].sum() > 0, "expected at least one flagged window at contamination=0.1"

            tp, total_true = flagged_point_overlap(
                result["is_anomaly"], test_raw.shape[0], WINDOW_SIZE, STRIDE, anomaly_sequences
            )
            pct = 100 * tp / total_true if total_true else 0
            print(f"        diagnostic: {result['is_anomaly'].sum()}/{len(result['is_anomaly'])} windows flagged; "
                  f"{tp}/{total_true} true anomalous timesteps ({pct:.0f}%) covered by a flagged window")

        check(f"[{detector_cls.__name__}] predict() output contract", _predict_shape)

        def _raw_window(m=manager):
            raw_buffer = test_raw[:WINDOW_SIZE]
            result = m.predict_raw_window(TEST_CHANNEL, raw_buffer)
            assert result["n_samples"] == 1

        check(f"[{detector_cls.__name__}] predict_raw_window()", _raw_window)

        def _save_load(m=manager):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "artifact.joblib")
                m.save(path)
                loaded = MultiChannelDetector.load(path)
                assert loaded.model_type == m.model_type
                assert loaded.meta["model_type"] == detector_cls.MODEL_TYPE
                assert "interface_version" in loaded.meta
                result = loaded.predict(TEST_CHANNEL, X_test)
                assert result["n_samples"] == X_test.shape[0]

        check(f"[{detector_cls.__name__}] save() / load() round-trip", _save_load)
        print()

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
