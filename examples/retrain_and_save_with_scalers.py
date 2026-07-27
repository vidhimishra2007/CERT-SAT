"""Retrain and save a backend artifact that includes per-channel scalers.

This script requires access to the NASA SMAP/MSL dataset through kagglehub.
The resulting artifact can be used with src.inference.live_window for raw
telemetry buffers.
"""
project_root="/Users/vidhimishra/Desktop/Anomaly Detection"
import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path

from src.data.loader import build_shape_summary, download_dataset, load_labels
from src.utils.config import get_paths
from src.models.isolation_forest import MultiChannelIsolationForest, run_isolation_forest
from src.utils.config import IF_DEFAULT_CONTAMINATION, STRIDE, WINDOW_SIZE


OUTPUT_PATH = "models_saved/isoforest_all_channels_with_scalers.joblib"
LOCAL_DATASET_PATH = Path(
    "/Users/vidhimishra/.cache/kagglehub/datasets/"
    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl/versions/1"
)


def resolve_dataset_paths():
    if LOCAL_DATASET_PATH.exists():
        return get_paths(str(LOCAL_DATASET_PATH))
    return download_dataset()


def main():
    paths = resolve_dataset_paths()
    labels = load_labels(paths["labels_csv"])
    shape_df = build_shape_summary(labels, paths["train_dir"], paths["test_dir"])

    manager = MultiChannelIsolationForest()
    run_isolation_forest(
        labels=labels,
        shape_df=shape_df,
        train_dir=paths["train_dir"],
        test_dir=paths["test_dir"],
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        contam_lookup={},
        manager=manager,
    )
    manager.save(OUTPUT_PATH)

    print(f"Saved {len(manager.detectors)} detectors to {OUTPUT_PATH}")
    print(f"Saved {len(manager.scalers)} scalers using contamination={IF_DEFAULT_CONTAMINATION}")


if __name__ == "__main__":
    main()
