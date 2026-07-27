"""
Isolation Forest baseline orchestration script (mirrors the "Isolation
Forest" section of nasa_smap_msl_explore.ipynb).

Run from the project root:
    python -m notebooks.02_isolation_forest
"""

import os

project_root = "/Users/vidhimishra/Desktop/Anomaly Detection"

import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.data.loader import download_dataset, load_labels, build_shape_summary, build_imbalance_summary
from src.models.isolation_forest import run_isolation_forest
from src.evaluation.metrics import aggregate_results
from src.utils.config import WINDOW_SIZE, STRIDE
from src.models.isolation_forest import MultiChannelIsolationForest
manager = MultiChannelIsolationForest()

pd.set_option("display.width", 120)


def main():
    paths = download_dataset()
    labels = load_labels(paths["labels_csv"])
    shape_df = build_shape_summary(labels, paths["train_dir"], paths["test_dir"])
    imbalance_df = build_imbalance_summary(labels, shape_df)
    contam_lookup = dict(zip(imbalance_df["ch_id"], imbalance_df["anomaly_fraction"]))

    print(f"Using WINDOW_SIZE={WINDOW_SIZE}, STRIDE={STRIDE}")

    results, skipped_channels = run_isolation_forest(
        labels, shape_df, paths["train_dir"], paths["test_dir"],
        window_size=WINDOW_SIZE, stride=STRIDE, contam_lookup=contam_lookup,manager=manager,
    )
    print(f"\nSkipped {len(skipped_channels)} channels (too short for WINDOW_SIZE={WINDOW_SIZE}):")
    print(skipped_channels)
    print(f"\nSuccessfully modeled {len(results)} channels.")

    # F1 (beta=1.0); switch to beta=0.5 to match the project's primary F0.5 metric
    per_channel_df, overall = aggregate_results(results, WINDOW_SIZE, STRIDE, beta=1.0)

    print("\nPer-channel results:")
    print(per_channel_df.head(10))

    print(f"\nOverall micro-aggregated -> P: {overall['precision']:.4f}, "
          f"R: {overall['recall']:.4f}, F1: {overall['f1']:.4f}")

    print("\nMean F1 by spacecraft:")
    print(per_channel_df.groupby("spacecraft")["f1"].mean())

    print("\nMean F1 by anomaly class (point vs contextual):")
    print(per_channel_df.groupby("class")["f1"].mean())

    per_channel_df.to_csv(os.path.join(project_root, "experiments", "results","isolation_forest_per_channel_results.csv"), index=False)
    print("\nSaved per-channel IF results for benchmarking against LSTM/LSTM-AE.")

    return results, per_channel_df, overall,  manager

results, per_channel_df, overall, manager = main()

save_path = os.path.join(os.path.join(project_root, "models_saved"), "isoforest_all_channels.joblib")
manager.save(save_path)
print(f"\nSaved trained Isolation Forest model to {save_path}")