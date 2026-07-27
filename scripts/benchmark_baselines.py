"""
scripts/benchmark_baselines.py

Benchmark: Isolation Forest (CERT-SAT's production model) vs. Local Outlier
Factor (classical baseline), run through the identical pipeline:

    same channels, same StandardScaler-per-channel, same summary-statistic
    windowing (window_size=120, stride=10), same fixed contamination=0.05
    (leakage-free -- no test-label lookup), same point-adjust + micro-averaged
    precision/recall/F1 aggregation.

Only the model itself changes, so any difference in the results table is
attributable to the model, not the data pipeline.

Run from the project root:
    python -m scripts.benchmark_baselines

Set `local_data_root` below if you already have the SMAP/MSL archive on
disk (avoids a kagglehub download).
"""

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# If the archive is already extracted locally, point directly at it to skip
# the kagglehub download. Falls back to download_dataset() otherwise.
local_data_root = os.path.join(project_root, "archive")

import pandas as pd

from src.data.loader import load_labels, build_shape_summary
from src.models.isolation_forest import run_isolation_forest
from src.models.lof_baseline import run_lof
from src.evaluation.metrics import aggregate_results
from src.utils.config import WINDOW_SIZE, STRIDE, IF_DEFAULT_CONTAMINATION

pd.set_option("display.width", 120)


def resolve_paths():
    """Use the local archive if present, else fall back to kagglehub download."""
    if os.path.isdir(local_data_root):
        labels_csv = os.path.join(local_data_root, "labeled_anomalies.csv")
        base_npy_dir = os.path.join(local_data_root, "data", "data")
        return {
            "labels_csv": labels_csv,
            "base_npy_dir": base_npy_dir,
            "train_dir": os.path.join(base_npy_dir, "train"),
            "test_dir": os.path.join(base_npy_dir, "test"),
        }
    from src.data.loader import download_dataset
    return download_dataset()


def main():
    paths = resolve_paths()
    labels = load_labels(paths["labels_csv"])
    shape_df = build_shape_summary(labels, paths["train_dir"], paths["test_dir"])

    # Fixed, leakage-free contamination for both models -- no test-set
    # anomaly-fraction lookup, matching the IF "fixed_low_0.05" baseline
    # already reported in experiments/results/.
    fixed_contam = {ch_id: IF_DEFAULT_CONTAMINATION for ch_id in labels["chan_id"]}

    all_per_channel = {}
    all_overall = {}
    timings = {}
    skip_counts = {}

    print("=== Running Isolation Forest ===")
    t0 = time.time()
    if_results, if_skipped = run_isolation_forest(
        labels, shape_df, paths["train_dir"], paths["test_dir"],
        window_size=WINDOW_SIZE, stride=STRIDE, contam_lookup=fixed_contam,
    )
    timings["Isolation Forest"] = time.time() - t0
    skip_counts["Isolation Forest"] = len(if_skipped)
    if_per_channel, if_overall = aggregate_results(if_results, WINDOW_SIZE, STRIDE, beta=1.0)
    all_per_channel["Isolation Forest"] = if_per_channel
    all_overall["Isolation Forest"] = if_overall
    print(f"Isolation Forest -> P: {if_overall['precision']:.4f}, R: {if_overall['recall']:.4f}, "
          f"F1: {if_overall['f1']:.4f}  ({timings['Isolation Forest']:.1f}s, "
          f"{len(if_results)} channels modeled, {len(if_skipped)} skipped)")

    print("\n=== Running Local Outlier Factor (novelty=True, n_neighbors=20) ===")
    t0 = time.time()
    lof_results, lof_skipped = run_lof(
        labels, shape_df, paths["train_dir"], paths["test_dir"],
        window_size=WINDOW_SIZE, stride=STRIDE, contam_lookup=fixed_contam,
        n_neighbors=20,
    )
    timings["Local Outlier Factor"] = time.time() - t0
    skip_counts["Local Outlier Factor"] = len(lof_skipped)
    lof_per_channel, lof_overall = aggregate_results(lof_results, WINDOW_SIZE, STRIDE, beta=1.0)
    all_per_channel["Local Outlier Factor"] = lof_per_channel
    all_overall["Local Outlier Factor"] = lof_overall
    print(f"Local Outlier Factor -> P: {lof_overall['precision']:.4f}, R: {lof_overall['recall']:.4f}, "
          f"F1: {lof_overall['f1']:.4f}  ({timings['Local Outlier Factor']:.1f}s, "
          f"{len(lof_results)} channels modeled, {len(lof_skipped)} skipped)")

    # --- Summary comparison table ---
    summary_rows = []
    for name, overall in all_overall.items():
        summary_rows.append({
            "model": name,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "channels_modeled": len(if_results) if name == "Isolation Forest" else len(lof_results),
            "channels_skipped": skip_counts[name],
            "wall_clock_seconds": round(timings[name], 1),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Benchmark summary (overall, micro-aggregated, contamination=0.05 fixed) ===")
    print(summary_df.to_string(index=False))

    # Save outputs
    output_dir = os.path.join(project_root, "experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, "baseline_comparison_summary.csv"), index=False)
    if_per_channel.to_csv(os.path.join(output_dir, "if_per_channel_benchmark.csv"), index=False)
    lof_per_channel.to_csv(os.path.join(output_dir, "lof_per_channel_benchmark.csv"), index=False)
    print(f"\nSaved benchmark results to {output_dir}")

    return summary_df, all_per_channel, all_overall


if __name__ == "__main__":
    main()