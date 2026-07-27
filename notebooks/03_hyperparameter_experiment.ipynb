"""
Hyperparameter experiment: contamination strategy for Isolation Forest.

Experiment with at least one hyperparameter.

This script runs Isolation Forest three times, varying ONLY the
`contamination` parameter, holding everything else (window_size, stride,
n_estimators, random_state) fixed -- so any difference in results is
attributable to this one change.

Run from the project root:
    python -m notebooks.03_hyperparameter_experiment
"""

# ---------------------------------------------------------------------------
# EDIT THIS: set to the absolute path of your CERT-SAT project root folder
# ---------------------------------------------------------------------------
project_root = "/Users/vidhimishra/Desktop/Anomaly Detection"

import sys
import os
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.data.loader import download_dataset, load_labels, build_shape_summary, build_imbalance_summary
from src.models.isolation_forest import run_isolation_forest
from src.evaluation.metrics import aggregate_results
from src.utils.config import WINDOW_SIZE, STRIDE

pd.set_option("display.width", 120)


def main():
    paths = download_dataset()
    labels = load_labels(paths["labels_csv"])
    shape_df = build_shape_summary(labels, paths["train_dir"], paths["test_dir"])
    imbalance_df = build_imbalance_summary(labels, shape_df)

    # --- Define the three contamination strategies being compared ---
    # Strategy 1: fixed low contamination (conservative -- assumes anomalies are rare)
    fixed_low = {ch_id: 0.05 for ch_id in labels["chan_id"]}

    # Strategy 2: fixed moderate contamination (assumes anomalies are more common)
    fixed_moderate = {ch_id: 0.15 for ch_id in labels["chan_id"]}

    # Strategy 3: oracle contamination (derived from TRUE test-set anomaly
    # fraction via imbalance_df). This leaks test labels into a
    # hyperparameter -- included deliberately as an upper-bound reference
    # point, not a fair baseline. See README "Known limitations".
    oracle = dict(zip(imbalance_df["ch_id"], imbalance_df["anomaly_fraction"]))

    strategies = {
        "fixed_low_0.05": fixed_low,
        "fixed_moderate_0.15": fixed_moderate,
        "oracle_leaky": oracle,
    }

    all_per_channel = {}
    all_overall = {}

    for name, contam_lookup in strategies.items():
        print(f"\n=== Running Isolation Forest with contamination strategy: {name} ===")
        results, skipped = run_isolation_forest(
            labels, shape_df, paths["train_dir"], paths["test_dir"],
            window_size=WINDOW_SIZE, stride=STRIDE, contam_lookup=contam_lookup,
        )
        per_channel_df, overall = aggregate_results(results, WINDOW_SIZE, STRIDE, beta=1.0)
        all_per_channel[name] = per_channel_df
        all_overall[name] = overall
        print(f"{name} -> P: {overall['precision']:.4f}, R: {overall['recall']:.4f}, F1: {overall['f1']:.4f}")
        

    # --- Summary comparison table ---
    summary_rows = []
    for name, overall in all_overall.items():
        summary_rows.append({
            "strategy": name,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Contamination strategy comparison (overall, micro-aggregated) ===")
    print(summary_df)

    # Save outputs
    output_dir = os.path.join(project_root, "experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, "contamination_experiment_summary.csv"), index=False)
    for name, df in all_per_channel.items():
        df.to_csv(os.path.join(output_dir, f"if_per_channel_{name}.csv"), index=False)
    print(f"\nSaved experiment results to {output_dir}")

    return summary_df, all_per_channel, all_overall


if __name__ == "__main__":
    main()
