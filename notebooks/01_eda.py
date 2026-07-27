"""
EDA orchestration script (mirrors the "Exploratory Data Analysis" section
of nasa_smap_msl_explore.ipynb).

Run from the project root:
    python -m notebooks.01_eda

Or copy cells into a real .ipynb / Colab notebook -- the logic is identical,
just delegated to src/ modules instead of being inline.
"""

import pandas as pd

project_root="/Users/vidhimishra/Desktop/Anomaly Detection"
import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.loader import download_dataset, load_labels, build_shape_summary, build_imbalance_summary
from src.evaluation.eda_utils import build_scale_summary, check_feature_variance, plot_channel

pd.set_option("display.width", 120)


def main():
    paths = download_dataset()
    labels = load_labels(paths["labels_csv"])

    print("Shape:", labels.shape)
    print(labels.head())
    print("\nColumns:", labels.columns.tolist())
    print("\nNull Value Sum:")
    print(labels.isnull().sum())

    print("\nChannels per spacecraft:")
    print(labels["spacecraft"].value_counts())

    print("\nAnomaly class distribution (point vs contextual):")
    print(labels["class"].value_counts())

    print("\nAnomaly class distribution by spacecraft:")
    print(pd.crosstab(labels["spacecraft"], labels["class"]))

    # Anomaly sequence length distribution
    seq_lengths = []
    for _, row in labels.iterrows():
        for start, end in row["anomaly_sequences"]:
            seq_lengths.append(end - start)
    seq_lengths = pd.Series(seq_lengths)
    print("\nAnomaly sequence length distribution:")
    print(seq_lengths.describe())

    # Shape / NaN-Inf summary
    shape_df = build_shape_summary(labels, paths["train_dir"], paths["test_dir"])
    print("\nChannels with NaN/Inf issues:")
    nan_cols = ["train_has_nan", "train_has_inf", "test_has_nan", "test_has_inf"]
    print(shape_df[shape_df[nan_cols].any(axis=1)])
    print("\nPer-channel shape summary:")
    print(shape_df.head(10))
    print("\nDoes n_features vary across channels?")
    print(shape_df["n_features"].value_counts())

    # Class imbalance
    imbalance_df = build_imbalance_summary(labels, shape_df)
    print("\nAnomaly fraction stats across all channels:")
    print(imbalance_df["anomaly_fraction"].describe())

    # Sample plots: first SMAP and first MSL channel
    for spacecraft in ["SMAP", "MSL"]:
        sample_chan = labels[labels["spacecraft"] == spacecraft]["chan_id"].iloc[0]
        plot_channel(sample_chan, labels, paths["train_dir"], paths["test_dir"])

    # Train/test distribution shift
    scale_summary = build_scale_summary(labels, paths["train_dir"], paths["test_dir"])
    print("\nChannels with largest mean shift (all features averaged):")
    print(scale_summary.sort_values("mean_shift", ascending=False).head(10))

    # Dead/constant feature check on sample channels
    for spacecraft in ["SMAP", "MSL"]:
        sample_chan = labels[labels["spacecraft"] == spacecraft]["chan_id"].iloc[0]
        check_feature_variance(sample_chan, paths["train_dir"])

    print("\n--- EDA complete. Use shape_df / imbalance_df / scale_summary to decide ---")
    print("--- windowing strategy and per-channel scaling before Isolation Forest. ---")

    return labels, shape_df, imbalance_df, scale_summary


if __name__ == "__main__":
    main()