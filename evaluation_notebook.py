"""
evaluation_notebook.py

Full evaluation notebook for the CERT-SAT Isolation Forest model:
confusion matrix, precision/recall/F1, ROC-AUC, and real top-5
false-positive / false-negative windows per channel.

WHY THIS ISN'T PRE-RUN: this script depends on `src/data/loader.py`,
`src/models/isolation_forest.py`, `src/evaluation/metrics.py`,
`src/utils/config.py`, and the raw `archive/data/data/{train,test}/*.npy`
files. None of these were present in the uploaded Anomaly_Detection.zip
(only src/__init__.py and archive/data/.DS_Store came through), so this
could not be executed in this session. Drop this file into your project
root and run it there -- it uses the same function signatures your own
notebooks/02_isolation_forest.py and scripts/validate_backend_integration.py
already call, so it shouldn't need edits.

Usage:
    python evaluation_notebook.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)

from src.data.loader import download_dataset, load_labels
from src.data.windowing import create_windows_stats
from src.models.isolation_forest import MultiChannelIsolationForest
from src.utils.config import WINDOW_SIZE, STRIDE

MODEL_PATH = "models_saved/isoforest_all_channels_with_scalers.joblib"


def label_windows(num_values, anomaly_sequences, n_windows):
    """A window is a true anomaly if ANY raw timestep inside it falls in
    a labeled anomaly sequence (matches the windowing convention used
    elsewhere in this project)."""
    starts = np.arange(n_windows) * STRIDE
    ends = starts + WINDOW_SIZE
    y = np.zeros(n_windows, dtype=int)
    for s, e in anomaly_sequences:
        y |= ((starts < e) & (ends > s)).astype(int)
    return y


def evaluate_channel(ch_id, manager, test_dir, labels_row):
    test_arr = np.load(f"{test_dir}/{ch_id}.npy")
    scaler = manager.scalers[ch_id]
    det = manager.detectors[ch_id]

    X_scaled = scaler.transform(test_arr)
    X_windows = create_windows_stats(X_scaled, WINDOW_SIZE, STRIDE)
    if len(X_windows) == 0:
        return None

    # -1 = anomaly, 1 = normal (sklearn IsolationForest convention)
    y_pred_raw = det.predict(X_windows)
    y_pred = (y_pred_raw == -1).astype(int)
    raw_scores = det.score(X_windows)
    # lower score = more anomalous -> flip sign so higher = more anomalous,
    # consistent with sklearn's roc_auc_score expecting "higher = positive class"
    # scores = -det.score(X_windows)
    scores = -raw_scores

    y_true = label_windows(labels_row["num_values"], labels_row["anomaly_sequences"], len(X_windows))

    return {
        "ch_id": ch_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "raw_scores": raw_scores,
        "scores": scores,
        "X_windows": X_windows,
    }


def main():
    paths = download_dataset()
    labels = load_labels(paths["labels_csv"])
    manager = MultiChannelIsolationForest.load(MODEL_PATH)

    all_results = []
    for _, row in labels.iterrows():
        ch_id = row["chan_id"]
        if ch_id not in manager.detectors:
            continue
        res = evaluate_channel(ch_id, manager, paths["test_dir"], row)
        if res is not None:
            all_results.append(res)

    y_true_all = np.concatenate([r["y_true"] for r in all_results])
    y_pred_all = np.concatenate([r["y_pred"] for r in all_results])
    raw_scores_all = np.concatenate([r["raw_scores"] for r in all_results])
    scores_all = np.concatenate([r["scores"] for r in all_results])

    # --- i) Confusion matrix ---
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("Confusion matrix [[TN FP][FN TP]]:\n", cm)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Normal", "True Anomaly"])
    ax.set_title("Aggregate Confusion Matrix (all channels)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    # --- ii) Precision / Recall / F1 / ROC-AUC ---
    precision = precision_score(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all)
    auc = roc_auc_score(y_true_all, scores_all)
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} ROC-AUC={auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true_all, scores_all)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (all channels pooled)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")

    # --- iv) Top 5 false positives / false negatives, by confidence ---
    rows = []
    for r in all_results:
        for i in range(len(r["y_true"])):
            rows.append({
                "ch_id": r["ch_id"],
                "y_true": r["y_true"][i],
                "y_pred": r["y_pred"][i],
                "raw_score": r["raw_scores"][i],
                "score": r["scores"][i],
                "window_idx": i,
            })
    df = pd.DataFrame(rows)

    fp = df[(df.y_true == 0) & (df.y_pred == 1)].sort_values("score", ascending=False).head(5)
    fn = df[(df.y_true == 1) & (df.y_pred == 0)].sort_values("score").head(5)
    tp = df[(df.y_true == 1) & (df.y_pred == 1)].sort_values("score", ascending=False).head(5)
    tn = df[(df.y_true == 0) & (df.y_pred == 0)].sort_values("score").head(5)

    print("\nTop 5 false positives (model most confident, wrongly):")
    print(fp.to_string(index=False))
    print("\nTop 5 false negatives (true anomalies model was most confident were normal):")
    print(fn.to_string(index=False))
    print("\nTop 5 true positives (model most confident, correctly):")
    print(tp.to_string(index=False))
    print("\nTop 5 true negatives (model most confident, correctly):")
    print(tn.to_string(index=False))

    df["abs_raw_score"] = df["raw_score"].abs()
    borderline = df.sort_values("abs_raw_score").head(15)
    # .drop(columns="abs_raw_score")
    print("\nTop 15 borderline windows (raw_score closest to the decision boundary):")
    print(borderline.to_string(index=False))

    df.to_csv("all_window_predictions.csv", index=False)
    fp.to_csv("top5_false_positives.csv", index=False)
    fn.to_csv("top5_false_negatives.csv", index=False)
    tp.to_csv("top5_true_positives.csv", index=False)
    tn.to_csv("top5_true_negatives.csv", index=False)
    borderline.to_csv("top15_borderline_windows.csv", index=False)


if __name__ == "__main__":
    main()