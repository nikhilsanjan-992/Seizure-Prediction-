"""NeuroSelect Phase 6 evaluation (real test-set metrics + plots).

Every metric is computed on the UNTOUCHED test split of a configuration and
persisted to results/. The classification threshold is fixed at 0.5 and
never tuned on the test set. Metrics with an undefined denominator are
reported as ``null`` (JSON) / ``NA`` (CSV) - never a fabricated number.

Metric formulas (identical to sklearn.metrics definitions; the sklearn
confusion matrix is used for counts and roc_auc_score for ROC-AUC):

    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    precision   = TP / (TP + FP)
    recall      = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1          = 2 * TP / (2 * TP + FP + FN)
    roc_auc     = sklearn.metrics.roc_auc_score (both classes must exist)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from config import METRICS_DIR, PLOTS_DIR  # noqa: F401
except ImportError:  # pragma: no cover - fallback when src/ is on sys.path
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import METRICS_DIR, PLOTS_DIR  # noqa: F401

DEFAULT_THRESHOLD = 0.5

COMPARISON_COLUMNS = [
    "configuration", "channels", "accuracy", "precision", "recall",
    "specificity", "f1", "roc_auc", "test_samples", "positive_samples",
    "negative_samples",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_scores: np.ndarray | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute binary classification metrics from REAL test predictions.

    ``y_true`` must contain only 0/1. ``y_pred`` is the thresholded binary
    prediction; ``y_pred_scores`` (real-valued model output) is optional and
    required for ROC-AUC. Undefined metrics are returned as ``None``.
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true / y_pred must be 1-D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch.")
    classes = np.unique(y_true)
    if not set(classes.tolist()).issubset({0, 1}):
        raise ValueError(
            f"Test labels must be binary (0/1); found {classes.tolist()}."
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    denom = tp + fp + fn + tn
    accuracy = float((tp + tn) / denom) if denom > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else None

    roc_auc = None
    if y_pred_scores is not None and len(classes) == 2:
        scores = np.asarray(y_pred_scores).reshape(-1)
        if len(scores) == len(y_true) and np.isfinite(scores).all():
            roc_auc = float(roc_auc_score(y_true, scores))
        else:
            roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "test_samples": int(len(y_true)),
        "positive_samples": int((y_true == 1).sum()),
        "negative_samples": int((y_true == 0).sum()),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Predict on the untouched test split and compute the metric dict."""
    probs = np.asarray(model.predict(X_test, verbose=0))
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs[:, 0]
    y_pred = (probs >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_pred_scores=probs,
                              threshold=threshold)
    metrics["score_threshold"] = threshold
    return metrics


def _fmt(value, na="NA"):
    if value is None:
        return na
    return f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Outputs (writes nothing unless called from a real experiment run)
# ---------------------------------------------------------------------------
def save_comparison_csv(rows: list[dict], path=None) -> Path:
    """Write the 22/5/4 comparison table (real values only)."""
    if path is None:
        path = METRICS_DIR / "comparison.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k, None) for k in COMPARISON_COLUMNS}
            for key in ("accuracy", "precision", "recall",
                        "specificity", "f1", "roc_auc"):
                flat[key] = _fmt(flat.get(key))
            writer.writerow(flat)
    return path


def save_error_analysis(analysis: dict, path=None) -> Path:
    """error_analysis.json with observed TN/FP/FN/TP per configuration."""
    if path is None:
        path = METRICS_DIR / "error_analysis.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)
    return path


def plot_confusion_matrix(metrics: dict, title: str, save_path) -> Path:
    """One confusion-matrix figure (actual test predictions)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Background", "Seizure"])
    ax.set_yticklabels(["Background", "Seizure"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                    fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_training_curve(history, save_path, title: str = "Training curve") -> Path:
    """Training/validation loss (+ metric) from the real Keras history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = history.history if hasattr(history, "history") else history
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(list(epochs), history.get("loss", []), "o-", label="train loss")
    if "val_loss" in history:
        ax.plot(list(epochs), history["val_loss"], "s--", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} - loss")
    ax.legend()

    ax = axes[1]
    if "accuracy" in history:
        ax.plot(list(epochs), history["accuracy"], "o-", label="train accuracy")
    if "val_accuracy" in history:
        ax.plot(list(epochs), history["val_accuracy"], "s--",
                label="val accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - accuracy")
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_channel_comparison(rows: list[dict], save_path) -> Path:
    """One honest 22-vs-5-vs-4 metric comparison (0-1 scale, no tricks)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(r["configuration"]) for r in rows]
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["Accuracy", "Precision", "Recall/Sens", "F1"]
    x = np.arange(len(metrics_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, row in enumerate(rows):
        values = [row.get(k) for k in metrics_keys]
        values = [v if v is not None else 0.0 for v in values]
        offset = (i - (len(rows) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=labels[i])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("22 vs 5 vs 4 channels - CNN-BiLSTM test-set comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path