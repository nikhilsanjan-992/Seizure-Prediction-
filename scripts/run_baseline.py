"""NeuroSelect Phase 7 Random Forest baseline (REAL EEG only).

A single lightweight comparator for the Phase 6 CNN-BiLSTM results:

    RandomForestClassifier on per-window summary statistics, using the SAME
    training/test split, the SAME recordings, the SAME per-channel z-score
    normalization (training statistics only) and the SAME 22 / 5 / 4 montages
    as the deep model. Only the model class differs.

Feature representation (deliberately simple): for every labeled window and
every channel the baseline extracts 4 summary statistics -

    mean, std, root-mean-square, peak-to-peak

- producing an (n_windows, channels * 4) matrix. Trees are scale-ambivalent
per feature, but the features are still computed AFTER the standard
training-only z-score so the baseline consumes the same normalized windows.

Honesty gates (identical in spirit to run_experiment.py; nothing is written
until a gate passes):
    1. REAL EDF files must exist under data/raw/          (else exit 1)
    2. Phase 4 selection must exist: results/metrics/
       selected_channels.json with top_5 / top_4          (else exit 3)
    3. Every split must produce labeled windows           (else exit 2)

results/metrics/baseline_comparison.csv compares each configuration's
Random Forest test metrics with the CNN-BiLSTM row from the Phase 6
results/metrics/comparison.csv when that file exists; otherwise the
CNN-BiLSTM rows are omitted and a note records that they were unavailable.

NO superiority claim is made: the comparison is only written and read back
after real values exist.

Usage:
    python scripts/run_baseline.py
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import dataset as ds  # noqa: E402
import evaluate as ev  # noqa: E402

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_CLASS_WEIGHT = "balanced"
RF_THRESHOLD = ev.DEFAULT_THRESHOLD

COMPARISON_COLUMNS = [
    "configuration", "channels", "model", "accuracy", "precision",
    "recall", "specificity", "f1", "roc_auc", "test_samples", "note",
]


def print_blocker(message: str) -> None:
    print("BLOCKED")
    print("-" * 40)
    print(message)


def window_summary_features(X: np.ndarray) -> np.ndarray:
    """Per-window summary statistics for the RF baseline.

    ``X`` is the Phase 6 model-orientation array
    ``(n_windows, timesteps, channels)`` (already normalized with
    training-only statistics). Returns ``(n_windows, channels * 4)``
    concatenated as mean, std, RMS, peak-to-peak per channel.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    ptp = X.max(axis=1) - X.min(axis=1)
    return np.ascontiguousarray(np.concatenate([mu, sd, rms, ptp], axis=1))


def build_config_files(split: dict, channels: list[str]):
    """Per-split arrays for one montage (train-only normalization stats)."""
    train_files = split["train"]
    mean, std, files_used = ds.compute_training_stats(train_files, channels)
    print(f"  Training files used for normalization stats: {files_used}")

    Xtr, ytr, ctr = ds.collect_split(
        train_files, channels, mean, std,
        max_windows=ds.MAX_WINDOWS_TRAIN, split_name="train",
    )
    Xva, yva, cva = ds.collect_split(
        split["val"], channels, mean, std,
        max_windows=ds.MAX_WINDOWS_VAL, split_name="val",
    )
    Xte, yte, cte = ds.collect_split(
        split["test"], channels, mean, std,
        max_windows=ds.MAX_WINDOWS_TEST, split_name="test",
    )
    return (Xtr, ytr, ctr), (Xva, yva, cva), (Xte, yte, cte)


def run_rf_configuration(
    config_name: str,
    num_channels: int,
    channels: list[str],
    split: dict,
    n_estimators: int = RF_N_ESTIMATORS,
) -> dict:
    """Train + evaluate one montage with a Random Forest on real windows."""
    print()
    print(f"BASELINE CONFIGURATION {config_name} ({num_channels} channels)")

    (Xtr, ytr, ctr), (Xva, yva, cva), (Xte, yte, cte) = build_config_files(
        split, channels
    )
    del Xva, yva, cva
    gc.collect()

    if Xtr is None or ytr is None or Xte is None or yte is None:
        raise RuntimeError(
            f"Configuration {config_name} produced no labeled windows; "
            "cannot run the baseline."
        )

    timesteps = Xtr.shape[1]
    ds.validate_model_inputs(Xtr, ytr, num_channels, timesteps)
    ds.validate_model_inputs(Xte, yte, num_channels, timesteps)

    feat_tr = window_summary_features(Xtr)
    feat_te = window_summary_features(Xte)
    print(f"  train features: {feat_tr.shape}  test features: {feat_te.shape}")

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=RF_MAX_DEPTH,
        class_weight=RF_CLASS_WEIGHT,
        random_state=ds.RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(feat_tr, ytr)
    probs = np.asarray(clf.predict_proba(feat_te)[:, 1])
    y_pred = (probs >= RF_THRESHOLD).astype(int)

    metrics = ev.compute_metrics(yte, y_pred, y_pred_scores=probs,
                                 threshold=RF_THRESHOLD)
    print(f"  Test samples: {metrics['test_samples']}  "
          f"positive: {metrics['positive_samples']}  "
          f"negative: {metrics['negative_samples']}")
    print(f"  Accuracy: {metrics['accuracy']}  Precision: {metrics['precision']}  "
          f"Recall: {metrics['recall']}  Specificity: {metrics['specificity']}  "
          f"F1: {metrics['f1']}  ROC-AUC: {metrics['roc_auc']}")

    return {
        "configuration": config_name,
        "channels": num_channels,
        "model": "RandomForest",
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "test_samples": metrics.get("test_samples"),
        "note": ("RF on per-window summary statistics (mean/std/RMS/peak-peak "
                 "per channel), same split/normalization/windows as CNN-BiLSTM"),
    }


def _fmt(value, na="NA"):
    if value is None:
        return na
    return f"{float(value):.4f}"


def load_cnn_rows(comparison_path) -> list[dict]:
    """CNN-BiLSTM rows from Phase 6 comparison.csv (real values)."""
    comparison_path = Path(comparison_path)
    if not comparison_path.is_file():
        return []
    rows = []
    with open(comparison_path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append({
                "configuration": raw.get("configuration"),
                "channels": raw.get("channels"),
                "model": "CNN-BiLSTM",
                "accuracy": raw.get("accuracy"),
                "precision": raw.get("precision"),
                "recall": raw.get("recall"),
                "specificity": raw.get("specificity"),
                "f1": raw.get("f1"),
                "roc_auc": raw.get("roc_auc"),
                "test_samples": raw.get("test_samples"),
                "note": "Phase 6 result (as reported in comparison.csv)",
            })
    return rows


def save_baseline_comparison(rows: list[dict], path=None) -> Path:
    if path is None:
        path = config.METRICS_DIR / "baseline_comparison.csv"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSelect Phase 7 baseline")
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS,
                        help="Random Forest tree count (default 200)")
    args = parser.parse_args()
    n_estimators = args.n_estimators

    edf_files = ds.discover_dataset()
    if not edf_files:
        print("NO REAL EDF DATASET FOUND")
        print("-" * 30)
        print("The Random Forest baseline requires real EEG, the same data ")
        print("the CNN-BiLSTM experiment needs. Place EDF/EDF+GZ files under:")
        print(f"  {config.RAW_DATA_DIR}")
        print("Then re-run scripts/select_electrodes.py (Phase 4) and")
        print("scripts/run_experiment.py (Phase 6) before this baseline.")
        return 1

    sel = ds.load_selected_channels()
    if sel is None:
        print_blocker(
            "RELIABLE CHANNEL SELECTION (results/metrics/selected_channels.json) "
            "NOT FOUND. The baseline must use the SAME top-5/top-4 montages as "
            "the CNN-BiLSTM experiment; run Phase 4 first."
        )
        return 3

    print("PHASE 7 BASELINE - Random Forest vs CNN-BiLSTM (22 vs 5 vs 4)")
    print("-" * 60)

    compat, dropped = ds.filter_files_with_channels(
        edf_files, sel["top_5"] + sel["top_4"]
    )
    print(f"Compatible recordings (usable by all configs): {len(compat)}")
    if dropped:
        print(f"  Recordings dropped (missing reduced-montage electrodes): "
              f"{[Path(f).name for f in dropped]}")

    train_files, val_files, test_files, split_desc = ds.split_recordings(
        compat, train_fraction=0.7, val_fraction=0.15
    )
    split = {"train": train_files, "val": val_files, "test": test_files,
             "description": split_desc}
    for name, files in (("train", train_files), ("val", val_files),
                        ("test", test_files)):
        if not files:
            print_blocker(
                f"Split '{name}' has no recordings after subject-aware "
                f"splitting; the dataset is too small for a 3-way split "
                "({split_desc}). No baseline is run."
            )
            return 2

    common22 = ds.common_channels_across(compat)
    print(f"22-channel montage (common across compatible recordings): "
          f"{len(common22)} channels")
    if len(common22) < 4:
        print_blocker(f"Only {len(common22)} common channels; baseline cannot run.")
        return 2

    montages = [("22", common22), ("5", sel["top_5"]), ("4", sel["top_4"])]
    rows = []
    for config_id, channels in montages:
        if len(channels) < 2:
            print_blocker(
                f"Configuration {config_id} has only {len(channels)} channels; "
                "cannot train. Stop."
            )
            return 2
        rows.append(run_rf_configuration(config_id, len(channels), channels,
                                         split, n_estimators=n_estimators))
        gc.collect()

    cnn_rows = load_cnn_rows(config.METRICS_DIR / "comparison.csv")
    if cnn_rows:
        rows = cnn_rows + rows
        status_note = "CNN-BiLSTM rows added from Phase 6 comparison.csv"
    else:
        status_note = ("CNN-BiLSTM rows omitted: results/metrics/comparison.csv "
                       "does not exist (Phase 6 never ran)")
    print()
    print(f"Note: {status_note}")

    path = save_baseline_comparison(rows)
    print(f"Baseline comparison saved to: {path}")

    print()
    print("BASELINE SUMMARY (test set, threshold=0.5)")
    for row in rows:
        print(f"  {row['configuration']:>2} ch [{row['model']:>11}]: "
              f"acc={row['accuracy']} prec={row['precision']} "
              f"rec={row['recall']} spec={row['specificity']} "
              f"f1={row['f1']} auc={row['roc_auc']} n={row['test_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())