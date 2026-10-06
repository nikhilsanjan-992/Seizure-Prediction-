"""NeuroSelect Phase 6 experiment runner (REAL EEG only).

Leakage-safe, memory-light workflow:
    real EDFs -> subject/session-aware split -> TRAIN/VAL/TEST
    -> training-only normalization stats -> segmentation -> labeling
    -> 22 / 5 / 4 montages (same CNN-BiLSTM) -> train -> evaluate on the
       untouched TEST set -> results/metrics + results/plots.

Hard gates (nothing is written until a gate passes):
    1. REAL EDF files must exist under data/raw/          (else exit 1)
    2. Phase 4 selection must exist: results/metrics/
       selected_channels.json with top_5 / top_4          (else exit 3)
    3. Every split must produce enough labeled windows
       with both classes (else exit 2, exact stage printed)

Usage:
    python scripts/run_experiment.py [--epochs 10] [--batch-size 8]
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import dataset as ds  # noqa: E402
import evaluate as ev  # noqa: E402
import train as tr  # noqa: E402

FIRST_RUN_EPOCHS = 10
FIRST_RUN_BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 3


def release_memory() -> None:
    """Free Keras session + Python heap between configurations."""
    import tensorflow as tf

    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()


def print_blocker(message: str) -> None:
    print("BLOCKED")
    print("-" * 40)
    print(message)


def print_split_sizes(train, val, test, desc: str) -> None:
    print(f"Split method: {desc}")
    print(f"Training recordings: {len(train)}")
    print(f"Validation recordings: {len(val)}")
    print(f"Test recordings: {len(test)}")


def check_split_labels(split: dict) -> None:
    for name, files in (("train", split["train"]),
                        ("val", split["val"]),
                        ("test", split["test"])):
        if not files:
            print_blocker(
                f"Split '{name}' has no recordings after subject-aware "
                "splitting; the dataset is too small for a 3-way subject split. "
                "No experiment is run."
            )
            sys.exit(2)


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


def validate_built_split(X, y, counts, name: str, num_channels: int) -> int:
    if X is None or y is None:
        print_blocker(
            f"Split '{name}' produced no labeled windows "
            f"(counts={counts}). Stopping; inspect annotations and the "
            "Phase 4 selection before adjusting."
        )
        sys.exit(2)
    timesteps = X.shape[1]
    ds.validate_model_inputs(X, y, num_channels, timesteps)
    print(f"    {name}: shape={tuple(X.shape)}  "
          f"seizure={counts['seizure']}  background={counts['background']}  "
          f"unlabeled={counts.get('unlabeled', 0)}")
    return timesteps


def run_configuration(
    config_name: str,
    num_channels: int,
    channels: list[str],
    split: dict,
    args,
) -> dict:
    print()
    print("=" * 60)
    print(f"CONFIGURATION {config_name} ({num_channels} channels)")
    print(f"Channel list: {', '.join(channels)}")
    print("=" * 60)

    (Xtr, ytr, ctr), (Xva, yva, cva), (Xte, yte, cte) = build_config_files(
        split, channels
    )
    t_tr = validate_built_split(Xtr, ytr, ctr, "train", num_channels)
    t_va = validate_built_split(Xva, yva, cva, "val", num_channels)
    t_te = validate_built_split(Xte, yte, cte, "test", num_channels)
    if not (t_tr == t_va == t_te):
        print_blocker(
            f"Mismatched timesteps across splits "
            f"(train={t_tr}, val={t_va}, test={t_te}). Sampling-rate mismatch "
            "or inconsistent windowing; stop."
        )
        sys.exit(2)

    print(f"  Training class distribution: "
          f"{int((ytr == 1).sum())} seizure / {int((ytr == 0).sum())} background")
    print("  Computing class weights from TRAINING labels only ...")
    weights = tr.compute_class_weights(ytr)
    print(f"    class weights: {weights}")

    print(f"  Training with epochs={args.epochs}, batch_size={args.batch_size}, "
          f"patience={EARLY_STOP_PATIENCE} ...")
    model, history = tr.train_cnn_bilstm(
        Xtr, ytr, Xva, yva,
        num_channels=num_channels,
        timesteps=t_tr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=EARLY_STOP_PATIENCE,
        model_name=f"cnn_bilstm_{num_channels}",
    )
    del Xtr, ytr
    gc.collect()

    saved_path = config.MODEL_DIR / f"cnn_bilstm_{num_channels}.keras"
    print(f"  Evaluating UNTOUCHED test set with best saved model "
          f"({saved_path.name}) ...")

    import tensorflow as tf

    best = tf.keras.models.load_model(saved_path)
    metrics = ev.evaluate_model(best, Xte, yte)
    del Xva, yva, Xte, yte, model, best
    release_memory()

    title = f"CNN-BiLSTM {num_channels} channels - test set"
    cm_path = ev.plot_confusion_matrix(
        metrics, title, config.PLOTS_DIR / f"confusion_matrix_{num_channels}.png"
    )
    curve_path = ev.plot_training_curve(
        history, config.PLOTS_DIR / f"training_curve_{num_channels}.png",
        title=title,
    )
    print(f"  Test samples: {metrics['test_samples']}  "
          f"positive: {metrics['positive_samples']}  "
          f"negative: {metrics['negative_samples']}")
    print(f"  Accuracy: {metrics['accuracy']}  Precision: {metrics['precision']}  "
          f"Recall: {metrics['recall']}  Specificity: {metrics['specificity']}  "
          f"F1: {metrics['f1']}  ROC-AUC: {metrics['roc_auc']}")

    return {
        "configuration": config_name,
        "channels": num_channels,
        **{k: metrics.get(k) for k in (
            "accuracy", "precision", "recall", "specificity", "f1", "roc_auc",
            "test_samples", "positive_samples", "negative_samples",
        )},
        "confusion_matrix": metrics["confusion_matrix"],
        "channel_list": channels,
        "plots": {"confusion": str(cm_path), "training_curve": str(curve_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSelect Phase 6 experiment")
    parser.add_argument("--epochs", type=int, default=FIRST_RUN_EPOCHS,
                        help="max epochs (default 10 per Section 8)")
    parser.add_argument("--batch-size", type=int, default=FIRST_RUN_BATCH_SIZE,
                        help="batch size (default 8)")
    args = parser.parse_args()

    edf_files = ds.discover_dataset()
    if not edf_files:
        print("NO REAL EDF DATASET FOUND")
        print("-" * 30)
        print("Phase 6 requires real EEG. Place EDF/EDF+GZ files inside:")
        print(f"  {config.RAW_DATA_DIR}")
        print("Then re-run scripts/select_electrodes.py (Phase 4) before this.")
        return 1

    sel = ds.load_selected_channels()
    if sel is None:
        print_blocker(
            "RELIABLE CHANNEL SELECTION (results/metrics/selected_channels.json) "
            "NOT FOUND. Top-5/top-4 channels must come from the real Phase 4 run; "
            "an experiment without them would invent a montage. "
            "Run scripts/select_electrodes.py after placing real EDF data."
        )
        return 3

    print("PHASE 6 EXPERIMENT - 22 vs 5 vs 4 channels (CNN-BiLSTM)")
    print("-" * 56)

    # Fair comparison: recordings usable by EVERY configuration.
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
    print_split_sizes(train_files, val_files, test_files, split_desc)
    check_split_labels(split)

    common22 = ds.common_channels_across(compat)
    print(f"22-channel montage (common across compatible recordings): "
          f"{len(common22)} channels")
    if len(common22) < 4:
        print_blocker(f"Only {len(common22)} common channels; experiment "
                      "cannot run. Inspect the dataset montage.")
        return 2

    montages = [
        ("22", common22),
        ("5", sel["top_5"]),
        ("4", sel["top_4"]),
    ]

    rows = []
    error_analysis = {}

    for config_id, channels in montages:
        if len(channels) < 2:
            print_blocker(
                f"Configuration {config_id} has only {len(channels)} "
                "channels; cannot train. Stop."
            )
            return 2
        row = run_configuration(config_id, len(channels), channels, split, args)
        rows.append(row)
        cm = row["confusion_matrix"]
        error_analysis[config_id] = {
            "configuration": config_id,
            "channels": row["channels"],
            "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
            "note": "Observed classifier errors on the untouched test set; "
                    "no clinical interpretation is claimed.",
        }
        del row
        gc.collect()

    # ---- Aggregate real outputs -------------------------------------------------
    comparison_path = ev.save_comparison_csv(rows)
    error_path = ev.save_error_analysis(error_analysis)
    comparison_plot = ev.plot_channel_comparison(
        rows, config.PLOTS_DIR / "channel_comparison.png"
    )

    montage_payload = {
        "22": {"channels": list(montages[0][1]),
               "source": "structural common montage across compatible recordings"},
        "5": {"channels": list(montages[1][1]),
              "source": "Phase 4 selected_channels.json top_5 (training data only)"},
        "4": {"channels": list(montages[2][1]),
              "source": "Phase 4 selected_channels.json top_4 (training data only)"},
    }
    payload = ds.build_experiment_config(
        dataset_files=list(compat),
        split=split,
        montages=montage_payload,
        preprocessing={
            "low_freq_hz": ds.LOW_FREQ,
            "high_freq_hz": ds.HIGH_FREQ,
            "notch_hz": ds.NOTCH_FREQ,
            "normalization": "per-channel z-score (training statistics only)",
            "window_seconds": ds.WINDOW_SECONDS,
            "overlap_seconds": ds.OVERLAP_SECONDS,
        },
        training={
            "optimizer": "Adam",
            "learning_rate": tr.LEARNING_RATE,
            "epochs_limit": args.epochs,
            "batch_size": args.batch_size,
            "loss": "binary_crossentropy",
            "callbacks": f"EarlyStopping(patience={EARLY_STOP_PATIENCE}, "
                         "restore_best_weights=True) + ModelCheckpoint(best)",
            "class_weights": "balanced, computed from training labels only",
            "seed": ds.RANDOM_SEED,
        },
        threshold=ev.DEFAULT_THRESHOLD,
        seed=ds.RANDOM_SEED,
    )
    experiment_config_path = ds.save_experiment_config(payload)

    print()
    print("=" * 60)
    print("RESULTS RECORDED")
    print(f"  comparison.csv        : {comparison_path}")
    print(f"  error_analysis.json   : {error_path}")
    print(f"  experiment_config.json: {experiment_config_path}")
    print(f"  channel_comparison.png: {comparison_plot}")

    print()
    print("EXPERIMENT SUMMARY (test set, threshold=0.5)")
    for row in rows:
        print(f"  {row['configuration']:>2} ch: acc={row['accuracy']} "
              f"prec={row['precision']} rec={row['recall']} "
              f"spec={row['specificity']} f1={row['f1']} "
              f"auc={row['roc_auc']} n={row['test_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())