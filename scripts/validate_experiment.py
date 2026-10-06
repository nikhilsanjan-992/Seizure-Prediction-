"""NeuroSelect Phase 7 experiment validation runner.

Validates a completed Phase 6 experiment against its stored artifacts
without re-training anything. Only real values are reported; when the
required artifacts do not exist the audit records BLOCKED / NOT_VERIFIABLE
and metric-dependent outputs are NOT written (nothing is fabricated).

Checks performed when Phase 6 results exist:
    * leakage audit (A-H) assembled from experiment_config.json + source audit
      -> results/metrics/leakage_audit.json
    * metric validation recomputed from the stored confusion matrices
      -> results/metrics/validated_metrics.csv
    * detailed error analysis (FP / FN / class-specific errors / error rate)
      -> results/metrics/error_analysis_detailed.json
    * research table + reproducibility record + research summary markdown

Exit codes:
    0  validation completed and every check that could run passed
    2  validation completed with one or more FAIL / inconsistent findings
    1  validation blocked: no Phase 6 results to audit (or no dataset)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # noqa: E402
import validation as v  # noqa: E402
import electrode_selection as es  # noqa: E402


# ---------------------------------------------------------------------------
# Small reusable writers (honest-content only)
# ---------------------------------------------------------------------------
def _dump_json(payload, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def _dump_csv(rows, columns, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def _fmt(value, na="NA"):
    if value is None:
        return na
    return f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Blocked path (no Phase 6 results): the real current state
# ---------------------------------------------------------------------------
def _blocked(missing: list[str]) -> int:
    print("PHASE 7 VALIDATION - BLOCKED")
    print("-" * 40)
    print("No Phase 6 experiment artifacts were found.")
    missing_files = [f"  results/metrics/{m}" for m in missing]
    print("\n".join(missing_files))
    print()
    print("The Phase 6 experiment (scripts/run_experiment.py) has never run:")
    print("  - no EDF dataset under data/raw/")
    print("  - no Phase 4 selected_channels.json")
    print("  - therefore no trained models, comparison.csv or confusion")
    print("    matrices exist to validate.")
    print()
    print("Every auditable leakage item is recorded as NOT_VERIFIABLE.")
    print("No metric-dependent output file is written (nothing is fabricated).")
    print()
    print("To unblock: place real EDF files in data/raw/, then run")
    print("  python scripts/select_electrodes.py   (Phase 4)")
    print("  python scripts/run_experiment.py      (Phase 6)")
    print("  python scripts/validate_experiment.py (Phase 7)")
    return 1


# ---------------------------------------------------------------------------
# Environment record (real values only)
# ---------------------------------------------------------------------------
def _environment_record() -> dict:
    import platform
    import sys as _sys

    record = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        from importlib import metadata

        for pkg in ("mne", "numpy", "scipy", "pandas", "matplotlib",
                    "scikit-learn", "tensorflow"):
            try:
                record[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                record[pkg] = "NOT_INSTALLED"
    except Exception as exc:  # pragma: no cover - best-effort only
        record["package_versions_error"] = str(exc)
    return record


def _configured_reproducibility_record() -> dict:
    """Reproducibility payload from the project's configured settings.

    Values come from the real source configuration (src/*.py), not from any
    invented experiment. Runtime facts that never happened are recorded
    literally as NOT_AVAILABLE.
    """
    import evaluate as ev
    import train as tr
    from dataset import (LOW_FREQ, HIGH_FREQ, NOTCH_FREQ, WINDOW_SECONDS,
                         OVERLAP_SECONDS, RANDOM_SEED)
    from model import (DEFAULT_CNN_FILTERS, DEFAULT_CNN_KERNELS,
                       DEFAULT_LSTM_UNITS, DEFAULT_DROPOUT)

    return {
        "status": "NOT_RUN - NO EXPERIMENT RESULTS EXIST",
        "dataset": {
            "identifier_path": "NOT_AVAILABLE",
            "edf_files_found": 0,
            "raw_data_dir": str(config.RAW_DATA_DIR),
        },
        "random_seed": RANDOM_SEED,
        "preprocessing_settings": {
            "low_freq_hz": LOW_FREQ,
            "high_freq_hz": HIGH_FREQ,
            "notch_hz": NOTCH_FREQ,
            "normalization": "per-channel z-score (training statistics only)",
        },
        "segmentation_settings": {
            "window_seconds": WINDOW_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS,
        },
        "selected_electrodes": {
            "top_5": "NOT_AVAILABLE (Phase 4 never ran: no dataset)",
            "top_4": "NOT_AVAILABLE (Phase 4 never ran: no dataset)",
            "selection_method": es.METHOD_DESCRIPTION,
        },
        "model_architecture": (
            "shared CNN-BiLSTM: Conv1D(32,k5)->BN->ReLU->MaxPool->"
            "Conv1D(64,k3)->BN->ReLU->BiLSTM(32)->Dropout->Dense(1,sigmoid)"
        ),
        "architecture_hyperparameters": {
            "cnn_filters": list(DEFAULT_CNN_FILTERS),
            "cnn_kernels": list(DEFAULT_CNN_KERNELS),
            "lstm_units": DEFAULT_LSTM_UNITS,
            "dropout": DEFAULT_DROPOUT,
        },
        "optimizer": "Adam",
        "learning_rate": tr.LEARNING_RATE,
        "batch_size": tr.BATCH_SIZE,
        "epochs_limit": "NOT_AVAILABLE (experiment never trained; Phase 6 "
                        "first-run default limit is 10)",
        "classification_threshold": ev.DEFAULT_THRESHOLD,
        "training_status": "NOT_RUN",
        "software_environment": _environment_record(),
    }


# ---------------------------------------------------------------------------
# Complete path (Phase 6 results exist)
# ---------------------------------------------------------------------------
def _validate_present(comparison_path, error_path, config_path, sel_path,
                      model_dir, out) -> int:
    print("PHASE 7 VALIDATION")
    print("-" * 40)
    rows = v.load_comparison_csv(comparison_path)
    error = v.load_json(error_path)
    experiment_config = v.load_json(config_path)
    selected = v.load_json(sel_path)
    if rows is None or error is None or experiment_config is None:
        print("BLOCKED: required Phase 6 artifact(s) missing:",
              ", ".join(m for m, p in (
                  ("comparison.csv", comparison_path),
                  ("error_analysis.json", error_path),
                  ("experiment_config.json", config_path)) if not Path(p).is_file()))
        return 1

    metrics_missing = v.check_missing_values(rows, ("accuracy", "precision",
                                                    "recall", "specificity",
                                                    "f1", "roc_auc"))
    print(f"Missing/NA cells in comparison.csv: "
          f"{'YES' if metrics_missing['any_missing'] else 'no'}")

    cm_fail = 0
    validated_rows = []
    print("\nMetric re-validation (recomputed from stored confusion matrices, "
          "tolerance=0.001):")
    for row in rows:
        cid = str(row.get("configuration", "")).strip()
        cm = error.get(cid, {}) if isinstance(error, dict) else {}
        if not cm or any(k not in cm for k in ("tn", "fp", "fn", "tp")):
            print(f"  {cid:>2}) no confusion matrix -> NOT_VERIFIABLE")
            continue
        check = v.verify_metric_consistency(
            {"tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"]},
            {k: row.get(k) for k in ("accuracy", "precision", "recall",
                                     "specificity", "f1")},
        )
        statuses = [f["status"] for f in check["findings"].values()]
        cm_fail += statuses.count("FAIL")
        print(f"  {cid:>2}) " + ", ".join(
            f"{k}={f['reported'] if isinstance(f.get('reported'), float) else f['status']}"
            for k, f in check["findings"].items()
            if k != "roc_auc"))
        rec = check["recomputed"]
        validated_rows.append({
            "configuration": cid,
            "accuracy": _fmt(rec["accuracy"]),
            "precision": _fmt(rec["precision"]),
            "recall": _fmt(rec["recall"]),
            "specificity": _fmt(rec["specificity"]),
            "f1": _fmt(rec["f1"]),
            "roc_auc": "NOT_VERIFIABLE",
        })

    if validated_rows:
        _dump_csv(validated_rows,
                  ["configuration", "accuracy", "precision", "recall",
                   "specificity", "f1", "roc_auc"],
                  out / "validated_metrics.csv")
        print(f"  wrote results/metrics/validated_metrics.csv "
              f"({len(validated_rows)} rows)")

    detailed = {}
    for row in rows:
        cid = str(row.get("configuration", "")).strip()
        cm = error.get(cid, {}) if isinstance(error, dict) else {}
        if not cm:
            detailed[cid] = {"status": "NOT_VERIFIABLE",
                             "reason": "no confusion matrix stored"}
            continue
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        n = tn + fp + fn + tp
        err_rate = v.safe_divide(fp + fn, n)
        detailed[cid] = {
            "configuration": cid,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "total_errors": int(fp + fn),
            "error_rate": err_rate,
            "class_specific_errors": {
                "false_positives": {
                    "count": int(fp),
                    "class": "background predicted as seizure",
                },
                "false_negatives": {
                    "count": int(fn),
                    "class": "seizure predicted as background",
                },
            },
            "note": ("Observed classifier errors on the untouched test set; "
                     "no clinical interpretation is claimed."),
        }
    if detailed:
        _dump_json(detailed, out / "error_analysis_detailed.json")
        print("  wrote results/metrics/error_analysis_detailed.json")

    split = experiment_config.get("dataset", {}) if experiment_config else {}
    train_f = split.get("train_recordings", [])
    val_f = split.get("val_recordings", [])
    test_f = split.get("test_recordings", [])

    recording_overlap = v.check_split_overlap(train_f, val_f, test_f)
    subject_overlap = v.check_subject_overlap(train_f, val_f, test_f,
                                              es.infer_subject_id)
    segment_overlap = v.check_segment_overlap(arrays_available=False)

    print(f"\nSplit overlap (paths): "
          f"{recording_overlap['status']} "
          f"(train/test={recording_overlap['train_test_overlap_count']}, "
          f"val/test={recording_overlap['val_test_overlap_count']})")
    print(f"Subject overlap: {subject_overlap['status']}")
    print(f"Segment overlap: {segment_overlap['status']}")

    chan_check = v.check_channel_consistency(experiment_config, selected, rows)
    print("\nChannel consistency:")
    for cid, f in chan_check["findings"].items():
        extra = f.get("channel_count")
        print(f"  {cid:>2}) {f['status']}"
              f"{f' ({extra} channels)' if extra else ''}")
    print(f"  top_5 matches Phase 4 selection: {chan_check['top5_check']}")
    print(f"  top_4 matches Phase 4 selection: {chan_check['top4_check']}")

    config_check = v.check_config_consistency(experiment_config)
    print(f"\nConfig consistency: {config_check['status']} "
          f"(threshold={config_check.get('threshold')}, "
          f"seed={config_check.get('random_seed')})")

    class_checks = {}
    for row in rows:
        cid = str(row.get("configuration", "")).strip()
        class_checks[cid] = v.check_class_distribution(
            row.get("positive_samples"), row.get("negative_samples"),
            row.get("test_samples"))
        print(f"  class distribution {cid:>2}: {class_checks[cid]['status']} "
              f"(pos={class_checks[cid].get('positive')}, "
              f"neg={class_checks[cid].get('negative')})")

    leakage = v.build_leakage_audit(
        split_recordings={
            "available": all((train_f, val_f, test_f)),
            "n_train": len(train_f), "n_val": len(val_f), "n_test": len(test_f),
        },
        subject_overlap=subject_overlap,
        recording_overlap=recording_overlap,
        segment_overlap=segment_overlap,
        electrode_selection={
            "status": "PASS" if chan_check["top5_check"] == "PASS"
            and chan_check["top4_check"] == "PASS" else "NOT_VERIFIABLE",
            "basis": ("Phase 4 selection ranked electrodes from the training "
                      "split only (code audit); stored top_5/top_4 cross-checked "
                      "against the Phase 6 montage lists"),
        },
        normalization={
            "status": "PASS_BY_DESIGN",
            "basis": ("per-channel z-score statistics computed by "
                      "dataset.compute_training_stats() on the training "
                      "recordings only, then applied to validation/test "
                      "(code audit)"),
        },
        threshold={
            "status": "PASS_BY_DESIGN",
            "basis": ("classification threshold fixed at the constant "
                      "evaluate.DEFAULT_THRESHOLD = "
                      f"{experiment_config.get('classification_threshold')}; "
                      "never tuned on any split (code audit)"),
        },
        class_weight={
            "status": "PASS_BY_DESIGN",
            "basis": ("balanced class weights computed by "
                      "train.compute_class_weights() from training labels only "
                      "(code audit)"),
        },
    )
    _dump_json(leakage, out / "leakage_audit.json")
    print("\n  wrote results/metrics/leakage_audit.json")

    table = []
    for row in rows:
        cid = str(row.get("configuration", "")).strip()
        rec = next((r for r in validated_rows if r["configuration"] == cid), None)

        def pick(key, fallback):
            if rec is not None:
                return rec.get(key)
            return _fmt(row.get(fallback))

        table.append({
            "configuration": cid,
            "channels": row.get("channels"),
            "accuracy": pick("accuracy", "accuracy"),
            "precision": pick("precision", "precision"),
            "recall": pick("recall", "recall"),
            "specificity": pick("specificity", "specificity"),
            "f1": pick("f1", "f1"),
            "roc_auc": _fmt(row.get("roc_auc")),
            "test_samples": row.get("test_samples"),
        })
    if table:
        _dump_csv(table,
                  ["configuration", "channels", "accuracy", "precision",
                   "recall", "specificity", "f1", "roc_auc", "test_samples"],
                  out / "final_research_table.csv")
        print("  wrote results/metrics/final_research_table.csv")

    model_files = sorted(p.name for p in Path(model_dir).glob("*.keras"))
    print(f"\nModel checkpoints present: {model_files or 'none'}")

    ok = not cm_fail and recording_overlap["status"] == "PASS" \
        and subject_overlap["status"] == "PASS" \
        and channel_check_pass(chan_check) and config_check["status"] == "PASS"
    return 0 if ok else 2


def channel_check_pass(chan_check: dict) -> bool:
    return chan_check["top5_check"] == "PASS" and chan_check["top4_check"] == "PASS"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSelect Phase 7 validation")
    parser.add_argument("--metrics-dir", type=str, default=None,
                        help="metrics directory (default: results/metrics)")
    parser.add_argument("--model-dir", type=str, default=str(config.MODEL_DIR),
                        help="model directory (default: models/)")
    args = parser.parse_args()

    out = Path(args.metrics_dir) if args.metrics_dir else config.METRICS_DIR
    out.mkdir(parents=True, exist_ok=True)

    comparison_path = out / "comparison.csv"
    error_path = out / "error_analysis.json"
    config_path = out / "experiment_config.json"
    sel_path = out / "selected_channels.json"

    missing = [n for n, p in (
        ("comparison.csv", comparison_path),
        ("error_analysis.json", error_path),
        ("experiment_config.json", config_path),
        ("selected_channels.json", sel_path),
    ) if not Path(p).is_file()]

    reproducibility = _configured_reproducibility_record()

    if missing:
        reason = ("No Phase 6 experiment results exist: no EDF dataset in "
                  f"{config.RAW_DATA_DIR} and results/metrics is empty. The "
                  "Phase 6 experiment was never run, so nothing can be audited.")
        audit = v.blocked_leakage_audit(reason, artifacts={
            "comparison.csv": comparison_path.is_file(),
            "experiment_config.json": config_path.is_file(),
            "error_analysis.json": error_path.is_file(),
            "selected_channels.json": sel_path.is_file(),
            "models": any(Path(config.MODEL_DIR).glob("*.keras")),
        })
        _dump_json(audit, out / "leakage_audit.json")
        reproducibility["dataset"] = {
            "identifier_path": "NOT_AVAILABLE (no EDF files in data/raw/)",
            "edf_files_found": 0,
            "raw_data_dir": str(config.RAW_DATA_DIR),
        }
        _dump_json(reproducibility, out / "reproducibility.json")
        _write_blocked_research_summary(out)
        return _blocked([n for n, _ in [
            ("comparison.csv", comparison_path),
            ("error_analysis.json", error_path),
            ("experiment_config.json", config_path),
            ("selected_channels.json", sel_path),
        ]])

    return _validate_present(comparison_path, error_path, config_path,
                             sel_path, args.model_dir, out)


def _write_blocked_research_summary(out: Path) -> None:
    summary = """# NeuroSelect - Phase 7 Research Summary

## Status

BLOCKED. The Phase 6 experiment (22 vs 5 vs 4 channels, CNN-BiLSTM) has not
been run: no real EDF dataset is present under `data/raw/` and no Phase 6
output files exist in `results/metrics/`. This summary therefore contains no
experimental performance values - nothing has been invented.

## Dataset

No dataset was found. `data/raw/` is empty (0 EDF/EDF.gz files). The pipeline
requires real EEG recordings and refuses to run on synthetic data.

## Experimental Setup (as designed, from source)

- Split: subject/session-aware train/val/test (train 0.7 / val 0.15 / test
  0.15), deterministic (seed 42), never mixes one subject's recordings across
  splits; documented session-level fallback for < 3 subjects.
- Preprocessing: FIR band-pass 0.5-40 Hz, optional notch, per-channel z-score
  normalization with TRAINING-only statistics.
- Segmentation: sliding windows (10 s, 5 s overlap), full windows only;
  windows labeled from real EDF annotations, unlabeled windows excluded.
- Model: shared CNN-BiLSTM (Conv1D 32 -> BN -> ReLU -> MaxPool -> Conv1D 64 ->
  BN -> ReLU -> BiLSTM 32 -> Dropout -> Dense(1, sigmoid)).
- Training: Adam lr=1e-3, batch 8, binary cross-entropy, class weights from
  training labels only, threshold fixed at 0.5.

## Electrode Selection

Not run. The data-driven ANOVA-F ranking (Phase 4, training-data only) has no
dataset to run on, so no `selected_channels.json` and no top-5 / top-4 lists
exist. Top-5 / top-4 are therefore NOT_AVAILABLE.

## Main Results

NOT_AVAILABLE - no experiment was run. No accuracy/precision/recall/
specificity/F1/ROC-AUC values exist for any configuration.

## Baseline

NOT RUN. A Random Forest comparator is implemented in
`scripts/run_baseline.py` (same split, same windows, same training-only
normalization, same 22/5/4 montages; per-window mean/std/RMS/peak-to-peak
features). It cannot be run without the dataset and writes `baseline_comparison.csv`
only from real test results, so none has been written yet.

## Robustness

NOT ASSESSED. No repeated runs exist. Channel stability is NOT ASSESSED
(no electrode-selection run exists, and only one selection run is ever planned
by the pipeline).

## Limitations

- The project has no real EEG dataset, so no experiment results exist to
  validate. Phase 6 was never executed (`run_experiment.py` writes nothing
  without EDF files and Phase 4 selection).
- Even a completed Phase 6 run would not persist per-sample test predictions
  or scores, so ROC-AUC and paired statistical comparisons cannot be
  re-derived from artifacts; only confusion-matrix-derived metrics can be
  independently re-computed.

## Conclusion

No quantitative conclusion can be drawn because no experiment has been run.
The experiments indicate nothing yet; results will be reported only after the
real dataset is added and Phases 4 and 6 are executed.
"""
    path = out / "research_summary.md"
    path.write_text(summary, encoding="utf-8")
    print("  wrote results/metrics/research_summary.md (blocked-mode summary)")


if __name__ == "__main__":
    sys.exit(main())