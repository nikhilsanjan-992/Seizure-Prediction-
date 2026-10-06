"""NeuroSelect Phase 7 validation utilities.

Reusable, dependency-light helpers for auditing a completed Phase 6
experiment and its stored artifacts under ``results/metrics/``:

    * missing / non-finite values
    * class distribution
    * duplicate samples where detectable
    * split overlap (recording / subject / segment)
    * metric consistency (recomputed from the stored confusion matrix)
    * channel / configuration consistency across the 22 / 5 / 4 montages
    * leakage-audit assembly

Each function returns plain dicts so callers can persist them. Nothing in
this module writes result files and nothing invents values: when the required
information is not available the helpers report ``NOT_VERIFIABLE`` instead of
guessing a number.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from config import METRICS_DIR  # noqa: F401
except ImportError:  # pragma: no cover - fallback when src/ is on sys.path
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import METRICS_DIR  # noqa: F401

NOT_VERIFIABLE = "NOT_VERIFIABLE"
NA = "NA"

_METRIC_COLUMNS = (
    "accuracy", "precision", "recall", "specificity", "f1", "roc_auc",
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_json(path) -> dict | list | None:
    """Load a JSON artifact; None when absent/unreadable (nothing guessed)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _parse_float(value):
    """Parse a metric cell: '' / 'NA' / None -> None; else float or None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in ("NA", "NAN", "NONE", "NULL"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_comparison_csv(path):
    """comparison.csv -> list of rows with metric cells parsed as float."""
    path = Path(path)
    if not path.is_file():
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for key in _METRIC_COLUMNS:
                row[key] = _parse_float(row.get(key))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Metrics (binary classification, safe zero denominators)
# ---------------------------------------------------------------------------
def safe_divide(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator or None for a zero denominator."""
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator <= 0:
        return None
    return numerator / denominator


def metrics_from_confusion_matrix(tn, fp, fn, tp) -> dict:
    """Compute accuracy/precision/recall/specificity/F1 from CM counts.

    Identical formulas to :func:`evaluate.compute_metrics`. Undefined
    denominators -> None (never a fabricated value).

        accuracy    = (TP + TN) / (TP + TN + FP + FN)
        precision   = TP / (TP + FP)
        recall      = TP / (TP + FN)
        specificity = TN / (TN + FP)
        f1          = 2 * TP / (2 * TP + FP + FN)
    """
    t = {
        "accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }
    t["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return t


def verify_metric_consistency(
    cm_counts: dict,
    reported: dict,
    tolerance: float = 0.001,
) -> dict:
    """Recompute metrics from a confusion matrix and compare to reported.

    ``cm_counts`` contains TN/FP/FN/TP. ``reported`` may hold the metric
    values from comparison.csv (already parsed to float). Returns per-metric
    pass/FAIL/NOT_VERIFIABLE findings plus recomputed values and deltas.
    ROC-AUC cannot be recomputed from a confusion matrix alone (it needs the
    predicted scores) and is reported as NOT_VERIFIABLE for that reason.
    """
    recomputed = metrics_from_confusion_matrix(
        cm_counts.get("tn", 0),
        cm_counts.get("fp", 0),
        cm_counts.get("fn", 0),
        cm_counts.get("tp", 0),
    )
    findings = {}
    for key in ("accuracy", "precision", "recall", "specificity", "f1"):
        actual = recomputed.get(key)
        reported_value = reported.get(key) if isinstance(reported, dict) else None
        if actual is None or reported_value is None:
            findings[key] = {
                "status": "NOT_VERIFIABLE",
                "reason": "reported value missing or denominator undefined",
            }
            continue
        delta = abs(actual - float(reported_value))
        findings[key] = {
            "status": "PASS" if delta <= tolerance else "FAIL",
            "recomputed": float(actual),
            "reported": float(reported_value),
            "abs_delta": float(delta),
            "tolerance": tolerance,
        }
    findings["roc_auc"] = {
        "status": "NOT_VERIFIABLE",
        "reason": (
            "ROC-AUC requires stored predicted scores; Phase 6 does not persist "
            "per-sample predictions"
        ),
    }
    return {"recomputed": recomputed, "findings": findings}


def check_missing_values(table: list[dict], columns) -> dict:
    """Count missing / NA cells per column in a list of row dicts."""
    cell = {}
    for col in columns:
        missing = 0
        for row in table:
            val = row.get(col)
            if val is None:
                missing += 1
            elif isinstance(val, (int, float)):
                if not np.isfinite(float(val)):
                    missing += 1
            elif str(val).strip().upper() in ("", "NA", "NAN", "NONE", "NULL"):
                missing += 1
        cell[col] = {"missing": missing, "total": len(table)}
    return {
        "any_missing": any(cell[c]["missing"] for c in cell),
        "columns": cell,
    }


def check_non_finite(values) -> dict:
    """Report non-finite entries in a numeric sequence (array-safe)."""
    values = np.asarray(values, dtype=np.float64)
    bad = np.logical_not(np.isfinite(values))
    count = int(bad.sum())
    indices = np.nonzero(bad)[0][:10].tolist() if count else []
    return {"non_finite_count": count, "sample_indices": indices}


def check_class_distribution(positive: int, negative: int, total: int) -> dict:
    """Class balance check. None counts -> NOT_VERIFIABLE."""
    if positive is None or negative is None or total is None:
        return {
            "status": "NOT_VERIFIABLE",
            "reason": "sample counts not stored",
        }
    p, n, t = int(positive), int(negative), int(total)
    if p + n != t:
        return {
            "status": "INCONSISTENT",
            "positive": p, "negative": n, "total": t,
            "reason": "positive + negative != test_samples",
        }
    return {
        "status": "PASS",
        "positive": p, "negative": n, "total": t,
        "class_imbalance_ratio": None if min(p, n) == 0
        else float(max(p, n) / min(p, n)),
    }


# ---------------------------------------------------------------------------
# Duplicate / overlap checks (recording, subject, segment)
# ---------------------------------------------------------------------------
def check_duplicate_items(items) -> dict:
    """Duplicate detection over an iterable of hashable sample identifiers."""
    items = list(items)
    seen = {}
    dup_ids = {}
    for it in items:
        key = it if isinstance(it, (str, int)) else str(it)
        seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > 1:
            dup_ids[key] = count
    if dup_ids:
        return {"status": "FAIL", "duplicate_count": int(sum(dup_ids.values())
                                                          - len(dup_ids)),
                "duplicates": dup_ids}
    return {"status": "PASS", "duplicate_count": 0}


def check_split_overlap(train: list, val: list, test: list) -> dict:
    """Overlap between official split recording lists (path-level)."""
    train, val, test = [list(map(str, x)) for x in (train, val, test)]
    s_train, s_val, s_test = set(train), set(val), set(test)
    return {
        "train_val_overlap": sorted(s_train & s_val),
        "train_test_overlap": sorted(s_train & s_test),
        "val_test_overlap": sorted(s_val & s_test),
        "train_val_overlap_count": int(len(s_train & s_val)),
        "train_test_overlap_count": int(len(s_train & s_test)),
        "val_test_overlap_count": int(len(s_val & s_test)),
        "status": "PASS" if not (s_train & s_val or s_train & s_test
                                 or s_val & s_test) else "FAIL",
    }


def check_subject_overlap(train: list, val: list, test: list,
                          infer_subject_id) -> dict:
    """Subject-level overlap using subject ids inferred from file names.

    ``infer_subject_id`` mirrors electrode_selection.infer_subject_id. When
    an id cannot be inferred for any file of a split, subject independence is
    reported as NOT_VERIFIABLE rather than assumed.
    """
    def _ids(paths):
        ids = [infer_subject_id(p) for p in paths]
        if any(i is None for i in ids):
            return None
        return set(ids)

    s_train, s_val, s_test = _ids(train), _ids(val), _ids(test)
    if s_train is None or s_val is None or s_test is None:
        return {
            "status": "NOT_VERIFIABLE",
            "reason": ("subject id not inferable from every split file name; "
                       "subject independence cannot be verified from artifacts"),
        }
    overlaps = {
        "train_val_subject_overlap": sorted(s_train & s_val),
        "train_test_subject_overlap": sorted(s_train & s_test),
        "val_test_subject_overlap": sorted(s_val & s_test),
    }
    any_overlap = any(overlaps.values())
    return {
        **overlaps,
        "status": "FAIL" if any_overlap else "PASS",
        "n_train_subjects": len(s_train),
        "n_val_subjects": len(s_val),
        "n_test_subjects": len(s_test),
    }


def check_segment_overlap(arrays_available: bool = False) -> dict:
    """Segment-level overlap. Windows are not persisted by Phase 6, so this
    is only verifiable when the actual segment arrays are supplied."""
    if not arrays_available:
        return {
            "status": NOT_VERIFIABLE,
            "reason": ("per-window segment arrays are not persisted; overlapping "
                       "segments between splits cannot be detected from artifacts"),
        }
    return {"status": "NOT_RUN"}


# ---------------------------------------------------------------------------
# Channel / configuration consistency
# ---------------------------------------------------------------------------
def check_channel_consistency(
    experiment_config: dict | None,
    selected_channels: dict | None,
    rows: list[dict],
    expected_montages=(("22", 22), ("5", 5), ("4", 4)),
) -> dict:
    """Verify montage channels match Phase 4 selection and comparison rows.

    Returns a dict of PASS / FAIL / NOT_VERIFIABLE per expected montage plus
    the actual channel lists found in the experiment configuration.
    """
    findings = {}
    if experiment_config is None:
        for cid, n in expected_montages:
            findings[cid] = {
                "status": NOT_VERIFIABLE,
                "reason": "experiment_config.json not available",
            }
        return {"findings": findings, "top5_check": NOT_VERIFIABLE,
                "top4_check": NOT_VERIFIABLE}

    montages = experiment_config.get("montages", {})
    channel_counts = {}
    for cid, _n in expected_montages:
        channels = montages.get(cid, {}).get("channels")
        if not isinstance(channels, list):
            findings[cid] = {
                "status": NOT_VERIFIABLE,
                "reason": f"montage '{cid}' channel list missing in config",
            }
        else:
            channel_counts[cid] = len(channels)
            findings[cid] = {
                "status": "PASS",
                "channel_count": len(channels),
                "channels": list(channels),
                "source": montages.get(cid, {}).get("source"),
            }

    top5 = [str(c) for c in (selected_channels or {}).get("top_5", [])]
    top4 = [str(c) for c in (selected_channels or {}).get("top_4", [])]
    top5_check = "PASS" if montages.get("5", {}).get("channels") == top5 \
        else (NOT_VERIFIABLE if not top5 else "FAIL")
    top4_check = "PASS" if montages.get("4", {}).get("channels") == top4 \
        else (NOT_VERIFIABLE if not top4 else "FAIL")

    rows_by_config = {}
    if rows is not None:
        for row in rows:
            rows_by_config[str(row.get("configuration", "")).strip()] = row
    for cid, n in expected_montages:
        row = rows_by_config.get(cid)
        if row is None:
            continue
        reported_n = row.get("channels")
        expected = channel_counts.get(cid)
        if reported_n is None or expected is None:
            continue
        if int(reported_n) != int(expected):
            findings[cid]["status"] = "FAIL"
            findings[cid]["reason"] = (
                f"comparison.csv channels={reported_n} != config montage "
                f"channels={expected}"
            )
    return {"findings": findings, "top5_check": top5_check,
            "top4_check": top4_check}


def check_config_consistency(experiment_config: dict | None) -> dict:
    """Sanity checks on the stored experiment configuration itself."""
    if experiment_config is None:
        return {"status": NOT_VERIFIABLE, "reason": "experiment_config.json missing"}
    keys = ("dataset", "montages", "preprocessing", "training",
            "classification_threshold", "random_seed", "architecture")
    missing = [k for k in keys if k not in experiment_config]
    return {
        "status": "PASS" if not missing else "FAIL",
        "missing_keys": missing,
        "random_seed": experiment_config.get("random_seed"),
        "threshold": experiment_config.get("classification_threshold"),
        "split_method": experiment_config.get("dataset", {}).get("split_method"),
    }


# ---------------------------------------------------------------------------
# Leakage audit assembly
# ---------------------------------------------------------------------------
def build_leakage_audit(
    split_recordings: dict | None,
    subject_overlap: dict,
    recording_overlap: dict,
    segment_overlap: dict,
    electrode_selection: dict | None,
    normalization: dict | None,
    threshold: dict | None,
    class_weight: dict | None,
) -> dict:
    """Assemble the leakage_audit.json payload from individual check dicts.

    Individual checks already carry PASS / FAIL / NOT_VERIFIABLE semantics;
    this only flattens them into the documented structure so a single file
    records the whole audit.
    """
    return {
        "train_test_subject_overlap": subject_overlap,
        "train_test_recording_overlap": recording_overlap,
        "segment_overlap": segment_overlap,
        "electrode_selection_test_leakage": electrode_selection,
        "normalization_test_leakage": normalization,
        "threshold_test_leakage": threshold,
        "class_weight_leakage": class_weight,
        "split_recordings_available": split_recordings,
    }


def blocked_leakage_audit(reason: str, artifacts: dict | None = None) -> dict:
    """Leakage audit when no Phase 6 artifacts exist (nothing verifiable)."""
    item = {"status": NOT_VERIFIABLE, "reason": reason}
    payload = {
        "status": "BLOCKED",
        "reason": reason,
        "artifacts_found": artifacts or {},
        "train_test_subject_overlap": dict(item),
        "train_test_recording_overlap": dict(item),
        "subject_overlap": dict(item),
        "recording_session_overlap": dict(item),
        "segment_overlap": dict(item),
        "electrode_selection_test_leakage": dict(item),
        "normalization_test_leakage": dict(item),
        "threshold_test_leakage": dict(item),
        "class_weight_leakage": dict(item),
    }
    return payload