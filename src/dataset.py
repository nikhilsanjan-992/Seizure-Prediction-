"""NeuroSelect Phase 6 dataset preparation (leakage-safe, memory-light).

Builds the three experimental configurations (22 / 5 / 4 channels) from REAL
EDF recordings without producing any fabricated data:

    1. subject/session-aware train/val/test split (never mixes windows of a
       recording across splits)
    2. montage per configuration:
          * 22 channels -> common EEG channels across TRAINING recordings
          *  5 channels -> Phase 4 selected_channels.json["top_5"]
          *  4 channels -> Phase 4 selected_channels.json["top_4"]
    3. per-channel normalization statistics computed on TRAINING records only
       and applied to validation/test
    4. sequential, memory-light segmentation + annotation-based labeling
    5. final layout (samples, timesteps, channels), float32, ready for the
       CNN-BiLSTM model (model.to_model_orientation contract)

Labels come only from the real EDF annotations. Windows that cannot be
confidently labeled are excluded - never guessed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

try:
    from config import METRICS_DIR, RAW_DATA_DIR  # noqa: F401
except ImportError:  # pragma: no cover - fallback when src/ is on sys.path
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import METRICS_DIR, RAW_DATA_DIR  # noqa: F401

import eeg_pipeline as ep  # noqa: E402
import electrode_selection as es  # noqa: E402
import preprocessing as pp  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants (documented; identical settings for 22/5/4)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Preprocessing (reuse Phase 3 defaults; MUST match Phase 4 selection run and
# the segmentation used at training so windows are directly comparable).
LOW_FREQ = pp.DEFAULT_LOW_FREQ
HIGH_FREQ = pp.DEFAULT_HIGH_FREQ
NOTCH_FREQ = pp.DEFAULT_NOTCH_FREQ
WINDOW_SECONDS = pp.DEFAULT_WINDOW_SECONDS
OVERLAP_SECONDS = pp.DEFAULT_OVERLAP_SECONDS

MAX_SECONDS_PER_RECORDING = 3600.0  # at most 1 h per recording
MAX_WINDOWS_PER_FILE = 1000         # stride-shrink very long recordings

# Memory guards: keep every split's window count bounded so the in-RAM
# float32 tensors stay small on ~7.9 GB usable machines.
MAX_WINDOWS_TRAIN = 2000
MAX_WINDOWS_VAL = 500
MAX_WINDOWS_TEST = 500

USE_UNANNOTATED_AS_BACKGROUND = True  # mirrors Phase 4 choice

# Arrays are stored as float32 for the model (half the memory of float64).
DTYPE = np.float32

EPS = 1e-12


# ---------------------------------------------------------------------------
# Discovery + split
# ---------------------------------------------------------------------------
def discover_dataset() -> list[Path]:
    """All EDF/EDF+GZ files under config.RAW_DATA_DIR (reuses Phase 2/3)."""
    return list(pp.discover_edf_files())


def split_recordings(
    edf_files,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
):
    """Deterministic subject/session-aware train/val/test split.

    Every recording of a subject stays in exactly one split. Subject groups
    are ordered by id and assigned by SUBJECT count (not recording count),
    guaranteeing a non-empty val/test whenever >= 3 subjects (or >= 3 files
    in the fallbacks) exist.

    With a single subject a subject-independent split is impossible; the
    code falls back to a session-level (file-level) split and documents the
    leakage limitation rather than pretending independence.
    """
    files = sorted(edf_files)
    total = len(files)
    if total == 0:
        return [], [], [], "empty dataset"

    ids = [es.infer_subject_id(f) for f in files]

    def _trim(n_train: int, n_val: int, avail: int):
        """Ensure train >= 1, val >= 1, test >= 1 whenever avail >= 3."""
        if avail >= 3:
            if n_train + n_val >= avail:
                n_val = max(1, avail - 1 - n_train)
            if n_train + n_val >= avail:
                n_train = max(1, avail - 2)
                n_val = 1
        return n_train, n_val

    if all(i is not None for i in ids):
        subjects = sorted(set(ids))
        n_subj = len(subjects)
        if n_subj >= 3:
            n_tr_s = max(1, int(round(n_subj * train_fraction)))
            n_va_s = max(1, int(round(n_subj * val_fraction)))
            n_tr_s, n_va_s = _trim(n_tr_s, n_va_s, n_subj)
            train, val, test = [], [], []
            for idx, subj in enumerate(subjects):
                subj_files = [f for f, i in zip(files, ids) if i == subj]
                if idx < n_tr_s:
                    train.extend(subj_files)
                elif idx < n_tr_s + n_va_s:
                    val.extend(subj_files)
                else:
                    test.extend(subj_files)
            return train, val, test, (
                "subject-level split (deterministic; every recording of a "
                "subject stays in exactly one split)"
            )
        # one or two subjects only: session-level fallback, documented
        n_tr_f = max(1, int(round(total * train_fraction)))
        n_va_f = max(1, int(round(total * val_fraction)))
        n_tr_f, n_va_f = _trim(n_tr_f, n_va_f, total)
        train = files[:n_tr_f]
        val = files[n_tr_f:n_tr_f + n_va_f]
        test = files[n_tr_f + n_va_f:]
        return train, val, test, (
            f"session-level split of the same {n_subj} subject(s); NOT "
            "subject-independent (subject leakage between splits is present "
            "and documented)"
        )

    n_tr = max(1, int(round(total * train_fraction)))
    n_va = max(1, int(round(total * val_fraction)))
    n_tr, n_va = _trim(n_tr, n_va, total)
    train = files[:n_tr]
    val = files[n_tr:n_tr + n_va]
    test = files[n_tr + n_va:]
    return train, val, test, (
        "file-level split (subject id not inferable from file names; subject "
        "independence cannot be guaranteed)"
    )


# ---------------------------------------------------------------------------
# Channel metadata (no data arrays involved)
# ---------------------------------------------------------------------------
def common_channels_across(files) -> list[str]:
    """EEG channel names present in EVERY listed recording.

    Metadata-only (never loads the data array). Used for the 22-channel
    configuration montage.
    """
    common: list[str] | None = None
    for f in files:
        raw = ep.load_edf(f, preload=False)
        try:
            names = pp.get_eeg_channel_names(raw)
        finally:
            raw.close()
        if common is None:
            common = list(names)
        else:
            keep = set(names)
            common = [c for c in common if c in keep]
        if not common:
            break
    return common or []


def filter_files_with_channels(files, channels: list[str]):
    """Keep recordings that contain every requested electrode (normalized match).

    Fairness guarantee: recordings missing any electrode of the reduced
    montage(s) cannot serve all three configurations, so they are dropped
    from the whole experiment. Returns ``(ok_files, dropped_files)``.
    """
    needed = {pp._normalize_channel_label(c) for c in channels}
    ok: list = []
    dropped: list = []
    for f in files:
        raw = ep.load_edf(f, preload=False)
        try:
            present = {
                pp._normalize_channel_label(name)
                for name in pp.get_eeg_channel_names(raw)
            }
        finally:
            raw.close()
        if needed.issubset(present):
            ok.append(f)
        else:
            dropped.append(f)
    return ok, dropped


def load_selected_channels(path=None):
    """Load Phase 4 ``selected_channels.json``.

    Returns ``({"top_5": [...], "top_4": [...]})`` or ``None`` when the file
    is absent/malformed (the experiment then must not invent channels).
    """
    if path is None:
        path = METRICS_DIR / "selected_channels.json"
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read selected channels file: %s", exc)
        return None
    top5 = data.get("top_5")
    top4 = data.get("top_4")
    if not isinstance(top5, list) or not isinstance(top4, list):
        return None
    return {"top_5": [str(c) for c in top5][:5],
            "top_4": [str(c) for c in top4][:4]}


# ---------------------------------------------------------------------------
# Per-recording preparation (sequential, memory-light)
# ---------------------------------------------------------------------------
def _close(raw=None, picked=None, filtered=None) -> None:
    for obj in (raw, picked, filtered):
        if obj is None:
            continue
        try:
            obj.close()
        except Exception:
            pass


def _prepare_recording(path, channels: list[str], extract_labels: bool = True):
    """Open one recording once; load, crop, select, filter, extract labels.

    Returns a dict ``{"data", "sfreq", "regions", "unmapped"}`` where
    ``data`` is float64 ``(n_channels, n_times)`` aligned with pattern
    ``channels``, or ``None`` when the recording is unusable (missing
    requested channels, no EEG, too short). All underlying raw objects are
    closed before returning the data array.
    """
    raw = ep.load_edf(path, preload=False)
    picked = None
    filtered = None
    try:
        duration = ep.get_duration(raw)
        if duration > MAX_SECONDS_PER_RECORDING:
            raw.crop(tmax=MAX_SECONDS_PER_RECORDING)

        available, missing = pp.match_channel_requests(raw, channels)
        if missing:
            logger.warning(
                "SKIP %s: missing requested channels %s",
                Path(path).name, missing,
            )
            return None
        picked, _ignored, _missed = pp.select_channels(raw, requested=channels)
        if picked is None or getattr(picked, "n_times", 0) == 0:
            return None
        if not getattr(picked, "preload", False):
            picked.load_data()

        filtered = pp.filter_eeg(
            picked, copy=False,
            low_freq=LOW_FREQ, high_freq=HIGH_FREQ, notch_freq=NOTCH_FREQ,
        )
        sfreq = float(ep.get_sampling_frequency(filtered))
        data = np.asarray(filtered.get_data(), dtype=np.float64)

        result = {"data": data, "sfreq": sfreq}
        if extract_labels:
            regions, unmapped = pp.label_event_regions(filtered)
            result["regions"] = regions
            result["unmapped"] = list(unmapped)
        return result
    finally:
        _close(raw, picked, filtered)


def compute_training_stats(files, channels: list[str]):
    """Per-channel mean/std over TRAINING recordings only (online, memory-light).

    Two-pass approach avoids storing any recording: first pass accumulates
    sum / sum-of-squares, the caller re-loads files to normalize and segment.
    Returns ``(mean, std, files_used)`` where ``mean``/``std`` are
    ``(len(channels),)`` float64 arrays. Raises when no file contributes.
    """
    n = len(channels)
    sums = np.zeros(n, dtype=np.float64)
    sumsq = np.zeros(n, dtype=np.float64)
    count = 0
    files_used = 0
    for f in files:
        info = _prepare_recording(f, channels, extract_labels=False)
        if info is None:
            continue
        data = info["data"]
        sums += data.sum(axis=1)
        sumsq += (data ** 2).sum(axis=1)
        count += data.shape[1]
        files_used += 1
    if files_used == 0 or count == 0:
        raise ValueError(
            "No training recording contributed usable data; cannot compute "
            "normalization statistics."
        )
    mean = sums / count
    var = np.maximum(sumsq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean, std, files_used


def collect_split(
    files,
    channels: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    max_windows: int | None = None,
    split_name: str = "split",
):
    """Segments + binary labels for a set of recordings (sequential, light).

    Normalizes with the supplied TRAINING statistics only (leakage-safe),
    segments, and labels windows using the real EDF annotations.

    Returns ``(X, y, counts)``:
      * ``X``: float32 ``(n, timesteps, channels)`` (model orientation)
      * ``y``: int64 ``(n,)`` in {0, 1}
      * ``counts``: dict of collection statistics
    ``X``/``y`` are ``None`` when no window was labeled.
    """
    if max_windows is None:
        max_windows = MAX_WINDOWS_TRAIN

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    counts = {
        "seizure": 0, "background": 0, "unlabeled": 0,
        "too_short": 0, "skipped_files": 0, "no_usable_annotations": 0,
        "sfreqs": [],
    }

    for f in files:
        if len(y_parts) and sum(len(y) for y in y_parts) >= max_windows:
            break
        info = _prepare_recording(f, channels, extract_labels=True)
        if info is None:
            counts["skipped_files"] += 1
            continue

        sfreq = info["sfreq"]
        if counts["sfreqs"] and abs(counts["sfreqs"][-1] - sfreq) > 1e-6:
            raise ValueError(
                f"Mixed sampling rates across recordings in split '{split_name}' "
                f"({counts['sfreqs'][-1]} vs {sfreq} Hz); unsupported in one "
                "montage. Resort or resample the dataset."
            )
        counts["sfreqs"].append(sfreq)

        regions = info["regions"]
        seizure_iv, background_iv, unknown_iv = es._annotation_intervals(regions)
        if not seizure_iv and not background_iv:
            counts["no_usable_annotations"] += 1
            continue

        data = np.asarray(pp.apply_normalization(info["data"], mean, std))
        ws = pp._samples_at(sfreq, WINDOW_SECONDS)
        step = pp._samples_at(sfreq, WINDOW_SECONDS - OVERLAP_SECONDS)
        n_samples = data.shape[1]
        if n_samples < ws:
            counts["too_short"] += 1
            continue
        n_total = (n_samples - ws) // step + 1
        stride = max(1, int(np.ceil(n_total / MAX_WINDOWS_PER_FILE)))

        seg_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        for idx in range(0, n_total, stride):
            s = idx * step
            onset = s / sfreq
            offset = (s + ws) / sfreq
            label = es.label_window(
                onset, offset, seizure_iv, background_iv, unknown_iv,
                USE_UNANNOTATED_AS_BACKGROUND,
            )
            if label is None:
                counts["unlabeled"] += 1
                continue
            seg_rows.append(data[:, s : s + ws])
            y_rows.append(label)

        if not seg_rows:
            continue
        seg = np.stack(seg_rows, axis=0).astype(DTYPE)  # (n, ch, t)
        seg = np.ascontiguousarray(seg.transpose(0, 2, 1))  # (n, t, ch)
        X_parts.append(seg)
        y_parts.append(np.asarray(y_rows, dtype=np.int64))
        counts["seizure"] += sum(1 for v in y_rows if v == 1)
        counts["background"] += sum(1 for v in y_rows if v == 0)

        # free per-file temporaries before moving on
        del data, seg_rows, y_rows, seg

    if not X_parts:
        return None, None, counts

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if len(y) > max_windows:
        X = X[:max_windows]
        y = y[:max_windows]
    counts["seizure"] = int((y == 1).sum())
    counts["background"] = int((y == 0).sum())
    return X, y, counts


def validate_model_inputs(X, y, num_channels: int, timesteps: int) -> None:
    """Strict pre-train integrity checks (Section 24):
    shape, no NaN/Inf, labels binary, both classes present.
    """
    from model import validate_model_arrays

    validate_model_arrays(X, y, num_channels, timesteps)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in X/y - pipeline bug.")
    classes = np.unique(y)
    if not set(classes.tolist()).issubset({0, 1}):
        raise ValueError(f"Labels must be binary {0,1}; found {classes.tolist()}.")
    if len(classes) < 2:
        raise ValueError(
            f"Only one class present ({classes.tolist()}); cannot train a "
            "binary classifier with a single class."
        )
    return None


# ---------------------------------------------------------------------------
# Experiment configuration record (written only at real-run time)
# ---------------------------------------------------------------------------
def build_experiment_config(
    dataset_files: list[str],
    split: dict,
    montages: dict,
    preprocessing: dict,
    training: dict,
    threshold: float = 0.5,
    seed: int = RANDOM_SEED,
) -> dict:
    """Assemble the reproducible experiment configuration payload."""
    return {
        "dataset": {
            "edf_files": [str(f) for f in dataset_files],
            "split_method": split.get("description"),
            "train_recordings": [str(f) for f in split.get("train", [])],
            "val_recordings": [str(f) for f in split.get("val", [])],
            "test_recordings": [str(f) for f in split.get("test", [])],
        },
        "montages": montages,
        "preprocessing": preprocessing,
        "training": training,
        "classification_threshold": threshold,
        "random_seed": seed,
        "architecture": (
            "shared CNN-BiLSTM: Conv1D(32,k5)->BN->ReLU->MaxPool->"
            "Conv1D(64,k3)->BN->ReLU->BiLSTM(32)->Dropout->Dense(1,sigmoid)"
        ),
        "data_variable": "only real EDF annotations; unlabeled windows excluded",
    }


def save_experiment_config(payload: dict, path=None) -> Path:
    """Write ``results/metrics/experiment_config.json`` from a real run."""
    if path is None:
        path = METRICS_DIR / "experiment_config.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path