"""NeuroSelect data-driven optimal electrode selection.

Phase 4: rank EEG electrodes from REAL training data only and report the
top-5 / top-4 subsets for the 22-vs-5-vs-4 comparison.

Scientific constraints implemented here:

1. DATA-DRIVEN — the ranking comes from the actual feature discriminability
   of each channel on training windows. No hard-coded electrode lists.
2. TRAINING-ONLY — electrodes are ranked using a subject/session-aware
   training subset. Validation/test recordings are excluded from ranking.
3. RELIABLE LABELS ONLY — if seizure/background windows cannot be formed
   from real annotation data, the pipeline refuses rather than inventing
   labels.
4. EXPLAINABLE — the importance metric is the mean ANOVA F-statistic of
   the per-channel features (seizure vs background windows), normalized to
   [0, 1] across channels. Deterministic; no random sampling is used.

No model is trained here and no synthetic EEG is generated.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.feature_selection import f_classif

from preprocessing import (
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    PLOTS_DIR,
    get_eeg_channel_names,
)

try:
    from config import METRICS_DIR
except ImportError:  # pragma: no cover - fallback when src/ is not on sys.path
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import METRICS_DIR  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration (documented; adjust for each real dataset)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42  # reproducibility if a randomized method is ever selected

MAX_TRAIN_SECONDS_PER_RECORDING = 3600.0  # process at most 1 h per recording
MAX_TRAIN_WINDOWS_PER_FILE = 500          # stride-shrink very long recordings
MAX_LABELED_WINDOWS_TOTAL = 3000          # global cap for the feature matrix

# Reliability gates: refuse to run if these counts are not reached.
MIN_SEIZURE_WINDOWS = 10
MIN_BACKGROUND_WINDOWS = 10

# A window counts as a seizure window only if >= this fraction overlaps an
# explicit seizure region (or its center lies inside one).
SEIZURE_OVERLAP_FRACTION = 0.5

# Explicit, dataset-specific label patterns. Populate ONLY after inspecting
# the actual dataset's annotation descriptions. Empty by default: the
# conservative default matcher in preprocessing.label_event_regions() is
# used unless these are set.
SEIZURE_LABEL_PATTERNS: tuple[str, ...] = ()
BACKGROUND_LABEL_PATTERNS: tuple[str, ...] = ()

# CHB-MIT stores seizure times in sidecar .seizure/.txt files, not in the
# EDF annotations. Reading those is dataset-specific and NOT implemented
# here; if the dataset relies on them, label extraction reports a blocker.
SUPPORT_SIDECAR_SEIZURE_FILES = False

EPS = 1e-12

# Human-readable description of the implemented selection method (used in
# every report/file; keep in sync with compute_channel_importance()).
METHOD_DESCRIPTION = (
    "Mean ANOVA F-statistic of per-channel features (seizure vs background "
    "training windows), channel score normalized to [0, 1] by the largest "
    "channel's mean F."
)

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
def get_available_eeg_channels(raw) -> list[str]:
    """EEG channels available in one recording (reuses Phase 3 helper)."""
    return get_eeg_channel_names(raw)


def common_eeg_channels(raws) -> list[str]:
    """Channels present in every raw (intersection), preserving first order.

    Electrode selection must operate on a shared montage; channels that a
    recording lacks cannot be compared.
    """
    if not raws:
        return []
    common: list[str] | None = None
    for raw in raws:
        names = get_eeg_channel_names(raw)
        if common is None:
            common = list(names)
        else:
            common_set = set(common)
            names_set = set(names)
            common = [ch for ch in common if ch in names_set]
    return common or []


# ---------------------------------------------------------------------------
# Per-channel features
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    "mean",
    "std",
    "variance",
    "rms",
    "line_length",
    "hjorth_mobility",
    "hjorth_complexity",
    "band_delta",
    "band_theta",
    "band_alpha",
    "band_beta",
)

_BAND_EDGES = {"band_delta": (0.5, 4.0), "band_theta": (4.0, 8.0),
               "band_alpha": (8.0, 13.0), "band_beta": (13.0, 30.0)}


def calculate_channel_features(data: np.ndarray, sfreq: float):
    """Compute per-channel features for one window.

    ``data`` shape (n_channels, n_samples). Returns
    ``(feature_names, features)`` where ``features`` has shape
    ``(n_channels, n_features)`` — kept small on purpose.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    n_ch, n = data.shape

    mean = data.mean(axis=-1)
    std = data.std(axis=-1)
    var = std ** 2
    rms = np.sqrt(np.mean(data ** 2, axis=-1))

    diff1 = np.diff(data, axis=-1)
    line_length = np.abs(diff1).sum(axis=-1)

    s0 = std
    s1 = diff1.std(axis=-1)
    s2 = np.diff(diff1, axis=-1).std(axis=-1)
    mobility = s1 / (s0 + EPS)
    complexity = (s2 / (s1 + EPS)) / (mobility + EPS)

    band_cols: list[np.ndarray] = []
    nperseg = min(256, n)
    if nperseg >= 8 and sfreq > 0:
        freqs, pxx = welch(data, fs=sfreq, axis=-1, nperseg=nperseg,
                           noverlap=nperseg // 2)
        lo, hi = 0.5, min(40.0, sfreq / 2.0)
        total_mask = (freqs >= lo) & (freqs <= hi)
        total = pxx[:, total_mask].sum(axis=-1) + EPS
        for fname in ("band_delta", "band_theta", "band_alpha", "band_beta"):
            lo_f, hi_f = _BAND_EDGES[fname]
            hi_f = min(hi_f, hi)
            mask = (freqs >= lo_f) & (freqs <= hi_f)
            band = pxx[:, mask].sum(axis=-1)
            band_cols.append(band / total)
    else:
        for _ in ("band_delta", "band_theta", "band_alpha", "band_beta"):
            band_cols.append(np.zeros(n_ch))

    features = np.column_stack(
        [mean, std, var, rms, line_length, mobility, complexity, *band_cols]
    )
    return list(FEATURE_NAMES), np.ascontiguousarray(features)


# ---------------------------------------------------------------------------
# Window labeling from REAL annotations
# ---------------------------------------------------------------------------
def _annotation_intervals(regions) -> tuple[list, list, list]:
    seizure = [(r["onset"], r["onset"] + r["duration"])
               for r in regions if r["label"] == "seizure" and r["duration"] > 0]
    background = [(r["onset"], r["onset"] + r["duration"])
                  for r in regions if r["label"] == "background" and r["duration"] > 0]
    unknown = [(r["onset"], r["onset"] + r["duration"])
               for r in regions if r["label"] == "unknown" and r["duration"] > 0]
    return seizure, background, unknown


def label_window(
    onset: float,
    offset: float,
    seizure_intervals,
    background_intervals,
    unknown_intervals,
    use_unannotated_as_background: bool,
) -> int | None:
    """Classify one window; returns 1 (seizure), 0 (background) or None.

    None means "not confidently labeled" — the window is excluded, never
    guessed.
    """
    length = offset - onset
    if length <= 0:
        return None

    for s_start, s_end in seizure_intervals:
        overlap = max(0.0, min(offset, s_end) - max(onset, s_start))
        frac = overlap / length
        center = 0.5 * (onset + offset)
        if frac >= SEIZURE_OVERLAP_FRACTION or (s_start <= center <= s_end):
            return 1

    def _inside(iv):
        for iv_start, iv_end in iv:
            if onset >= iv_start and offset <= iv_end:
                return True
        return False

    if _inside(background_intervals):
        return 0

    if use_unannotated_as_background:
        if _inside(unknown_intervals):
            return None
        for u_start, u_end in unknown_intervals:
            if not (offset <= u_start or onset >= u_end):
                return None
        return 0
    return None


def _label_regions(raw):
    """Label-aware region extraction (uses configured patterns if set).

    By default this reuses the conservative matcher from Phase 3. If the
    dataset-specific ``SEIZURE_LABEL_PATTERNS``/``BACKGROUND_LABEL_PATTERNS``
    constants are populated, they temporarily override the matching rules.
    """
    import preprocessing as pp

    if not (SEIZURE_LABEL_PATTERNS or BACKGROUND_LABEL_PATTERNS):
        return pp.label_event_regions(raw)

    old_seizure = pp.DEFAULT_ANNOTATION_LABELS["seizure"]
    old_background = pp.DEFAULT_ANNOTATION_LABELS["background"]
    if SEIZURE_LABEL_PATTERNS:
        pp.DEFAULT_ANNOTATION_LABELS["seizure"] = SEIZURE_LABEL_PATTERNS
    if BACKGROUND_LABEL_PATTERNS:
        pp.DEFAULT_ANNOTATION_LABELS["background"] = BACKGROUND_LABEL_PATTERNS
    try:
        return pp.label_event_regions(raw)
    finally:
        pp.DEFAULT_ANNOTATION_LABELS["seizure"] = old_seizure
        pp.DEFAULT_ANNOTATION_LABELS["background"] = old_background


# ---------------------------------------------------------------------------
# Streaming (memory-light) training window collection
# ---------------------------------------------------------------------------
def collect_labeled_windows(
    raw,
    common_channels: list[str],
    sfreq: float,
    window_size: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP_SECONDS,
    use_unannotated_as_background: bool = True,
):
    """Collect per-window channel features + labels from ONE recording.

    Processes one file sequentially. Returns
    ``(X, y, labeled_count_by_class, excluded_counts)`` where
    ``X`` (n_windows, n_channels, n_features), ``y`` (n_windows,) in {0, 1}.
    """
    ws = int(round(window_size * sfreq))
    step = max(1, int(round((window_size - overlap) * sfreq)))
    n = raw.n_times
    n_total_windows = (n - ws) // step + 1 if n >= ws else 0

    if n_total_windows <= 0:
        return None, None, {}, {"too_short": 1}

    stride = 1
    if n_total_windows > MAX_TRAIN_WINDOWS_PER_FILE:
        stride = int(np.ceil(n_total_windows / MAX_TRAIN_WINDOWS_PER_FILE))

    regions, unmapped = _label_regions(raw)
    seizure_iv, background_iv, unknown_iv = _annotation_intervals(regions)

    if not seizure_iv and not background_iv:
        return None, None, {}, {"no_usable_annotations": len(unmapped)}

    start_candidates = range(0, n_total_windows, stride)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    counts = {"seizure": 0, "background": 0, "excluded": 0, "unlabeled": 0}

    # One window at a time: never materialize the full segment tensor.
    data_slice = raw.get_data(picks=common_channels)
    del raw  # do not hold the raw object once data is read

    for idx in start_candidates:
        s_start = idx * step
        s_end = s_start + ws
        onset = s_start / sfreq
        offset = s_end / sfreq

        label = label_window(
            onset, offset, seizure_iv, background_iv, unknown_iv,
            use_unannotated_as_background,
        )
        if label is None:
            counts["unlabeled"] += 1
            continue

        window = data_slice[:, s_start:s_end]
        _, feats = calculate_channel_features(window, sfreq)
        X_rows.append(feats)
        y_rows.append(label)
        counts["seizure" if label == 1 else "background"] += 1

    if not X_rows:
        return None, None, counts, {}

    X = np.stack(X_rows)
    y = np.asarray(y_rows, dtype=np.int64)
    return X, y, counts, {}


# ---------------------------------------------------------------------------
# Importance, ranking, selection
# ---------------------------------------------------------------------------
def compute_channel_importance(X: np.ndarray, y: np.ndarray,
                               channel_names: list[str]):
    """ANOVA F-statistic importance per channel (seizure vs background).

    ``X`` (n_windows, n_channels, n_features). For every channel the mean
    F-statistic across its features is computed, then normalized by the
    largest channel mean so scores live in [0, 1]. Deterministic.
    """
    unique, counts = np.unique(y, return_counts=True)
    expected = {0, 1}
    if set(unique.tolist()) != expected:
        raise ValueError(
            f"Need both classes (0/1) to compute importance; found {unique.tolist()}."
        )
    if int(counts.min()) < 2:
        raise ValueError(
            f"Each class needs >= 2 windows for ANOVA; found {counts.tolist()}."
        )

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n_ch = X.shape[1]

    F_mean = np.zeros(n_ch)
    for c in range(n_ch):
        with np.errstate(all="ignore"):
            F, _ = f_classif(X[:, c, :], y)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F_mean[c] = float(F.mean())

    max_f = float(F_mean.max())
    if max_f <= 0:
        return {name: 0.0 for name in channel_names}, {
            "max_channel_mean_f": 0.0, "degenerate": True,
        }

    scores = {name: float(F_mean[i] / max_f) for i, name in enumerate(channel_names)}
    return scores, {"max_channel_mean_f": max_f, "degenerate": False}


def rank_channels(importance: dict[str, float]) -> list[tuple[str, float]]:
    """Rank channels by importance, descending; ties broken alphabetically."""
    return sorted(importance.items(), key=lambda kv: (-kv[1], kv[0]))


def select_top_channels(ranked: list[tuple[str, float]], k: int) -> list[str]:
    """Return the top ``k`` channel names."""
    return [name for name, _ in ranked[:k]]


# ---------------------------------------------------------------------------
# Reproducible subject/session-aware split
# ---------------------------------------------------------------------------
def infer_subject_id(path: str | Path) -> int | None:
    """Best-effort subject id from a file name.

    Handles ``chb01_01.edf``, ``sub-003_sess1.edf``, ``patient07_a.edf``
    styles. Returns None when no recognizable pattern exists (the caller
    then documents a file-level limitation).
    """
    stem = Path(path).stem.lower()
    patterns = (
        r"chb(\d+)",            # chb01_01
        r"sub[-_]?0*(\d+)",     # sub-003
        r"patient\s*0*(\d+)",   # patient07
        r"(?:^|_{0,1})0*(\d+)[_-](?=[a-z0-9]*$)",  # 01-a / 01_01 style
    )
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))
    return None


def subject_aware_train_split(edf_files, train_fraction: float = 0.8):
    """Split EDF files into (training, holdout) at subject/session level.

    Subjects are sorted by id and the first ``train_fraction`` subjects form
    the training set. If no subject id can be inferred, falls back to a
    file-level split and documents the limitation. Deterministic.
    """
    ids = [infer_subject_id(f) for f in edf_files]
    if all(i is not None for i in ids):
        unique_ids = sorted(set(ids), key=lambda i: (i is None, i))
        n_train_subjects = max(1, int(round(len(unique_ids) * train_fraction)))
        train_ids = set(unique_ids[:n_train_subjects])
        train_files = [f for f, i in zip(edf_files, ids) if i in train_ids]
        holdout_files = [f for f, i in zip(edf_files, ids) if i not in train_ids]
        return train_files, holdout_files, "subject-level split"
    n_train = max(1, int(round(len(edf_files) * train_fraction)))
    return edf_files[:n_train], edf_files[n_train:], (
        "file-level split (subject id not inferable; subject independence "
        "cannot be guaranteed)"
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_selection_results(
    ranked: list[tuple[str, float]],
    top5: list[str],
    top4: list[str],
    method: str,
    extra: dict | None = None,
    out_dir: str | Path = METRICS_DIR,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "electrode_ranking.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "channel", "importance_score"])
        for idx, (name, score) in enumerate(ranked, start=1):
            writer.writerow([idx, name, f"{score:.6f}"])

    payload = {
        "top_5": top5,
        "top_4": top4,
        "method": method,
        "candidate_channels": [name for name, _ in ranked],
        "random_seed": RANDOM_SEED,
    }
    if extra:
        payload.update(extra)
    json_path = out_dir / "selected_channels.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return csv_path, json_path


def plot_channel_importance(
    ranked: list[tuple[str, float]],
    save_path: str | Path = PLOTS_DIR / "electrode_importance.png",
):
    """One professional importance-ranking figure, sorted descending."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name, _ in reversed(ranked)]
    scores = [float(score) for _, score in reversed(ranked)]

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 2))
    ax.barh(range(len(names)), scores, color="#4C72B0", edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Adaptive channel importance (mean ANOVA F, normalized)")
    ax.set_title("NeuroSelect electrode importance ranking")
    ax.set_xlim(0, 1.0)
    for i, s in enumerate(scores):
        ax.text(min(s, 0.98) + 0.01, i, f"{s:.3f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path