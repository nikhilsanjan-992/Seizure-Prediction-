"""NeuroSelect lightweight EEG preprocessing + segmentation.

Phase 3: prepare real EEG recordings for later electrode selection and
CNN-BiLSTM training. Nothing here trains a model and no synthetic data
is ever generated.

Selectivity rule set by an experimenter, then:
    select_channels -> crop -> filter/notch -> normalize -> segment

Memory: all functions operate on one raw object at a time and prefer
small cropped windows. No function loads every EDF in the dataset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from eeg_pipeline import (
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    _normalize_channel_label,
    discover_edf_files,
    get_channel_types,
    get_channel_names,
    get_sampling_frequency,
    get_standardized_channel_names,
    load_edf,
)

try:
    import mne
except Exception:  # pragma: no cover - defensive
    mne = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (configurable; never blindly applied to an incompatible dataset)
# ---------------------------------------------------------------------------
DEFAULT_LOW_FREQ = 0.5        # Hz, band-pass low edge
DEFAULT_HIGH_FREQ = 40.0      # Hz, band-pass high edge
DEFAULT_NOTCH_FREQ = 50.0     # Hz, mains notch (set 60.0 for US power)
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_OVERLAP_SECONDS = 5.0
DEFAULT_NORMALIZATION = "zscore"

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def validate_filter_params(
    sfreq: float,
    low_freq: float = DEFAULT_LOW_FREQ,
    high_freq: float = DEFAULT_HIGH_FREQ,
    notch_freq: float | None = DEFAULT_NOTCH_FREQ,
) -> list[str]:
    """Return a list of problems for the requested filter configuration.

    Returns an empty list if the configuration is valid for ``sfreq``.
    """
    problems: list[str] = []
    if sfreq <= 0:
        problems.append(f"Invalid sampling frequency ({sfreq} Hz).")
        # Nothing else can be judged meaningfully.
        return problems

    nyquist = sfreq / 2.0

    if low_freq is not None and low_freq <= 0:
        problems.append(f"Low-pass edge must be positive (got {low_freq}).")
    if high_freq is not None and high_freq <= 0:
        problems.append(f"High-pass edge must be positive (got {high_freq}).")
    if (
        low_freq is not None
        and high_freq is not None
        and low_freq >= high_freq
    ):
        problems.append(
            f"Low-pass edge ({low_freq}) must be below high-pass edge ({high_freq})."
        )

    for label, value in (("low", low_freq), ("high", high_freq)):
        if value is None:
            continue
        if value >= nyquist:
            problems.append(
                f"{label}-pass edge {value} Hz must be below the "
                f"Nyquist frequency ({nyquist:g} Hz) of this dataset."
            )

    if notch_freq is not None:
        if notch_freq <= 0:
            problems.append(f"Notch frequency must be positive (got {notch_freq}).")
        elif notch_freq >= nyquist:
            problems.append(
                f"Notch frequency {notch_freq} Hz is at/above the Nyquist "
                f"frequency ({nyquist:g} Hz) of this dataset."
            )
        elif low_freq is not None and notch_freq < low_freq:
            problems.append(
                f"Notch frequency {notch_freq} Hz lies below the band-pass "
                f"low edge ({low_freq} Hz); it would filter nothing useful."
            )
    return problems


def filter_eeg(
    raw,
    low_freq: float = DEFAULT_LOW_FREQ,
    high_freq: float = DEFAULT_HIGH_FREQ,
    notch_freq: float | None = DEFAULT_NOTCH_FREQ,
    picks: list[str] | None = None,
    copy: bool = True,
):
    """Apply band-pass (+ optional notch) filtering to a raw object.

    Channel parameters are validated against the dataset's actual sampling
    frequency first. Raises ``ValueError`` with actionable reasons if the
    requested filters are not representable on this recording. Returns a
    filtered copy by default; original data is never modified in place.
    """
    if mne is None:
        raise RuntimeError("MNE is not importable; cannot filter EEG.")

    sfreq = get_sampling_frequency(raw)
    problems = validate_filter_params(sfreq, low_freq, high_freq, notch_freq)
    if problems:
        raise ValueError("Filter configuration incompatible with dataset:\n  " + "\n  ".join(problems))

    working = raw.copy() if copy else raw
    if working.n_times == 0:
        raise ValueError("Cannot filter an empty recording.")

    if picks is None:
        picks = get_channels_by_type(working)["eeg"]

    if picks:
        # Ensure working object has data loaded (required for _data write-back).
        if not getattr(working, "preload", False):
            working.load_data()
        fil = working.copy()
        fil.pick(picks)
        fil.filter(
            l_freq=low_freq,
            h_freq=high_freq,
            picks="eeg",
            method="fir",
            verbose="ERROR",
        )
        if notch_freq is not None:
            fil.notch_filter(
                freqs=[notch_freq],
                picks="eeg",
                method="fir",
                verbose="ERROR",
            )
        # Re-insert the filtered EEG channels back into the full working copy.
        original = working
        keep_idxs = [i for i, ch in enumerate(original.ch_names) if ch in picks]
        for i, ch in zip(keep_idxs, picks):
            original._data[i] = fil.get_data(picks=[ch])[0]
        original.info["bads"] = [b for b in original.info.get("bads", [])]
        return original

    return working


# ---------------------------------------------------------------------------
# Channel handling
# ---------------------------------------------------------------------------
def get_channels_by_type(raw) -> dict[str, list[str]]:
    """Group channel names by MNE channel type (eeg, eog, stim, misc ...)."""
    names = get_channel_names(raw)
    types = get_channel_types(raw)
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(types[name], []).append(name)
    return grouped


def get_eeg_channel_names(raw) -> list[str]:
    """Return only channels whose MNE type is 'eeg'."""
    grouped = get_channels_by_type(raw)
    return list(grouped.get("eeg", []))


CHANNEL_ALIASES: dict[str, list[str]] = {
    "T3": ["T3", "T7"],
    "T7": ["T7", "T3"],
    "T4": ["T4", "T8"],
    "T8": ["T8", "T4"],
    "T5": ["T5", "P7"],
    "P7": ["P7", "T5"],
    "T6": ["T6", "P8"],
    "P8": ["P8", "T6"],
}


def match_channel_requests(raw, requested: list[str]) -> tuple[list[str], list[str]]:
    """Match requested channel labels against the recording.

    Matching is case/reference insensitive (uses the Phase 2 helper
    ``eeg_pipeline._normalize_channel_label``). Supports matching monopolar
    electrodes from bipolar pair labels (e.g. 'FP1-F7' matches 'FP1') and
    10-20 standard aliases (e.g. 'T4' matches 'T8').
    Returns ``(available_original_names, missing_requests)``. The original
    channel names are always preserved.
    """
    lookup: dict[str, str] = {}  # normalized label -> original name
    for original in get_channel_names(raw):
        norm = _normalize_channel_label(original)
        lookup.setdefault(norm, original)
        # Support bipolar EEG pair labels (e.g. "FP1-F7" -> "FP1", "F7")
        if "-" in original:
            parts = original.split("-")
            for part in parts:
                p_norm = _normalize_channel_label(part)
                if p_norm and p_norm not in lookup:
                    lookup[p_norm] = original

    available: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for req in requested:
        key = _normalize_channel_label(req)
        matched = lookup.get(key)
        if matched is None:
            # Check 10-20 standard electrode aliases (e.g. T4 <-> T8)
            for alt in CHANNEL_ALIASES.get(key, []):
                matched = lookup.get(alt)
                if matched is not None:
                    break
        if matched is not None:
            if matched not in seen:
                available.append(matched)
                seen.add(matched)
        else:
            missing.append(req)
    return available, missing


def select_channels(
    raw,
    requested: list[str] | None = None,
    ignore_non_eeg: bool = True,
):
    """Select a safe channel subset.

    Returns ``(picked_raw, ignored_names, missing_channels)``.

    - ``requested=None``: pick all EEG channels; non-EEG channels are
      reported in ``ignored_names``.
    - ``requested=[...]``: match by normalized label; unmatched requests go
      to ``missing_channels`` and are NOT silently synthesized.
    - If nothing is selected, ``picked_raw`` is ``None``.
    Specific 5/4-electrode selection is Phase 4 and is not performed here.
    """
    if requested is None:
        available = get_eeg_channel_names(raw)
        all_names = get_channel_names(raw)
        ignored = [ch for ch in all_names if ch not in available]
        missing: list[str] = []
    else:
        available, missing = match_channel_requests(raw, requested)
        ignored = [ch for ch in get_channel_names(raw) if ch not in available]

    if not available:
        return None, ignored, missing

    selected = raw.copy()
    selected.pick(available)
    kept = get_channel_names(selected)
    if len(kept) < len(available):
        missing.extend([ch for ch in available if ch not in kept])
    return selected, ignored, missing


# ---------------------------------------------------------------------------
# Normalization (per-channel, leakage-safe)
# ---------------------------------------------------------------------------
NZERO_EPS = 1e-12


def compute_normalization_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the time axis (axis=1). Data (n, t)."""
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    return mean, std


def apply_normalization(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = NZERO_EPS,
) -> np.ndarray:
    """Apply precomputed per-channel z-score to ``data`` (n, t).

    Channels with std <= eps are left at zero to avoid division blow-ups.
    ``mean``/``std`` must be computed on the TRAINING fold only when used
    inside the ML pipeline (see module docstring).
    """
    data = np.asarray(data, dtype=np.float64)
    std_safe = np.where(std <= eps, 1.0, std)
    normalized = (data - mean[:, None]) / std_safe[:, None]
    if np.any(std <= eps):
        logger.warning("Channels with (near-)zero std were zeroed: %d", int(np.count_nonzero(std <= eps)))
    return normalized


def normalize_channels(
    data: np.ndarray,
    method: str = DEFAULT_NORMALIZATION,
) -> np.ndarray:
    """Fit and apply per-channel normalization (convenience, exploration only).

    ``method='zscore'`` (default). This version computes statistics from the
    data it receives; when building the final pipeline, use
    ``compute_normalization_stats`` on training data and
    ``apply_normalization`` on validation/test data to avoid leakage.
    """
    if method not in ("zscore",):
        raise ValueError(f"Unsupported normalization method: {method!r}")
    mean, std = compute_normalization_stats(data)
    return apply_normalization(data, mean, std)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def _samples_at(sfreq: float, seconds: float) -> int:
    return int(round(seconds * sfreq))


def segment_eeg(
    data: np.ndarray,
    window_size: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP_SECONDS,
    sfreq: float = 100.0,
) -> np.ndarray:
    """Sliding-window segmentation of ``data``.

    Array layout: ``(n_segments, n_channels, window_samples)``. Only full
    windows are produced; a leftover tail is dropped rather than padded
    (no fabricated samples). Errors are raised for invalid configs.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0 (got {window_size}).")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0 (got {overlap}).")
    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than window_size ({window_size})."
        )
    if sfreq <= 0:
        raise ValueError(f"sfreq must be > 0 (got {sfreq}).")

    window_samples = _samples_at(sfreq, window_size)
    step_samples = _samples_at(sfreq, window_size - overlap)
    if step_samples < 1:
        raise ValueError(
            f"Window extent ({window_size}s, {window_samples} samples) / step "
            "resolves to <1 sample; increase window_size or sfreq."
        )

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
        n_channels, n_samples = data.shape
    elif data.ndim == 2:
        n_channels, n_samples = data.shape
    else:
        raise ValueError(f"data must be 2-D (channels, samples); got shape {data.shape}")

    if n_samples < window_samples:
        raise ValueError(
            f"Data has only {n_samples} samples (< window {window_samples}). "
            "Cannot form a single full window."
        )

    n_segments = (n_samples - window_samples) // step_samples + 1
    segments = np.stack(
        [
            data[:, i * step_samples : i * step_samples + window_samples]
            for i in range(n_segments)
        ]
    )
    return np.ascontiguousarray(segments)


# ---------------------------------------------------------------------------
# Annotation / label handling (no invented labels)
# ---------------------------------------------------------------------------
# Very conservative, dataset-agnostic defaults expressed as regex patterns
# (searched case-insensitively). Real datasets MUST still be audited in a
# dataset-specific mapping block before training.
DEFAULT_ANNOTATION_LABELS: dict[str, tuple[str, ...]] = {
    "seizure": (r"\bsz\d*\b", r"seiz", r"ictal"),
    "background": (r"background", r"non.?seizure", r"interictal", r"preictal", r"normal"),
}


def label_event_regions(raw):
    """Read annotations and tag each with one of seizure/background/unknown.

    Returns ``(regions, unmapped_descriptions)`` where each region is:
    ``{onset, duration, description, label}``. Matching uses conservative
    regex patterns; descriptions that match nothing are tagged ``'unknown'``
    and reported in ``unmapped_descriptions`` - they are never assumed to
    mean anything. `sz1`, `sz2` style tokens (CHB-MIT style) are recognized.
    """
    ann = raw.annotations

    def score(description: str) -> list[str]:
        text = description.lower()
        hits: list[str] = []
        for label, patterns in DEFAULT_ANNOTATION_LABELS.items():
            if any(re.search(p, text) for p in patterns):
                hits.append(label)
        return hits

    regions: list[dict] = []
    unmapped: set[str] = set()
    for i in range(len(ann)):
        desc = str(ann.description[i])
        hits = score(desc)
        # Prefer 'seizure' whenever any seizure pattern fired.
        label = "seizure" if "seizure" in hits else (hits[0] if hits else "unknown")
        if not hits:
            unmapped.add(desc)
        regions.append(
            {
                "onset": float(ann.onset[i]),
                "duration": float(ann.duration[i]),
                "description": desc,
                "label": label,
            }
        )
    return regions, sorted(unmapped)


# ---------------------------------------------------------------------------
# Pipeline helpers for memory-safe, chunked processing
# ---------------------------------------------------------------------------
def crop_raw(raw, max_seconds: float | None = 60.0, max_samples: int | None = None):
    """Return a cropped copy of ``raw`` limited to the first N seconds/samples.

    Keeps heavy operations off whole multi-hour recordings while testing.
    """
    n_total = raw.n_times
    if max_samples is None and max_seconds is not None:
        max_samples = int(max_seconds * get_sampling_frequency(raw))
    elif max_samples is None:
        max_samples = n_total
    n_keep = max(1, min(int(max_samples), n_total))
    return raw.copy().crop(tmin=0.0, tmax=(n_keep - 1) / get_sampling_frequency(raw))