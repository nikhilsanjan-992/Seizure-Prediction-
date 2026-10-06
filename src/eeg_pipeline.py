"""NeuroSelect EEG inspection pipeline.

Phase 2: connect to real EDF EEG datasets and inspect them reliably.

This module ONLY discovers, loads (metadata-light), normalizes channel
names, checks pipeline compatibility, and optionally plots short EEG
windows. It never trains models and never fabricates data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from config import PROCESSED_DATA_DIR, PLOTS_DIR, RAW_DATA_DIR
except ImportError:  # pragma: no cover - fallback when src/ is not on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PROCESSED_DATA_DIR, PLOTS_DIR, RAW_DATA_DIR  # noqa: F401

import mne
import numpy as np  # noqa: F401  (kept for future preprocessing steps)

logger = logging.getLogger(__name__)

EDF_SUFFIXES = (".edf", ".edf.gz")

# Reference montage planned for the 22-channel comparison experiment.
# Used only for compatibility reporting; NO final electrode selection
# is performed in this phase.
PLANNED_REFERENCE_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ", "FPZ",
    "A1", "A2",
)


def discover_edf_files(root: str | Path | None = None) -> list[Path]:
    """Recursively find EDF / EDF+GZ files under a root directory.

    If ``root`` is None, ``config.RAW_DATA_DIR`` is used.
    """
    if root is None:
        root = RAW_DATA_DIR
    root = Path(root)
    if not root.is_dir():
        logger.warning("Data root does not exist: %s", root)
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith(EDF_SUFFIXES)
    )


def load_edf(path: str | Path, preload: bool = False):
    """Load an EDF/EDF+GZ file with MNE, memory-light by default."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"EDF file not found: {path}")
    logger.info("Loading EDF (preload=%s): %s", preload, path.name)
    return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")


def _format_timedelta(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes} min {rem:.1f} s ({seconds:.2f} s total)"


def get_channel_names(raw) -> list[str]:
    """Return the original channel names exactly as stored in the file."""
    return list(raw.ch_names)


def get_sampling_frequency(raw) -> float:
    return float(raw.info["sfreq"])


def get_duration(raw) -> float:
    """Recording duration in seconds."""
    sfreq = get_sampling_frequency(raw)
    return float(raw.n_times) / float(sfreq)


def get_number_of_channels(raw) -> int:
    return len(raw.ch_names)


def get_annotations(raw, max_items: int = 50):
    """Return a summary list of annotations.

    Each item is a dict with onset, duration, description and a channel
    list (typically empty for EDF annotations).
    """
    ann = raw.annotations
    out = []
    for item in range(min(len(ann), max_items)):
        onset = float(ann.onset[item])
        duration = float(ann.duration[item])
        out.append(
            {
                "onset": onset,
                "duration": duration,
                "description": str(ann.description[item]),
                "channels": list(ann.ch_names[item]),
            }
        )
    return out


def get_measurement_date(raw):
    """Return the measurement date if present, otherwise None."""
    date = raw.info.get("meas_date")
    if date is None:
        return None
    try:
        return date.replace(tzinfo=None)
    except Exception:
        return date


def get_channel_types(raw) -> dict[str, str]:
    """Return a mapping of channel name -> channel type."""
    return {name: str(t) for name, t in zip(raw.ch_names, raw.get_channel_types())}


def inspect_raw(raw) -> None:
    """Print metadata about a raw object to stdout."""
    print(f"  Number of channels          : {get_number_of_channels(raw)}")
    print(f"  Channel names               : {', '.join(get_channel_names(raw))}")
    print(f"  Sampling frequency          : {get_sampling_frequency(raw):g} Hz")
    print(f"  Recording duration          : {_format_timedelta(get_duration(raw))}")
    print(f"  Number of samples           : {raw.n_times}")
    types = get_channel_types(raw)
    print(f"  Channel types               : {', '.join(sorted(set(types.values())))}")
    date = get_measurement_date(raw)
    print(f"  Measurement date            : {date.isoformat() if date else 'Not available'}")
    n_annot = len(raw.annotations)
    if n_annot:
        shown = min(n_annot, 50)
        print(f"  Annotations                 : {shown} shown of {n_annot} total (first)")
    else:
        print(f"  Annotations                 : 0 total")


def _normalize_channel_label(label: str) -> str:
    """Normalize a single channel label for comparison.

    Uppercases, removes a trailing reference suffix (``-REF``, ``-AV`` ...)
    and removes separators (``-``, ``_``, spaces).
    The original label is always preserved by the caller.
    """
    name = str(label).strip().upper()
    for suffix in ("-AVREF", "_AVREF", "-REF", "_REF", "-AV", "_AV"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = name.replace("-", "").replace("_", "").replace(" ", "")
    return name


def get_standardized_channel_names(raw) -> dict[str, str]:
    """Map each original channel name to a normalized identifier.

    Returns ``{original_name: normalized_name}``. Allows later
    identification of standard EEG electrodes without renaming anything
    in the raw object.
    """
    return {name: _normalize_channel_label(name) for name in get_channel_names(raw)}


def build_channel_mapping(raw) -> dict[str, str]:
    """Alias for :func:`get_standardized_channel_names` (helper naming)."""
    return get_standardized_channel_names(raw)


def identify_standard_eeg_channels(raw) -> dict[str, str]:
    """Map original channel names to standard 10-20 identifiers when possible.

    Only names that clearly correspond to a known 10-20 electrode are
    included. Original casing/reference suffixes are preserved in the key;
    the mapped value is the canonical uppercase electrode label.
    """
    valid = {_normalize_channel_label(ch) for ch in PLANNED_REFERENCE_CHANNELS}
    mapping: dict[str, str] = {}
    for original, normalized in get_standardized_channel_names(raw).items():
        if normalized in valid:
            mapping[original] = normalized
    return mapping


def check_compatibility(raw) -> dict:
    """Run the NeuroSelect compatibility checks against a loaded raw object.

    Returns a dict with keys: eeg_channels, sampling_frequency, duration,
    compatible, and reasons. Does NOT claim compatibility for an EDF that
    merely opens.
    """
    types = get_channel_types(raw)
    eeg_channels = [name for name, t in types.items() if t.lower() == "eeg"]
    sfreq = get_sampling_frequency(raw)

    reasons: list[str] = []
    compatible = True

    if not eeg_channels:
        compatible = False
        reasons.append("No EEG channels detected in this file.")
    elif len(eeg_channels) < 4:
        compatible = False
        reasons.append(
            f"Only {len(eeg_channels)} EEG channel(s) found; at least 4 are needed "
            "for the planned 22/5/4-channel comparison."
        )
    if sfreq <= 0:
        compatible = False
        reasons.append("Unusable sampling frequency (<= 0 Hz).")
        duration = 0.0
    else:
        if sfreq < 1:
            compatible = False
            reasons.append("Sampling frequency below 1 Hz is unsupported.")
        duration = get_duration(raw)
        if duration <= 0:
            compatible = False
            reasons.append("Recording duration is zero or negative.")

    if not reasons:
        reasons.append(
            f"File provides {len(eeg_channels)} EEG channel(s) at {sfreq:g} Hz "
            "with enough content for selector experiments."
        )

    return {
        "eeg_channels": len(eeg_channels),
        "sampling_frequency": sfreq,
        "duration": duration,
        "compatible": compatible,
        "reasons": reasons,
    }


def print_compatibility(compat: dict, n_edf: int = 0) -> None:
    """Print the DATASET COMPATIBILITY block for a compatibility dict."""
    n_eeg = compat["eeg_channels"]
    print()
    print("DATASET COMPATIBILITY")
    print("-" * 20)
    print(f"EDF files found: {n_edf}")
    print(f"EEG channels: {n_eeg}")
    print(f"Sampling frequency: {compat['sampling_frequency']:g} Hz")
    print(f"Duration: {compat['duration']:.2f} seconds")
    if n_eeg >= 22:
        print("22-channel compatible: YES")
    elif n_eeg >= 4:
        print("22-channel compatible: NO (reduced-channel analysis possible)")
    else:
        print("22-channel compatible: NO")
    for reason in compat["reasons"]:
        print(f"Reason: {reason}")


def print_dataset_summary(edf_files: list[Path], max_files: int = 5) -> None:
    """Print a concise summary of the discovered EDF dataset."""
    print(f"EDF files found: {len(edf_files)}")
    if not edf_files:
        return
    for path in edf_files[:max_files]:
        print(f"  - {path.name}")
    if len(edf_files) > max_files:
        print(f"  ... and {len(edf_files) - max_files} more")


def plot_eeg_segment(
    raw,
    duration: float = 10.0,
    channels: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot a short EEG segment on a copy, without full preloading.

    If saving, the figure is written to ``save_path`` and the figure is
    closed. Without a save path the plot window is shown interactively.
    Only a small time window is read into memory.
    """
    import matplotlib

    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sfreq = get_sampling_frequency(raw)
    t_start = 0.0
    t_end = min(float(duration), get_duration(raw))
    if t_end <= 0:
        print("  No data available to plot.")
        return None

    with raw.copy():
        try:
            data, times = raw[:, int(t_start * sfreq):int(t_end * sfreq)]
        except Exception as exc:
            print(f"  Could not read EEG segment: {exc}")
            return None

    n_rows = min(len(channels) if channels else len(raw.ch_names), 8)
    if n_rows <= 0:
        print("  No channels available to plot.")
        return None
    names = (channels[:n_rows] if channels else raw.ch_names[:n_rows])
    try:
        rows = [raw.ch_names.index(name) for name in names]
    except ValueError as exc:
        print(f"  Unknown channel name in plot request: {exc}")
        return None

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 0.8 * n_rows + 1), sharex=True)
    if n_rows == 1:
        axes = [axes]
    for ax, row, name in zip(axes, rows, names):
        ax.plot(times, data[row], lw=0.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EEG segment with shape {data.shape}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved inspection plot: {save_path}")
        return save_path

    plt.show()
    plt.close(fig)
    return None