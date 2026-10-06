"""NeuroSelect Phase 3 preprocessing validation script.

Usage:
    python scripts/test_preprocessing.py [--plot] [--max-seconds N]

Finds a real EDF under data/raw/, crops a small interval, selects EEG
channels, filters, normalizes, segments, and reports shapes + NaN/Inf
counts. No model is trained. Exits cleanly if no real EDF exists.

The real run needs at least one real (non-synthetic) EDF.

Exit codes:
  0  test passed against a real EDF
  1  no real EDF found (data blocked, not a code failure)
  2  EDF present but preprocessing failed
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import eeg_pipeline as ep  # noqa: E402
import preprocessing as pp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSelect preprocessing validation")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save ONE before/after plot to results/plots/",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60.0,
        help="crop to the first N seconds before processing (default 60)",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="explicit channel list (default: all EEG channels)",
    )
    args = parser.parse_args()

    edf_files = pp.discover_edf_files()
    if not edf_files:
        print("NO REAL EDF DATASET FOUND")
        print("-" * 30)
        print("Place at least one real EDF/*.edf.gz recording inside:")
        print(f"  {pp.RAW_DATA_DIR}")
        print("Then re-run this script to validate preprocessing.")
        return 1

    path = edf_files[0]
    print("PREPROCESSING TEST")
    print("-" * 24)
    print(f"EDF: {path.name}")
    print(f"Max seconds used: {args.max_seconds}")

    try:
        raw = ep.load_edf(path, preload=False)
    except Exception as exc:
        print(f"ERROR: could not load EDF: {exc}")
        return 2

    try:
        sfreq = ep.get_sampling_frequency(raw)
        print(f"Sampling frequency: {sfreq} Hz")

        n_orig = getattr(raw, "n_times", 0)
        print(f"Original samples: {n_orig} ({n_orig / sfreq:.2f} s of recording)")

        # Crop to a small interval first: keeps whole files off RAM.
        raw = pp.crop_raw(raw, max_seconds=args.max_seconds)
        print(f"Cropped samples: {raw.n_times}")

        # 1. Channel selection (safe; no final 5/4-electrode selection here).
        picked, ignored, missing = pp.select_channels(raw, requested=args.channels)
        for ch in ignored:
            print(f"  Ignored non-EEG channel: {ch}")
        for ch in missing:
            print(f"  MISSING requested channel (no synthetic substitute): {ch}")
        if picked is None:
            print("ERROR: no EEG channels were selectable in this file")
            return 2
        channels = ep.get_channel_names(picked)
        print(f"Channels selected: {len(channels)} -> {', '.join(channels)}")

        # 2. Validate filter params against this dataset's sfreq (report not crash).
        problems = pp.validate_filter_params(
            sfreq,
            pp.DEFAULT_LOW_FREQ,
            pp.DEFAULT_HIGH_FREQ,
            pp.DEFAULT_NOTCH_FREQ,
        )
        if problems:
            print("ERROR: filter configuration incompatible with dataset:")
            for p in problems:
                print(f"  - {p}")
            return 2

        # 3. Band-pass + notch.
        filtered = pp.filter_eeg(picked, copy=True)

        # 4. Per-channel z-score normalization.
        data = filtered.get_data()
        print(f"Data shape after filtering: {data.shape}")
        normalized = pp.normalize_channels(data, method="zscore")

        # 5. Segmentation with configurable window/overlap.
        segments = pp.segment_eeg(
            normalized,
            window_size=pp.DEFAULT_WINDOW_SECONDS,
            overlap=pp.DEFAULT_OVERLAP_SECONDS,
            sfreq=sfreq,
        )

        n_nan = int(np.isnan(segments).sum())
        n_inf = int(np.isinf(segments).sum())

        print(f"Window size: {pp.DEFAULT_WINDOW_SECONDS:.1f} seconds")
        print(f"Overlap: {pp.DEFAULT_OVERLAP_SECONDS:.1f} seconds")
        print(f"Number of segments: {segments.shape[0]}")
        print(f"Segment shape: {tuple(segments.shape)}")
        print(f"NaN values: {n_nan}")
        print(f"Inf values: {n_inf}")

        # Optional single before/after plot.
        if args.plot:
            ep.plot_eeg_segment(raw, duration=10.0, save_path=ep.PLOTS_DIR / "pre_before.png")
            ep.plot_eeg_segment(filtered, duration=10.0, save_path=ep.PLOTS_DIR / "pre_after.png")
            print("  Saved before/after inspection plots to results/plots/")

        print("-" * 24)
        ok = n_nan == 0 and n_inf == 0 and segments.ndim == 3
        print("PREPROCESSING TEST: SUCCESS" if ok else "PREPROCESSING TEST: FAILED")
        return 0 if ok else 2
    finally:
        raw.close()


if __name__ == "__main__":
    sys.exit(main())