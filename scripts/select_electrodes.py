"""NeuroSelect Phase 4 electrode selection.

Usage:
    python scripts/select_electrodes.py [--train-fraction 0.8]
                                        [--max-seconds 3600]
                                        [--window 10] [--overlap 5]

Leakage-safe workflow (throughout):
    real EDFs -> subject/session-aware split -> TRAINING ONLY ->
    preprocessing -> segmentation -> per-channel features ->
    ANOVA importance -> ranking -> TOP 5 / TOP 4.

Training data alone drives the ranking; the hold-out subset is never used.
If reliable labels cannot be built from the real annotations, the script
reports a blocker and writes NO results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import eeg_pipeline as ep  # noqa: E402
import electrode_selection as es  # noqa: E402
import preprocessing as pp  # noqa: E402

BLOCKER_MSG = (
    "RELIABLE TRAINING LABELS NOT AVAILABLE FOR DATA-DRIVEN "
    "ELECTRODE SELECTION."
)


def load_raw(path, max_seconds: float):
    """Load an EDF memory-light, crop, then materialize only the crop."""
    raw = ep.load_edf(path, preload=False)
    duration = ep.get_duration(raw)
    if duration > max_seconds:
        raw.crop(tmax=max_seconds)
    raw.load_data()
    return raw


def annotation_inventory(raw):
    """Count annotation labels actually present (seizure/background/unknown)."""
    regions, unmapped = pp.label_event_regions(raw)
    counts = {"seizure": 0, "background": 0, "unknown": 0, "regions": len(regions)}
    for r in regions:
        counts[r["label"]] += 1
    return counts, unmapped


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSelect electrode selection")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-seconds", type=float,
                        default=es.MAX_TRAIN_SECONDS_PER_RECORDING)
    parser.add_argument("--window", type=float, default=pp.DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=pp.DEFAULT_OVERLAP_SECONDS)
    args = parser.parse_args()

    assert 0 < args.train_fraction <= 1.0, "train-fraction must be in (0, 1]"

    edf_files = pp.discover_edf_files()
    if not edf_files:
        print("NO REAL EDF DATASET FOUND")
        print("-" * 30)
        print(f"Place real EDF files inside: {pp.RAW_DATA_DIR}")
        return 1

    print("ELECTRODE SELECTION")
    print("-" * 19)

    train_files, holdout_files, split_desc = es.subject_aware_train_split(
        edf_files, args.train_fraction
    )
    print(f"Method: {es.METHOD_DESCRIPTION}")
    print(f"Split: {split_desc}")
    print(f"Discovered EDF files: {len(edf_files)}")
    print(f"Training recordings: {len(train_files)}")
    print(f"Hold-out recordings (NOT used for ranking): {len(holdout_files)}")

    # ---------------------------------------------------------------------
    # Pass A - metadata + annotation inventory on training files only
    # ---------------------------------------------------------------------
    raw_pool = []
    annotation_report = []
    for f in train_files:
        raw = load_raw(f, args.max_seconds)
        inventory, unmapped = annotation_inventory(raw)
        n_eeg = len(es.get_available_eeg_channels(raw))
        annotation_report.append(
            {"file": f.name, "eeg_channels": n_eeg, **inventory,
             "unmapped": unmapped}
        )
        raw_pool.append(raw)

    common_channels = es.common_eeg_channels(raw_pool)
    print(f"Candidate EEG channels (common across training files): {len(common_channels)}")

    if len(common_channels) < 4:
        print(BLOCKER_MSG)
        print(f"Reason: only {len(common_channels)} candidate channel(s); "
              "the planned 4-channel minimum cannot be selected.")
        return 2

    # ---------------------------------------------------------------------
    # Pass B - sequential, memory-light labeled window collection
    # ---------------------------------------------------------------------
    X_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    totals = {"seizure": 0, "background": 0, "unlabeled": 0, "excluded": 0}
    files_used = 0

    try:
        for raw, info in zip(raw_pool, annotation_report):
            if len(X_parts) >= es.MAX_LABELED_WINDOWS_TOTAL:
                print("  (global labeled-window cap reached; stopping collection)")
                break
            picked, ignored, missing = pp.select_channels(
                raw, requested=common_channels
            )
            if picked is None:
                print(f"  SKIP {info['file']}: no selectable EEG channels")
                continue
            filtered = pp.filter_eeg(
                picked, copy=False,
                low_freq=pp.DEFAULT_LOW_FREQ,
                high_freq=pp.DEFAULT_HIGH_FREQ,
                notch_freq=pp.DEFAULT_NOTCH_FREQ,
            )
            sfreq = ep.get_sampling_frequency(filtered)
            X, y, counts, excluded = es.collect_labeled_windows(
                filtered,
                common_channels,
                sfreq,
                window_size=args.window,
                overlap=args.overlap,
            )
            for key in totals:
                totals[key] += counts.get(key, 0)
            if X is not None:
                X_parts.append(X)
                y_parts.append(y)
                files_used += 1
            else:
                print(f"  NOTE {info['file']}: no labeled windows "
                      f"(annotations: {info})")
    finally:
        for raw in raw_pool:
            try:
                raw.close()
            except Exception:
                pass

    print(f"Training files with labeled windows: {files_used}")
    print(f"Seizure windows: {totals['seizure']}  "
          f"Background windows: {totals['background']}  "
          f"Excluded/unlabeled: {totals['excluded']}/{totals['unlabeled']}")

    if not X_parts:
        print(BLOCKER_MSG)
        print("Reason: no training recording yielded labeled windows.")
        print("Inspect the annotation inventory above; CHB-MIT stores seizure "
              "times in sidecar files and the in-EDF annotations may be empty "
              "until a dataset-specific mapping is added.")
        return 2

    n_seizure = totals["seizure"]
    n_background = totals["background"]
    if n_seizure < es.MIN_SEIZURE_WINDOWS or n_background < es.MIN_BACKGROUND_WINDOWS:
        print(BLOCKER_MSG)
        print(f"Reason: reliability gates not met - need >= "
              f"{es.MIN_SEIZURE_WINDOWS} seizure and >= "
              f"{es.MIN_BACKGROUND_WINDOWS} background windows; got "
              f"{n_seizure} and {n_background}.")
        return 2

    # ---------------------------------------------------------------------
    # Importance, ranking, selection (training data only)
    # ---------------------------------------------------------------------
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)

    scores, info = es.compute_channel_importance(X, y, common_channels)
    if any(not np.isfinite(v) for v in scores.values()):
        print("ERROR: non-finite importance scores produced; refusing to save.")
        return 2

    ranked = es.rank_channels(scores)
    top5 = es.select_top_channels(ranked, 5)
    top4 = es.select_top_channels(ranked, 4)

    csv_path, json_path = es.save_selection_results(
        ranked, top5, top4, es.METHOD_DESCRIPTION,
        extra={
            "training_recordings": len(train_files),
            "training_segments": int(len(y)),
            "seizure_windows": int(n_seizure),
            "background_windows": int(n_background),
            "split": split_desc,
            "max_channel_mean_f": info["max_channel_mean_f"],
        },
    )
    plot_path = es.plot_channel_importance(ranked)

    print(f"Training segments used: {len(y)}")
    print()
    print("TOP 5 CHANNELS:")
    for idx, name in enumerate(top5, start=1):
        print(f"{idx}. {name}")
    print()
    print("TOP 4 CHANNELS:")
    for idx, name in enumerate(top4, start=1):
        print(f"{idx}. {name}")
    print()
    print("Results saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"Plot saved to:")
    print(f"  {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())