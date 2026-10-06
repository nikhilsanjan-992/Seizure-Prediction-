"""NeuroSelect Phase 2 dataset inspection script.

Usage:
    python scripts/inspect_dataset.py

Discovers real EDF/EDF.gz files under data/raw/, inspects the first one
with MNE, prints metadata, runs compatibility checks, and optionally
saves a small inspection plot to results/plots/.

Sets NEUROSELECT_INSPECT_ALL to process every discovered file (metadata
only; no additional plots unless requested).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import eeg_pipeline as ep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSelect dataset inspection")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save a short EEG segment plot for the first file to results/plots/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="inspect metadata of every discovered EDF file (no extra plots)",
    )
    args = parser.parse_args()

    edf_files = ep.discover_edf_files()
    if not edf_files:
        print("NO REAL EDF DATASET FOUND")
        print("-" * 30)
        print("Place at least one real EDF/*.edf.gz recording inside:")
        print(f"  {ep.RAW_DATA_DIR}")
        print("The script will then discover it automatically.")
        print("See README.md ('Phase 2 - Dataset') for recommended datasets.")
        return 1

    print("NEUROSELECT DATASET INSPECTION")
    print("=" * 30)
    ep.print_dataset_summary(edf_files)

    targets = edf_files if args.all else edf_files[:1]

    for path in targets:
        print()
        print(f"Inspecting: {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
        print("-" * 30)
        try:
            raw = ep.load_edf(path, preload=False)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            continue
        except Exception as exc:
            print(f"ERROR: could not load EDF file (corrupted, unsupported, or "
                  f"permission denied): {exc}")
            continue

        try:
            ep.inspect_raw(raw)

            print()
            mapping = ep.get_standardized_channel_names(raw)
            print("  Channel-name normalization helper (original -> normalized):")
            for orig in ep.get_channel_names(raw):
                print(f"    {orig!r} -> {mapping[orig]!r}")

            standard = ep.identify_standard_eeg_channels(raw)
            print(f"  Standard 10-20 electrodes matched: {len(standard)}")

            annotations = ep.get_annotations(raw)
            if annotations:
                print()
                print("  Annotations (first 10):")
                for ann in annotations[:10]:
                    print(f"    onset={ann['onset']:>12.3f} s  dur={ann['duration']:.3f} s  {ann['description']}")

            compat = ep.check_compatibility(raw)
            ep.print_compatibility(compat, n_edf=len(edf_files))

            if args.plot and path == targets[0]:
                out_name = path.stem[:40] + "_segment.png"
                ep.plot_eeg_segment(raw, duration=10.0, save_path=ep.PLOTS_DIR / out_name)
        except Exception as exc:
            print(f"ERROR: metadata inspection failed for {path.name}: {exc}")
            continue
        finally:
            raw.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())