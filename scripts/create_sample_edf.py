"""Create sample 22-channel EDF recording files for testing the NeuroSelect UI & Pipeline.

Generates 3 subject recordings (chb01_01.edf, chb02_01.edf, chb03_01.edf) under data/raw/
so that the subject-aware train/val/test split has 1 recording per split.

Usage:
    python scripts/create_sample_edf.py
"""

from pathlib import Path
import numpy as np
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 22 standard EEG channels
CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ", "FPZ",
    "A1", "A2"
]
SFREQ = 256  # 256 Hz
DURATION_SECONDS = 180  # 180 seconds

FILES = ["chb01_01.edf", "chb02_01.edf", "chb03_01.edf"]

def main():
    print("Generating 3 sample subject EEG recordings (chb01, chb02, chb03)...")
    n_samples = SFREQ * DURATION_SECONDS
    t = np.linspace(0, DURATION_SECONDS, n_samples)

    for idx, fname in enumerate(FILES):
        out_file = OUTPUT_DIR / fname
        data = np.zeros((len(CHANNELS), n_samples))
        for i in range(len(CHANNELS)):
            data[i] = (
                0.00005 * np.sin(2 * np.pi * (10 + idx) * t + i) +
                0.00002 * np.sin(2 * np.pi * 20 * t) +
                0.00001 * np.random.randn(n_samples)
            )

        info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        
        annot = mne.Annotations(
            onset=[0.0, 90.0],
            duration=[90.0, 90.0],
            description=["background", "seizure"]
        )
        raw.set_annotations(annot)
        
        mne.export.export_raw(str(out_file), raw, fmt="edf", overwrite=True)
        print(f"  - Saved {fname}")

    print(f"\nSUCCESS: Created 3 subject recordings under:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
