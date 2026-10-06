# Reproducibility — NeuroSelect

Machine-readable record: `results/metrics/reproducibility.json` (produced by
`scripts/validate_experiment.py`, status `NOT_RUN - NO EXPERIMENT RESULTS EXIST`).

## Software environment

- Python 3.12.10 (Windows 11) — recorded in `reproducibility.json`
- Installed dependency versions recorded: mne 1.13.2, numpy 2.5.3, scipy 1.18.1,
  pandas 3.0.6, matplotlib 3.11.2, scikit-learn 1.9.1, tensorflow 2.21.0
- Full dependency list: `requirements.txt` (mne, numpy, scipy, pandas, matplotlib,
  scikit-learn, tensorflow, fastapi, uvicorn, python-multipart, python-dotenv)
- Frontend: React + Vite (see `frontend/package.json`); `npm ci` after install

## Dataset placement

- Location: `data/raw/`
- Accepted files: `*.edf`, `*.edf.gz`
- Current state: empty (0 files). No dataset has been installed.

## Preprocessing configuration

- FIR band-pass: 0.5-40 Hz (validated against Nyquist)
- Notch filter: 50 Hz (60 Hz option)
- Normalization: per-channel z-score, statistics from training recordings only

## Segmentation parameters

- Window: 10 s
- Overlap: 5 s
- Only full windows emitted; windows labeled from real EDF annotations; unlabeled windows
  excluded

## Random seed

- 42 (Python, NumPy, TensorFlow via `model.set_reproducible_seed` / `train.set_training_seed`)

## Electrode-selection method

- Mean ANOVA F-statistic of 11 per-channel features (mean, std, variance, RMS, line length,
  Hjorth mobility, Hjorth complexity, relative delta/theta/alpha/beta band power), seizure vs
  background training windows; channel score normalized to [0, 1]; training data only.
- Runner: `python scripts/select_electrodes.py`
- Output: `results/metrics/selected_channels.json` (`top_5`, `top_4`)

## Model architecture

- Shared CNN-BiLSTM: Input(2560, channels) -> Conv1D(32,k5) -> BN -> ReLU -> MaxPool ->
  Conv1D(64,k3) -> BN -> ReLU -> BiLSTM(32) -> Dropout(0.3) -> Dense(1, sigmoid)
- Input `(batch, timesteps, channels)`; output `(batch, 1)` — explicit orientation,
  validated (`src/model.py`)

## Training parameters

- Loss: binary cross-entropy; Optimizer: Adam; Learning rate: 1e-3; Batch size: 8
- Max epochs: Phase 6 first-run default 10 (`--epochs`), configurable; library default 20
- EarlyStopping: monitor `val_loss`, patience 5, restore best weights
- ModelCheckpoint: best weights only, saved to `models/`
- Class weights: balanced, from training labels only

## Evaluation procedure

- Test split only; threshold fixed at 0.5
- Metrics: accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC (when
  per-sample scores exist); zero denominators -> NA
- Confusion-matrix-derived metrics re-derived by the validation layer
- Split: subject-aware train/val/test 0.7 / 0.15 / 0.15 (session-level fallback for fewer
  than 3 subjects; documented as not subject-independent)

## Commands to reproduce the experiment

```bash
# 0. Environment
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1   /   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py            # Phase 1 setup verification

# 1. Place real EDF files in data/raw/

# 2. Inspect dataset compatibility
python scripts/inspect_dataset.py

# 3. Phase 4 — training-only electrode selection
python scripts/select_electrodes.py       # -> results/metrics/selected_channels.json

# 4. Phase 6 — 22/5/4 CNN-BiLSTM training + evaluation
python scripts/run_experiment.py          # -> results/metrics/*, results/plots/*

# 5. Phase 7 — Random Forest baseline (optional comparator)
python scripts/run_baseline.py            # -> baseline_comparison.csv

# 6. Phase 7 — validation: leakage audit + metric re-validation
python scripts/validate_experiment.py     # exit 0 = pass, 1 = blocked, 2 = inconsistent
```

## Current reproducibility status

Partial: configuration and commands are fully specified and recorded, but no experiment has
been executed, so no dataset identifier, training time, or result values exist to
reproduce. Exact bit-level training reproducibility depends on hardware/software and the
TensorFlow backend (noted in `model.set_reproducible_seed`).