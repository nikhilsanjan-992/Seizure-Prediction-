# NeuroSelect

EEG-Based Seizure Prediction Using Optimal Electrode Selection and CNN-BiLSTM

## Overview

NeuroSelect is a research codebase that evaluates whether a data-driven reduced EEG
electrode configuration (5 or 4 channels) can retain useful seizure-prediction
performance compared with a full 22-channel configuration. It provides a real-data-only
ML pipeline (EDF loading → preprocessing → segmentation → leakage-safe split → data-driven
electrode ranking → CNN-BiLSTM training → evaluation), a validation layer, and a React
dashboard that consumes a FastAPI backend.

The pipeline refuses to generate or use synthetic EEG and refuses to fabricate labels,
metrics, or predictions. Where an artifact is missing, the code and UI report that fact
instead of guessing.

## Research Question

Can a small, data-driven electrode subset (5 or 4 channels) preserve useful
seizure-prediction performance relative to the full 22-channel montage, when both are
trained and evaluated under identical, leakage-safe conditions?

## Research Objective

- Build a leakage-safe training/evaluation pipeline over real EEG recordings.
- Rank electrodes from training data only and select top-5 / top-4 montages.
- Train the same CNN-BiLSTM architecture on 22, 5, and 4 channels.
- Evaluate all configurations on the same untouched test set.
- Report measured differences only; no a-priori claims of sufficiency.

## Methodology

1. **Real EDF loading** — recordings are discovered under `data/raw/` (`*.edf`, `*.edf.gz`);
   nothing is synthetic.
2. **Preprocessing** — band-pass filtering (0.5–40 Hz), optional notch (50 Hz), per-channel
   z-score normalization computed from **training recordings only**.
3. **Segmentation** — sliding windows (10 s, 5 s overlap); windows labeled from the real EDF
   annotations; unlabeled windows are excluded, never guessed.
4. **Leakage-safe split** — subject/session-aware train/val/test split before anything is fit;
   a subject's recordings always stay in one split (session-level fallback for < 3 subjects
   is documented as NOT subject-independent).
5. **Electrode selection (training-only)** — ANOVA F-statistic importance of per-channel
   features (seizure vs background training windows), normalized to [0, 1]; top-5 / top-4.
6. **Training** — one shared CNN-BiLSTM for 22/5/4; identical optimizer, learning rate,
   batch size, class weighting, threshold; only the input channel dimension differs.
7. **Evaluation** — metrics computed on the untouched test split; fixed threshold 0.5;
   undefined denominators are reported as `NA`, never estimated.
8. **Serving** — a FastAPI backend (planned) exposes health / channels / metrics / predict;
   the React frontend displays only real backend values.

## Experimental Design

All three configurations use the same recordings, the same split, and the same CNN-BiLSTM
architecture. Recordings lacking any reduced-montage electrode are dropped from the whole
experiment so every configuration sees identical recordings.

### 22 Channels

The structural EEG montage common across all compatible recordings. Serves as the full
reference configuration.

### 5 Channels

Top-5 electrodes from the data-driven Phase 4 ranking (`selected_channels.json` → `top_5`).

### 4 Channels

Top-4 electrodes from the same ranking (`top_4`). Top-4 is a subset of top-5 by construction.

The labels "reduced configuration" are used for 5 and 4 channels; they are never described
as "better" or "best" — that is determined only by measured results.

## Electrode Selection

Method: mean ANOVA F-statistic of per-channel features (mean, std, variance, RMS, line length,
Hjorth mobility/complexity, relative delta/theta/alpha/beta band power) computed on training
windows, normalized to [0, 1]. Deterministic, no random sampling; `RANDOM_SEED = 42`.

Electrode selection uses the training subset only; the hold-out subset is never used to rank
channels.

## CNN-BiLSTM Architecture

```
Input (timesteps, num_channels)
  -> Conv1D(32, kernel 5) -> BatchNorm -> ReLU -> MaxPool
  -> Conv1D(64, kernel 3) -> BatchNorm -> ReLU
  -> BiLSTM(32)
  -> Dropout(0.3)
  -> Dense(1, sigmoid)
```

- Loss: binary cross-entropy. Optimizer: Adam (lr 1e-3). Batch size: 8.
- Class weights: balanced, computed from **training labels only**.
- Callbacks: EarlyStopping (monitor `val_loss`, restore best weights) + ModelCheckpoint (best).
- One `build_cnn_bilstm()` is used for all montages; only `num_channels` changes.
- Input contract `(batch, timesteps, channels)`; output `(batch, 1)` sigmoid.

## Dataset

Current local status: **no dataset is present.** `data/raw/` is empty (0 EDF/EDF.gz files).

The pipeline is designed for real EDF EEG datasets. Recommended candidates for academic use:

- **CHB-MIT Scalp EEG Database (PhysioNet)** — per-patient EDF recordings; note that CHB-MIT
  stores seizure times in sidecar files, which is not implemented for in-EDF annotation
  extraction.
- **TUH EEG (Temple University Hospital) Corpus** — subsets freely available.

Required characteristics for the planned experiments: >= 4 EEG channels, ~22 channels for the
full montage, a usable sampling frequency, and enough duration for labeled windows.

No dataset details are invented here; nothing has been trained or evaluated in this repository.

## Preprocessing

- FIR band-pass 0.5–40 Hz (validated against Nyquist).
- Optional notch 50 Hz (60 Hz for US mains).
- Per-channel z-score normalization: statistics fitted on training recordings, applied to
  validation/test.
- Original channel names preserved; non-EEG channels ignored; missing channels reported.
- Segmentation: 10 s windows, 5 s overlap; only full windows emitted.

## Evaluation Metrics

Binary classification (seizure = 1, background = 0), computed on the untouched test set with a
fixed 0.5 threshold:

- Accuracy, Precision, Recall/Sensitivity, Specificity, F1
- ROC-AUC (when both classes exist and scores are available)

Zero denominators → `NA`. Unverifiable values are reported as `NOT_VERIFIABLE`, never invented.

## Results

**No experimental results exist yet.** Phases 4 and 6 require a real EDF dataset, which has not
been added in this local setup. Consequently:

- no `selected_channels.json` (no channel selection run)
- no `models/*.keras` (no trained models)
- no `comparison.csv` / `validated_metrics.csv` (no Phase 6 results)
- `results/metrics/leakage_audit.json` records every leakage item as `NOT_VERIFIABLE`
- `results/metrics/research_summary.md` documents the blocked state without invented numbers

The repository will report 22/5/4 results only after a real dataset is placed in `data/raw/`
and Phases 4 and 6 are executed. The frontend likewise shows `Backend unavailable.` /
`Metrics unavailable.` until real artifacts exist.

## Project Structure

```
NeuroSelect/
├── src/                 # Core pipeline (config, eeg_pipeline, preprocessing,
│                        #   electrode_selection, model, train, evaluate, dataset,
│                        #   validation)
├── scripts/             # CLI runners: check_setup, inspect_dataset, test_preprocessing,
│                        #   select_electrodes, test_model, run_experiment, run_baseline,
│                        #   validate_experiment
├── backend/             # FastAPI backend (main.py: /health, /channels, /metrics, /predict)
├── frontend/            # React + Vite dashboard (src/{main,App,api}.jsx|js, index.css)
├── data/
│   ├── raw/             # Real EDF recordings (git-ignored, empty)
│   └── processed/       # Preprocessed/segmented arrays (git-ignored, empty)
├── models/              # Trained model checkpoints (git-ignored, empty)
├── results/
│   ├── metrics/         # Research artifacts (git-ignored)
│   └── plots/           # Figures (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

Python 3.12.

```bash
python -m venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
python scripts/check_setup.py
```

Frontend:

```bash
cd frontend
npm install
```

## Running the Backend

The FastAPI backend (`backend/main.py`) reuses the src/ pipeline and reports only real
state. It exposes `GET /health`, `GET /channels`, `GET /metrics`, and
`POST /predict` (real EDF upload + configuration). Start it from the repository root:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The frontend dev server defaults to `http://localhost:8000` (see `frontend/.env.example`,
`VITE_API_URL`); CORS is configured for the Vite dev origins (5173). Inference requires the
trained Phase 6 models under `models/` and the Phase 4 `selected_channels.json`; until those
exist the endpoints report the availability honestly and `/predict` returns 503. Uploads are
written to a temporary directory and removed after the request.

## Running the Frontend

```bash
cd frontend
npm run dev
```

Production build:

```bash
cd frontend
npm run build
```

Environment: copy `frontend/.env.example` to `frontend/.env` (root `.env.example` also
documents it). Never commit `.env`.

```
VITE_API_URL=http://localhost:8000
```

## API Endpoints

Contract implemented by the backend (`backend/main.py`) and consumed by the frontend:

| Method | Path       | Purpose / expected response                                  |
|--------|------------|--------------------------------------------------------------|
| GET    | `/health`  | `{"status":"ok","model_ready":bool,"channels_ready":bool,"metrics_ready":bool}` |
| GET    | `/channels`| `{"available":true,"top_5":[...],"top_4":[...]}`             |
| GET    | `/metrics` | `{"available":true,"rows":[{configuration,channels,accuracy,precision,recall,specificity,f1,roc_auc,test_samples}, ...]}` |
| POST   | `/predict` | multipart `file` (EDF) + `configuration` (`22`/`5`/`4`) → `{"prediction":"seizure"\|"non_seizure","probability":<0..1>,"segments_analyzed":int,"channels_used":[...],"configuration":"5","model":"CNN-BiLSTM"}` |

## Reproducibility

- Python 3.12; fixed seeds: `RANDOM_SEED = 42` (Python/NumPy/TensorFlow).
- Preprocessing: band-pass 0.5–40 Hz, notch 50 Hz, per-channel z-score (training only).
- Segmentation: 10 s windows, 5 s overlap.
- Model/config: see `src/model.py`, `src/train.py`, `src/dataset.py`.
- Dataset placement: `data/raw/` (accepted: `*.edf`, `*.edf.gz`).

Commands to reproduce the full lifecycle once a dataset exists:

```bash
python scripts/select_electrodes.py       # Phase 4 — training-only electrode ranking
python scripts/run_experiment.py          # Phase 6 — 22/5/4 CNN-BiLSTM training + evaluation
python scripts/run_baseline.py            # Phase 7 — Random Forest comparator
python scripts/validate_experiment.py     # Phase 7 — leakage + metric validation
```

A machine-readable record of the configured settings is produced by
`scripts/validate_experiment.py` (`results/metrics/reproducibility.json`).

## Limitations

- No real EEG dataset has been added locally, so no experiment has been run and no results
  exist. The pipeline is implemented and validated only up to its data-gates.
- The FastAPI backend (Phase 8) is implemented, but until a dataset, the Phase 4 selection,
  and trained models exist it reports availability honestly, the frontend shows
  "Backend unavailable." / "Metrics unavailable.", and `/predict` returns 503.
- Even after a Phase 6 run, per-sample test predictions/scores are not persisted, so ROC-AUC
  and paired statistical comparisons cannot be re-derived from stored artifacts.
- The 10 s/5 s window experiment is one configuration; results would generalize to other
  parameterizations only with further experiments.
- This project is a research prototype; any future quantitative claims must come from the
  actual validated runs, which have not occurred yet.

## Research Disclaimer

This project is a research prototype and is not a clinical diagnostic device.

## Future Work

- Add a real EDF dataset and execute Phases 4, 6 and 7 to obtain measured 22/5/4 results.
- Persist per-sample predictions at Phase 6 to enable reproducible ROC/statistical checks.
- Optional ablation (CNN-only vs BiLSTM-only) and seeded repeated runs (42/52/62) with
  mean ± std reporting. (None were forced; the project supports them once data exists.)
- GitHub release preparation after results and backend integration have been reviewed.