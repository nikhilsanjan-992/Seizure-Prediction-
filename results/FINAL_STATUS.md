# NeuroSelect — Final Project Status

Generated during Phase 10 (final integration / release preparation). This file records the
actual state of the repository. Nothing is invented; missing pieces are stated as missing.

## Research Pipeline

Status: IMPLEMENTED, NOT EXECUTED.
The real-data-only pipeline (EDF discovery → preprocessing → segmentation → leakage-safe
split → training-only electrode ranking → 22/5/4 CNN-BiLSTM training → evaluation → validation
→ Random Forest baseline) is implemented in `src/` and `scripts/` and passes syntax and import
checks. It has hard data-gates and refuses to run without real EEG, so no experiment has been
executed.

## Dataset

Actual dataset: NONE PRESENT.
`data/raw/` is empty (0 EDF / EDF.gz files). No dataset identifiers can be recorded. The project
targets real EEG datasets (e.g., CHB-MIT, TUH) but none have been added in this local setup.

## Electrode Selection

Actual method: mean ANOVA F-statistic of per-channel features (seizure vs background training
windows), normalized to [0, 1]; deterministic; training data only.
This method is implemented (`src/electrode_selection.py`, `scripts/select_electrodes.py`).
A selection run has NOT been executed (no dataset), so `results/metrics/selected_channels.json`
does not exist and no electrode ranking has been produced.

## Selected 5 Channels

NOT AVAILABLE — no selection run, no dataset. Would be read from `selected_channels.json`
(`top_5`).

## Selected 4 Channels

NOT AVAILABLE — no selection run, no dataset. Would be read from `selected_channels.json`
(`top_4`).

## CNN-BiLSTM

Status: IMPLEMENTED, NOT TRAINED.
Shared architecture for 22/5/4 (`src/model.py`): Conv1D(32,k5)→BN→ReLU→MaxPool→Conv1D(64,k3)→
BN→ReLU→BiLSTM(32)→Dropout(0.3)→Dense(1,sigmoid); Adam lr=1e-3, batch 8, BCE, early stopping +
best checkpoint. `scripts/test_model.py` validates building and forward shape. No `.keras`
checkpoints exist (`models/` is empty) because training requires a dataset (Phase 6 gate).

## 22 vs 5 vs 4 Experiment

Status: NOT COMPLETE / NOT RUN.
`scripts/run_experiment.py` implements the comparison with identical settings across montages,
but it requires real EDF files and Phase 4 selection. Neither exists, so there are no results:
`results/metrics/comparison.csv`, `validated_metrics.csv`, `error_analysis*.json`,
`final_research_table.csv` do not exist.

## Validation

Status: PARTIAL (tooling complete; no results to validate).
`src/validation.py` + `scripts/validate_experiment.py` implement leakage audit, metric
re-validation from stored confusion matrices, channel/config consistency, and reproducibility
recording. Run on the current repository it correctly reports BLOCKED /
`NOT_VERIFIABLE` for every item because the Phase 6 experiment was never run
(`results/metrics/leakage_audit.json`).

## Baseline

Status: NOT RUN.
`scripts/run_baseline.py` (Random Forest, same split/windows/normalization/montages, n=200,
class_weight=balanced, seed 42) is implemented and blocks cleanly without a dataset.
`results/metrics/baseline_comparison.csv` has not been written.

## FastAPI

Status: IMPLEMENTED, NOT DATA-VERIFIED.
`backend/main.py` serves `GET /health`, `GET /channels`, `GET /metrics`, and
`POST /predict` (real EDF/EDF.GZ upload + configuration 22/5/4), reusing the src/ pipeline,
reporting only real state, cleaning up uploads, and logging stage timings. It boots
(`python -m uvicorn backend.main:app`) and all endpoints were smoke-tested (200/400/413/422/503
behaviors verified, no stack traces). Live `/predict` requires a dataset, Phase 4 selection,
and trained models; until then the frontend reports the honest unavailable states.

## React

Status: IMPLEMENTED, BUILDS, NOT DATA-VERIFIED.
`frontend/` is a minimal Vite + React app (`src/{main,App,api}.jsx|js`, `index.css`).
Production build passes (`npm run build`); the production preview serves HTTP 200. Because
real models and a dataset do not exist, live data panels could not be exercised and display the honest
"unavailable" states.

## End-to-End Test

Status: NOT TESTED.
A real EDF → React → FastAPI → preprocessing → trained model → prediction → React flow is
impossible until a dataset, Phase 4 selection, and trained models exist. No synthetic EEG was used.

## Reproducibility

Status: PARTIAL.
Configured settings (seed 42; band-pass 0.5–40 Hz; notch 50 Hz; 10 s windows / 5 s overlap;
Adam 1e-3; batch 8; threshold 0.5; architecture) are recorded in
`results/metrics/reproducibility.json` and the README. No dataset identifier or training
runtime can be recorded because nothing was run.

## Known Limitations

- No real EEG dataset has been added, so Phases 4, 6 and 7 could not produce results.
- The FastAPI backend (Phase 8) exists but its `/predict` path is untested on real data
  (no dataset/models currently). Per-recording (not training-set) z-score is used at
  inference because current artifacts do not persist training normalization statistics.
- Phase 6 does not persist per-sample test predictions/scores, so ROC-AUC and paired
  statistical comparisons cannot be re-derived from stored artifacts later.
- The local absolute path present in `results/metrics/leakage_audit.json` /
  `reproducibility.json` is a development-machine path; those files are git-ignored.

## Final Notes

This repository is a complete, honest research scaffold: a real-data-only ML pipeline, a
validation layer, a Random Forest comparator, a FastAPI backend, and a buildable React
dashboard. It is NOT a paper-ready result because no experiment has been executed — there is
no dataset, no trained model, and no metrics. Working software and completed experiments are
therefore distinct here: the software runs; the experiments have not occurred. Results and
claims will only be added after real data is placed in `data/raw/` and Phases 4/6 are executed
and validated.