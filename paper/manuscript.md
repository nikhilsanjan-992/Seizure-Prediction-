# EEG-Based Seizure Prediction Using Optimal Electrode Selection and CNN-BiLSTM

Draft manuscript for the NeuroSelect research software. This manuscript describes the
implemented methodology and the experimental design. The Results sections report the true
state of the experimental record; no experiment has been executed and no results have been
produced. Placeholder values are deliberately absent.

## Abstract

The practical burden of multi-channel EEG recording motivates research into whether a
small, data-driven subset of electrodes can support seizure prediction. This study designs
a leakage-safe pipeline that loads real EDF recordings, preprocesses and segments them, ranks
electrodes from training data only (mean ANOVA F-statistic of per-channel features), and
evaluates three montages — 22, 5, and 4 channels — using the same CNN-BiLSTM architecture.
The full pipeline and validation layer were implemented and verified at the software level.

No experimental findings are reported here: at the time of writing, no real EEG dataset has
been added to the project, so the experiments (electrode selection, model training, and the
22 vs 5 vs 4 comparison) have not been executed, and no metrics exist. The primary limitation
is therefore the absence of data and results. A conclusion about channel reduction cannot yet
be drawn; the design only ensures that, once data is available, the comparison can be made
without leakage and without fabricated artifacts.

## 1. Introduction

EEG-based seizure prediction aims to identify pre-seizure brain states so that warnings can
be issued before clinical onset. Full-scalp EEG, however, is impractical for long-term,
ambulatory, and wearable use: the montage is large, the instrumentation is heavy, and
placement is demanding. Reducing the number of electrodes while preserving predictive
information could lower these barriers, but which electrodes to keep is an open question.

Deep models such as convolutional and recurrent networks can learn directly from raw or
lightly processed EEG. A CNN-BiLSTM combination is a common design choice for seizure-related
EEG because convolutional layers extract local spectral-temporal patterns and a bidirectional
recurrent layer models temporal context in both directions.

The research gap addressed here is not a claim of novelty in architecture or electrode
selection, but the construction of a reproducible, leakage-safe experimental harness that
answers one specific question with real data: do data-driven top-5 and top-4 montages retain
seizure-prediction performance relative to a full 22-channel montage under the same model,
split, and evaluation protocol? At present the harness exists; the question has not yet been
answered experimentally.

## 2. Research Objective

The exact objective is to evaluate, on real EDF EEG recordings, the seizure-prediction
performance of

- a 22-channel configuration (structural montage common to compatible recordings),
- a 5-channel configuration (data-derived top-5 electrodes), and
- a 4-channel configuration (data-derived top-4 electrodes),

using the same CNN-BiLSTM architecture, the same train/validation/test split, the same
training protocol, and the same evaluation metrics. The objective does not assume that
reduced montages are sufficient; that is the question to be answered by measurement.

## 3. Dataset

- Dataset name: Not reported in the current experimental record.
- Source: Not reported in the current experimental record.
- Number of subjects: Not reported in the current experimental record.
- Number of recordings: Not reported in the current experimental record.
- Sampling frequency: Not reported in the current experimental record.
- EEG channels: Not reported in the current experimental record.
- Annotation information: Not reported in the current experimental record.
- Seizure / non-seizure classes: Designed as a binary problem (seizure = 1, background = 0);
  labels are derived from real EDF annotations and unlabeled windows are excluded, never
  guessed. Actual class distributions: Not reported in the current experimental record.

The repository's `data/raw/` directory contains no EDF/EDF.gz files. The pipeline is
designed for real EEG datasets of the CHB-MIT or TUH type, but no dataset has been installed
or evaluated in this project.

## 4. Methodology

The implemented pipeline is:

```
Real EDF recordings (data/raw/)
 -> loading and compatibility check (src/eeg_pipeline.py)
 -> preprocessing: FIR band-pass 0.5-40 Hz, optional 50 Hz notch,
    per-channel z-score normalization (training statistics only)
 -> segmentation: sliding windows, 10 s, 5 s overlap of full windows
 -> subject-aware leakage-safe train/val/test split (0.7 / 0.15 / 0.15)
 -> electrode selection: training-only mean ANOVA F ranking (Phase 4)
 -> channel configurations: 22 (common montage) / 5 (top-5) / 4 (top-4)
 -> CNN-BiLSTM training per configuration (same architecture)
 -> evaluation on the untouched test set (threshold 0.5)
 -> validation layer: leakage audit + metric re-validation
```

The application layer (backend/frontend) has no influence on the research pipeline; the
pipeline is driven by the CLI runners under `scripts/`.

## 5. EEG Preprocessing

As implemented in `src/preprocessing.py` and `src/eeg_pipeline.py`:

- Filtering: FIR band-pass, low 0.5 Hz, high 40 Hz; filter parameters are validated against
  the recording's Nyquist frequency, and an incompatible configuration raises an error rather
  than silently corrupting data.
- Notch: optional, default 50 Hz (60 Hz available for US mains).
- Normalization: per-channel z-score, with the normalization statistics computed from
  training recordings only and applied identically to validation and test recordings.
- Segmentation: sliding windows of 10 s with 5 s overlap; only full windows are emitted.
- Channel handling: original channel names are preserved; channel labels are normalized for
  matching against the planned montage; missing channels are reported; non-EEG channels are
  ignored; recordings lacking required reduced-montage electrodes are dropped from the whole
  experiment (so all configurations see identical recordings).
- Orientation conversion: the model consumes `(batch, timesteps, channels)`; conversion is
  explicit and validated, never performed implicitly inside the model.

## 6. Optimal Electrode Selection

The Phase 4 selection procedure, as implemented in `src/electrode_selection.py`:

- Feature extraction: for each window and each channel, eleven features are computed —
  mean, standard deviation, variance, RMS, line length, Hjorth mobility, Hjorth complexity,
  and relative band power in delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta
  (13-30 Hz); Welch spectra use nperseg = 256 and 50% overlap.
- Importance: for every channel, the mean ANOVA F-statistic across its features is computed
  contrasting seizure vs background training windows, then normalized to [0, 1] by the
  channel with the largest mean F.
- Training-only: the statistics are computed from the training split exclusively.
- Ranking and selection: channels are ranked by normalized importance; the top 5 and top 4
  are recorded as `top_5` and `top_4` (top-4 is a subset of top-5 by construction).

No selection run has been executed in this project (no dataset), so no channels have yet
been ranked. Reported selected channels, once available, should be described as the
top-ranked channels under the proposed selection procedure for the training data of this
study, not as universally optimal electrodes.

## 7. CNN-BiLSTM Architecture

As implemented in `src/model.py`: one shared architecture is used for all three montages;
only the channel input dimension changes.

```
Input (timesteps=2560, num_channels)     # 10 s @ 256 Hz design default
 -> Conv1D(32, kernel 5) -> BatchNorm -> ReLU -> MaxPool
 -> Conv1D(64, kernel 3) -> BatchNorm -> ReLU
 -> BiLSTM(32)
 -> Dropout(0.3)
 -> Dense(1, sigmoid)
```

- Input: `(batch, timesteps, channels)`; output: `(batch, 1)` sigmoid (seizure probability).
- Filters (32, 64); kernels (5, 3); LSTM units 32; dropout 0.3.
- Loss: binary cross-entropy; optimizer Adam at learning rate 1e-3; batch size 8.
- Callbacks: EarlyStopping (monitor `val_loss`, patience 5, restore best weights) and
  ModelCheckpoint (save best only).
- Random seed 42 is set across Python, NumPy, and TensorFlow.
- Number of parameters: not recorded — no model has been trained in this project.

## 8. Experimental Design

- 22-channel configuration: structural montage common across all compatible recordings
  (at least 4 common channels required to proceed).
- 5-channel configuration: `top_5` from Phase 4 selection.
- 4-channel configuration: `top_4` from Phase 4 selection.
- All configurations share the same architecture, seed, split, normalization, optimizer,
  learning rate, batch size, class weighting, and classification threshold 0.5.
- Class weights: balanced, computed from training labels only.
- Evaluation: on the untouched test split; `scripts/run_experiment.py` writes
  `comparison.csv`, per-configuration result JSON, confusion-matrix plots, and training
  curves under `results/metrics/` and `results/plots/`.
- Leakage prevention: subject-aware splitting (a subject's recordings never cross splits;
  session-level fallback for fewer than 3 subjects is documented as not
  subject-independent); electrode selection on training data only; normalization and
  class-weight statistics on training data only; fixed test-set threshold.

## 9. Evaluation Metrics

Computed on test predictions with a fixed 0.5 threshold:

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall (Sensitivity) = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 = 2 * Precision * Recall / (Precision + Recall)
- ROC-AUC: area under the ROC curve using predicted probabilities (available only when both
  classes are present and per-sample scores are recorded).

Zero denominators are reported as NA, never estimated. The validation layer re-computes
confusion-matrix-derived metrics from stored matrices and reports `NOT_VERIFIABLE` for any
value that cannot be reproduced from artifacts.

## 10. Results

No experimental results exist in the current record. The Phase 6 experiment writes nothing
without a dataset, so no `comparison.csv`, `validated_metrics.csv`,
`final_research_table.csv`, or `selected_channels.json` exists. The table below reports the
true state:

| Configuration | Channels | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22 | 22 | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 5 | N/A | N/A | N/A | N/A | N/A | N/A |
| 4 | 4 | N/A | N/A | N/A | N/A | N/A | N/A |

N/A means the value was not produced because the experiment was not run. There are no
trained models (`models/` is empty) and no electrode ranking. The leakage audit
(`results/metrics/leakage_audit.json`) records every audited item as NOT_VERIFIABLE for this
reason, and `results/metrics/research_summary.md` documents the blocked state without
invented numbers.

## 11. Baseline Comparison

Baseline comparison was not completed. A Random Forest comparator is implemented in
`scripts/run_baseline.py` with the same split, windows, training-only normalization, and
22/5/4 montages, using per-window mean/std/RMS/peak-to-peak features; it writes
`baseline_comparison.csv` only from real test results, so no baseline values exist.

## 12. Ablation Study

Ablation experiments were not completed. The shared CNN-BiLSTM has no recorded CNN-only or
BiLSTM-only runs in this project.

## 13. Robustness Analysis

Repeated-run validation was not completed. No repeated runs exist beyond the configured
random seed 42, and per-sample test predictions are not persisted, so means and standard
deviations across seeds cannot be reported.

## 14. Error Analysis

No error analysis is possible in the current record: there are no confusion matrices,
false positives, false negatives, or misclassification patterns to inspect. The Phase 6
evaluation is designed to write confusion-matrix plots, but none have been produced. No
clinical claims can be or are made.

## 15. Discussion

Because no experiment has been run, there is nothing to interpret about actual performance:
the effect of reducing channels, the direction and size of any performance differences, the
computational benefits, and electrode-selection behavior all remain to be measured. The
design ensures that once data is available the three montages will be compared under
identical, leakage-safe conditions, and the reported wording will follow the evidence: a
decrease, equality, or increase in performance with fewer channels will be described
accordingly, without a default presumption that reduced montages are sufficient. Baseline
and robustness assessments also remain pending.

## 16. Limitations

Actual limitations of the current project:

- Dataset size and subjects: no dataset is installed, so no subject count or recording count
  can be reported and no experiment results exist to validate.
- Recording variability: unassessed (no data).
- Computational constraints: the design targets a moderate-CPU machine (memory-friendly
  batching); actual training cost is unmeasured.
- Single-dataset evaluation: the comparison is intended for one dataset; generalizability
  across datasets is unaddressed.
- External validation: not performed and not planned in the current pipeline.
- Repeated runs: the design supports seeds, but only seed 42 is configured and no repeated
  runs have been executed.
- Artifact limitations: Phase 6 does not persist per-sample test scores, so ROC-AUC and
  paired statistical comparisons cannot later be re-derived from stored artifacts.
- No claims of clinical or real-time readiness are made.

## 17. Conclusion

No quantitative conclusion can be drawn from this record because the experiments have not
been executed: the project contains no dataset, no trained model, and no evaluated metrics.
The implemented methodology provides a leakage-safe and reproducible basis for running the
22 vs 5 vs 4 comparison on real EEG. Reaching a scientific conclusion about electrode
reduction requires adding a real dataset and executing Phases 4 and 6; until then, nothing
supports-and nothing contradicts-the usefulness of reduced montages. This work makes no
clinical claim and is not ready for clinical deployment.

## 18. Future Work

Relevant directions only, pending experimental evidence:

- Install a real EEG dataset and execute electrode selection, training, and evaluation to
  obtain the missing 22/5/4 results.
- Multi-dataset and larger-cohort validation.
- External validation on an independent dataset.
- Adaptive/patient-specific electrode selection.
- Lightweight edge deployment and real-time optimization.
- Prospective validation of any future predictive claims.
- Persist per-sample predictions to enable reproducible ROC and paired statistics.