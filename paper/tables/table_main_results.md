# Table: Main Results — 22 vs 5 vs 4 Channels (CNN-BiLSTM)

Status: **no experiment has been run** (no dataset, no trained models, no electrode
selection). Every cell is therefore N/A. This table is intentionally empty of values; it is
provided so measured results can be inserted verbatim from the Phase 6 output files
(`results/metrics/comparison.csv`, re-validated by `scripts/validate_experiment.py`).

| Configuration | Channels | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22 | 22 | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | 5 | N/A | N/A | N/A | N/A | N/A | N/A |
| 4 | 4 | N/A | N/A | N/A | N/A | N/A | N/A |

Metric definitions (threshold 0.5, test split):

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 = 2 * Precision * Recall / (Precision + Recall)
- ROC-AUC = area under the receiver operating characteristic curve (requires recorded
  per-sample scores)

Zero denominators are reported as NA, never estimated.